"""Wrapper around `imapclient` providing guardrails for MailAI."""
from __future__ import annotations

import contextlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Optional, Set

from ..config.loader import get_runtime_config

try:  # pragma: no cover - optional dependency
    from imapclient import IMAPClient
except ImportError:  # pragma: no cover - fallback for test environment
    IMAPClient = None  # type: ignore[assignment]


@dataclass
class ImapConfig:
    """Connection parameters for an IMAP server."""

    host: str
    username: str
    password: str
    port: int = 993
    ssl: bool = True
    folder: Optional[str] = None
    control_namespace: Optional[str] = None
    quarantine_subfolder: Optional[str] = None

    def __post_init__(self) -> None:
        settings = get_runtime_config()
        if self.folder is None:
            self.folder = settings.imap.default_mailbox
        if self.control_namespace is None:
            self.control_namespace = settings.imap.control_namespace
        if self.quarantine_subfolder is None:
            self.quarantine_subfolder = settings.imap.quarantine_subfolder


class MailAIImapClient:
    """Context manager providing a constrained IMAP client."""

    def __init__(self, config: ImapConfig):
        self._config = config
        self._client: Optional[IMAPClient] = None
        self._delimiter: str = "/"
        self._mailboxes: Set[str] = set()
        self._selected: Optional[str] = None
        self._control_mailbox: Optional[str] = None
        self._quarantine_mailbox: Optional[str] = None
        self._actions: Deque[float] = deque()

    def __enter__(self) -> "MailAIImapClient":
        if IMAPClient is None:
            raise RuntimeError("imapclient dependency is not available")
        self._client = IMAPClient(self._config.host, port=self._config.port, ssl=self._config.ssl)
        self._client.login(self._config.username, self._config.password)
        self._refresh_mailboxes()
        self._bootstrap_control_mailboxes()
        if self._config.folder is None:
            raise RuntimeError("Default mailbox not configured")
        self._select(self._config.folder)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._client is None:
            return
        try:
            self._client.logout()
        finally:
            self._client = None

    @property
    def client(self) -> IMAPClient:
        if self._client is None:
            raise RuntimeError("IMAP client not connected")
        return self._client

    @property
    def config(self) -> ImapConfig:
        return self._config

    @property
    def control_mailbox(self) -> str:
        if self._control_mailbox is None:
            raise RuntimeError("Control mailbox not initialised")
        return self._control_mailbox

    @property
    def quarantine_mailbox(self) -> str:
        if self._quarantine_mailbox is None:
            raise RuntimeError("Quarantine mailbox not initialised")
        return self._quarantine_mailbox

    def _select(self, mailbox: str, *, readonly: bool = False) -> None:
        mailbox_name = self._ensure_mailbox(mailbox)
        self.client.select_folder(mailbox_name, readonly=readonly)
        self._selected = mailbox_name

    def _refresh_mailboxes(self) -> None:
        self._mailboxes.clear()
        for flags, delimiter, name in self.client.list_folders():
            if delimiter:
                decoded = delimiter.decode() if isinstance(delimiter, bytes) else str(delimiter)
                if decoded:
                    self._delimiter = decoded
            decoded_name = name.decode() if isinstance(name, bytes) else str(name)
            self._mailboxes.add(decoded_name)

    def _normalize_path(self, *parts: str) -> str:
        delimiter = self._delimiter or "/"
        segments: List[str] = []
        for part in parts:
            candidate = part.replace("/", delimiter).replace(".", delimiter)
            for chunk in candidate.split(delimiter):
                chunk = chunk.strip()
                if chunk:
                    segments.append(chunk)
        return delimiter.join(segments)

    def _ensure_mailbox(self, mailbox: str) -> str:
        normalized = self._normalize_path(mailbox)
        if normalized not in self._mailboxes:
            try:
                self.client.create_folder(normalized)
            except Exception:  # pragma: no cover - depends on server implementation
                self._refresh_mailboxes()
            else:
                self._mailboxes.add(normalized)
        return normalized

    def _bootstrap_control_mailboxes(self) -> None:
        if self._config.control_namespace is None or self._config.quarantine_subfolder is None:
            raise RuntimeError("Control namespace not configured")
        control = self._normalize_path(self._config.control_namespace)
        quarantine = self._normalize_path(
            self._config.control_namespace, self._config.quarantine_subfolder
        )
        self._control_mailbox = self._ensure_mailbox(control)
        self._quarantine_mailbox = self._ensure_mailbox(quarantine)
        if self._config.folder is not None:
            self._ensure_mailbox(self._config.folder)

    def _throttle(self) -> None:
        now = time.monotonic()
        while self._actions and now - self._actions[0] > 60:
            self._actions.popleft()
        if len(self._actions) >= 500:
            raise RuntimeError("IMAP action rate limit exceeded")
        self._actions.append(now)

    @contextlib.contextmanager
    def control_session(self, *, readonly: bool = False) -> Iterator[str]:
        previous = self._selected
        self._select(self.control_mailbox, readonly=readonly)
        try:
            yield self.control_mailbox
        finally:
            if previous:
                self._select(previous, readonly=False)

    @contextlib.contextmanager
    def session(self, mailbox: str, *, readonly: bool = False) -> Iterator[str]:
        """Temporarily select an arbitrary mailbox."""

        previous = self._selected
        self._select(mailbox, readonly=readonly)
        try:
            yield self._selected or mailbox
        finally:
            if previous:
                self._select(previous, readonly=False)

    def fetch_headers(self, uids: Iterable[int], data: str = "BODY.PEEK[HEADER]") -> dict:
        return self.client.fetch(uids, data)

    def move(self, uid: int, destination: str) -> None:
        dest = self._ensure_mailbox(destination)
        self._throttle()
        self.client.move(uid, dest)

    def copy(self, uid: int, destination: str) -> None:
        dest = self._ensure_mailbox(destination)
        self._throttle()
        self.client.copy(uid, dest)

    def add_label(self, uid: int, label: str) -> None:
        self._throttle()
        self.client.add_gmail_labels(uid, [label])

    def mark_read(self, uid: int, read: bool) -> None:
        self._throttle()
        if read:
            self.client.add_flags(uid, ["\\Seen"])
        else:
            self.client.remove_flags(uid, ["\\Seen"])

    def add_flag(self, uid: int, flag: str) -> None:
        self._throttle()
        self.client.add_flags(uid, [flag])

    def set_header(self, uid: int, name: str, value: str) -> None:
        patch = f"{name}: {value}\r\n"
        self._throttle()
        self.client.append(self._selected or self._config.folder, patch.encode("utf-8"))

    def uid_search(self, criteria: List[object]) -> List[int]:
        return list(self.client.search(criteria))

    def get_rules_email(self, subject: str) -> Optional[int]:
        with self.control_session(readonly=True):
            uids = self.client.search(["SUBJECT", subject])
        return max(uids) if uids else None

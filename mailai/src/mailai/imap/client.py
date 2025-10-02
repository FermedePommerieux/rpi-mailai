"""Wrapper around `imapclient` providing guardrails for MailAI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

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
    folder: str = "INBOX"


class MailAIImapClient:
    """Context manager providing a constrained IMAP client."""

    def __init__(self, config: ImapConfig):
        self._config = config
        self._client: Optional[IMAPClient] = None

    def __enter__(self) -> "MailAIImapClient":
        if IMAPClient is None:
            raise RuntimeError("imapclient dependency is not available")
        self._client = IMAPClient(self._config.host, port=self._config.port, ssl=self._config.ssl)
        self._client.login(self._config.username, self._config.password)
        self._client.select_folder(self._config.folder, readonly=False)
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

    def fetch_headers(self, uids: Iterable[int], data: str = "BODY.PEEK[HEADER]") -> dict:
        return self.client.fetch(uids, data)

    def move(self, uid: int, destination: str) -> None:
        self.client.move(uid, destination)

    def copy(self, uid: int, destination: str) -> None:
        self.client.copy(uid, destination)

    def add_label(self, uid: int, label: str) -> None:
        self.client.add_gmail_labels(uid, [label])

    def mark_read(self, uid: int, read: bool) -> None:
        if read:
            self.client.add_flags(uid, ["\\Seen"])
        else:
            self.client.remove_flags(uid, ["\\Seen"])

    def add_flag(self, uid: int, flag: str) -> None:
        self.client.add_flags(uid, [flag])

    def set_header(self, uid: int, name: str, value: str) -> None:
        patch = f"{name}: {value}\r\n"
        self.client.append(self._config.folder, patch.encode("utf-8"))

    def uid_search(self, criteria: List[object]) -> List[int]:
        return list(self.client.search(criteria))

    def get_rules_email(self, subject: str) -> Optional[int]:
        uids = self.client.search(["SUBJECT", subject])
        return max(uids) if uids else None

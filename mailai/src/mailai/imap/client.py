"""Stateful IMAP client with MailAI-specific guardrails.

What:
  Wrap the third-party ``imapclient`` library with configuration defaults,
  mailbox bootstrap logic, rate limiting, and context managers tailored to the
  MailAI runtime.

Why:
  Direct use of ``imapclient`` exposes sharp edges—mailbox delimiter quirks,
  accidental sequence-number operations, and unbounded command rates. Centralised
  guardrails keep IMAP usage predictable and auditable across rules, watchers,
  and status updaters.

How:
  Loads defaults from the runtime configuration, ensures required mailboxes
  exist, normalises folder names, and tracks operation timestamps to enforce a
  500-actions-per-minute limit. Provides helper context managers for switching
  mailboxes while automatically restoring the previous selection.

Interfaces:
  :class:`ImapConfig` and :class:`MailAIImapClient` plus the public methods on
  :class:`MailAIImapClient` (``session``, ``control_session``, ``move`` etc.).

Invariants & Safety:
  - All operations run in UID-first mode; sequence-number methods are avoided.
  - Rate limiting prevents abusive behaviour that might trigger provider bans.
  - Control and quarantine mailboxes are created upfront to guarantee idempotent
    configuration management.
"""
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
    """Connection parameters and folder hints for an IMAP server.

    What:
      Captures host credentials and optional mailbox overrides required to
      connect to a MailAI-managed IMAP account.

    Why:
      A strongly-typed configuration object makes it clear which fields can be
      overridden while allowing defaults to be injected from the runtime config
      when absent.

    How:
      Dataclass fields mirror the configuration schema; :meth:`__post_init__`
      fills in missing values via :func:`mailai.config.loader.get_runtime_config`.

    Attributes:
      host: IMAP hostname.
      username: Login credential.
      password: Password or app-specific token.
      port: IMAP port (defaults to 993).
      ssl: Whether to use TLS.
      folder: Default mailbox for rule processing.
      control_namespace: Namespace hosting control/status folders.
      quarantine_subfolder: Folder used for quarantined messages.
    """

    host: str
    username: str
    password: str
    port: int = 993
    ssl: bool = True
    folder: Optional[str] = None
    control_namespace: Optional[str] = None
    quarantine_subfolder: Optional[str] = None

    def __post_init__(self) -> None:
        """Populate optional fields from the runtime configuration defaults.

        What:
          Resolves unset optional attributes (folder locations and namespaces)
          using the globally cached runtime configuration.

        Why:
          Keeps configuration files concise—operators can omit values that match
          project defaults while still receiving explicit strings in the client.

        How:
          Reads :func:`get_runtime_config` once per instance and fills empty
          attributes in place.
        """

        settings = get_runtime_config()
        if self.folder is None:
            self.folder = settings.imap.default_mailbox
        if self.control_namespace is None:
            self.control_namespace = settings.imap.control_namespace
        if self.quarantine_subfolder is None:
            self.quarantine_subfolder = settings.imap.quarantine_subfolder


class MailAIImapClient:
    """Context manager exposing a rate-limited IMAP workflow.

    What:
      Owns a single ``imapclient.IMAPClient`` connection and mediates mailbox
      selection, control namespace bootstrap, and UID-based operations.

    Why:
      Ensures that every IMAP interaction respects provider rate limits, uses
      consistent mailbox naming, and recovers cleanly from context switches.

    How:
      Lazily connects in :meth:`__enter__`, tracks the active folder, and offers
      helper methods that wrap the underlying client while invoking
      :meth:`_throttle` before mutating operations.
    """

    def __init__(self, config: ImapConfig):
        """Initialise the client with the provided configuration envelope.

        What:
          Store :class:`ImapConfig` details and prepare internal caches used to
          manage mailbox selection, rate limiting, and control namespace paths.

        Why:
          Deferring connection establishment keeps object construction cheap
          while ensuring the runtime has all configuration hints required for
          later bootstrap steps.

        How:
          Persist ``config`` and set up empty caches for the IMAP connection,
          known mailboxes, and throttle queue. Actual network activity happens in
          :meth:`__enter__`.

        Args:
          config: Fully-populated IMAP configuration dataclass.
        """
        self._config = config
        self._client: Optional[IMAPClient] = None
        self._delimiter: str = "/"
        self._mailboxes: Set[str] = set()
        self._selected: Optional[str] = None
        self._control_mailbox: Optional[str] = None
        self._quarantine_mailbox: Optional[str] = None
        self._actions: Deque[float] = deque()

    def __enter__(self) -> "MailAIImapClient":
        """Establish the IMAP connection and prepare required mailboxes.

        What:
          Opens a connection, logs in, refreshes server folder listings, and
          bootstraps the control/quarantine namespace before selecting the
          default processing folder.

        Why:
          Running these steps up front ensures later operations can assume the
          presence of control mailboxes and a selected folder, reducing guard
          logic elsewhere.

        How:
          Instantiates ``IMAPClient`` with the configured host, performs login,
          and defers to helper methods for mailbox discovery and setup.

        Returns:
          The connected :class:`MailAIImapClient` instance.

        Raises:
          RuntimeError: When ``imapclient`` is not installed or the default
            mailbox is missing from the configuration.
        """

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
        """Log out of the IMAP session and release resources.

        What:
          Closes the connection opened in :meth:`__enter__`, ignoring errors so
          callers can rely on ``with`` semantics even during failures.

        Why:
          Properly disposing of the connection avoids leaving server sessions in
          idle states that might count against provider concurrency quotas.

        How:
          Attempts ``logout`` on the underlying client inside a ``try``/``finally``
          block and clears cached references regardless of outcome.
        """

        if self._client is None:
            return
        try:
            self._client.logout()
        finally:
            self._client = None

    @property
    def client(self) -> IMAPClient:
        """Expose the underlying ``IMAPClient`` connection.

        What:
          Returns the raw ``imapclient`` instance for operations not wrapped by
          this helper class.

        Why:
          Advanced callers occasionally need escape hatches (e.g., fetching
          specialised attributes) while still benefiting from connection
          lifecycle management.

        How:
          Raises a :class:`RuntimeError` if the client has not been connected via
          the context manager.

        Returns:
          The connected ``IMAPClient`` instance.

        Raises:
          RuntimeError: If accessed before :meth:`__enter__`.
        """

        if self._client is None:
            raise RuntimeError("IMAP client not connected")
        return self._client

    @property
    def config(self) -> ImapConfig:
        """Return the immutable :class:`ImapConfig` used to initialise the client.

        What:
          Provides read-only access to the configuration dataclass.

        Why:
          Callers may need to inspect folder hints (e.g., quarantine path) when
          orchestrating actions.

        How:
          Simply return the stored dataclass without copying.

        Returns:
          The :class:`ImapConfig` associated with this client.
        """

        return self._config

    @property
    def control_mailbox(self) -> str:
        """Return the fully-qualified control mailbox name.

        What:
          Exposes the normalised path to the control namespace folder created in
          :meth:`__enter__`.

        Why:
          Status uploads and rule watchers rely on the canonical path, which may
          include provider-specific delimiters.

        How:
          Verify :attr:`_control_mailbox` has been initialised and return it.

        Returns:
          Normalised mailbox string.

        Raises:
          RuntimeError: If accessed before bootstrapping completes.
        """

        if self._control_mailbox is None:
            raise RuntimeError("Control mailbox not initialised")
        return self._control_mailbox

    @property
    def quarantine_mailbox(self) -> str:
        """Return the quarantine mailbox used for suspicious messages.

        What:
          Provides the normalised quarantine folder name generated during
          bootstrap.

        Why:
          Rule actions occasionally need to move messages into quarantine while
          ensuring the folder exists.

        How:
          Ensure :attr:`_quarantine_mailbox` has been set during bootstrap and
          return the cached path.

        Returns:
          Normalised mailbox string.

        Raises:
          RuntimeError: If bootstrapping has not yet initialised the folder.
        """

        if self._quarantine_mailbox is None:
            raise RuntimeError("Quarantine mailbox not initialised")
        return self._quarantine_mailbox

    def _select(self, mailbox: str, *, readonly: bool = False) -> None:
        """Normalise ``mailbox`` and select it on the underlying client.

        What:
          Resolve the provider-specific folder path and make it the active
          selection for subsequent IMAP commands.

        Why:
          Ensures every operation targets the correct folder even when
          configuration uses ``/`` separators that differ from the server's
          delimiter.

        How:
          Call :meth:`_ensure_mailbox` to create or normalise the path, request
          selection via ``IMAPClient.select_folder``, and update ``_selected``.

        Args:
          mailbox: User-supplied mailbox string (may use generic separators).
          readonly: Whether to request read-only access.
        """

        mailbox_name = self._ensure_mailbox(mailbox)
        self.client.select_folder(mailbox_name, readonly=readonly)
        self._selected = mailbox_name

    def _refresh_mailboxes(self) -> None:
        """Synchronise the in-memory mailbox cache with the server listing.

        What:
          Populate ``_mailboxes`` with the server's folder names and remember the
          delimiter returned by ``LIST`` commands.

        Why:
          Subsequent operations (normalisation, creation) require an accurate
          view of available folders and the current delimiter.

        How:
          Issue ``IMAPClient.list_folders`` and iterate through responses,
          decoding bytes into strings, updating the delimiter when provided, and
          storing each folder name in ``_mailboxes``.
        """

        self._mailboxes.clear()
        for flags, delimiter, name in self.client.list_folders():
            if delimiter:
                decoded = delimiter.decode() if isinstance(delimiter, bytes) else str(delimiter)
                if decoded:
                    self._delimiter = decoded
            decoded_name = name.decode() if isinstance(name, bytes) else str(name)
            self._mailboxes.add(decoded_name)

    def _normalize_path(self, *parts: str) -> str:
        """Join ``parts`` using the server delimiter while trimming empties.

        What:
          Convert user-provided folder segments into the exact string the server
          expects, accounting for alternate separators.

        Why:
          IMAP servers may use ``.`` or ``/``; normalising prevents accidental
          creation of nested folders or selection failures.

        How:
          Replace ``/`` and ``.`` in each segment with the known delimiter,
          discard empty chunks, and join the remaining parts.

        Args:
          *parts: Sequence of folder path segments.

        Returns:
          Normalised mailbox string respecting the server delimiter.
        """

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
        """Create ``mailbox`` if missing and return the normalised name.

        What:
          Guarantee that the requested folder exists on the server.

        Why:
          MailAI must tolerate first-run scenarios where control or quarantine
          folders are absent; proactively creating them avoids race conditions.

        How:
          Normalise the name, attempt ``create_folder``, and refresh the cache on
          failure (some servers report errors for existing folders). Store the
          resulting name in ``_mailboxes`` before returning it.

        Args:
          mailbox: Folder to verify.

        Returns:
          Normalised mailbox string.
        """

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
        """Ensure control and quarantine mailboxes exist and record their paths.

        What:
          Build fully-qualified mailbox names for control/quarantine folders and
          create them if missing.

        Why:
          The runtime relies on these folders for configuration and status mail;
          missing mailboxes would break automation.

        How:
          Normalise the configured namespace and subfolder, call
          :meth:`_ensure_mailbox` for each, and store the resulting paths for
          later use.
        """

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
        """Enforce the per-minute action limit before mutating the mailbox.

        What:
          Track action timestamps and raise an error when more than 500
          operations occur within a minute.

        Why:
          Some providers aggressively limit automation; enforcing the cap avoids
          triggering bans while surfacing overload conditions to the caller.

        How:
          Drop timestamps older than 60 seconds, compare the queue size against
          the limit, raise on overflow, and append the current timestamp.
        """

        now = time.monotonic()
        while self._actions and now - self._actions[0] > 60:
            self._actions.popleft()
        if len(self._actions) >= 500:
            raise RuntimeError("IMAP action rate limit exceeded")
        self._actions.append(now)

    @contextlib.contextmanager
    def control_session(self, *, readonly: bool = False) -> Iterator[str]:
        """Temporarily switch to the control mailbox.

        What:
          Yields the control mailbox path while ensuring the previously selected
          folder is restored afterward.

        Why:
          Status uploads and config watchers frequently require short-lived
          excursions to the control namespace and must avoid leaving the client
          on the wrong folder.

        How:
          Stores the prior selection, selects the control mailbox, yields it, and
          reselects the original folder in a ``finally`` block.

        Args:
          readonly: Whether to request read-only mode during the session.

        Yields:
          The name of the control mailbox for convenience.
        """

        previous = self._selected
        self._select(self.control_mailbox, readonly=readonly)
        try:
            yield self.control_mailbox
        finally:
            if previous:
                self._select(previous, readonly=False)

    @contextlib.contextmanager
    def session(self, mailbox: str, *, readonly: bool = False) -> Iterator[str]:
        """Select ``mailbox`` for the duration of the context manager.

        What:
          Switches to ``mailbox`` and yields the normalised name before restoring
          the previous selection when the context exits.

        Why:
          Ensures operations such as rule mail discovery can work with arbitrary
          folders without permanently altering the client's active mailbox.

        How:
          Calls :meth:`_select` to normalise and open the folder, yields the
          resolved name, then restores the previous selection if one existed.

        Args:
          mailbox: Target mailbox name (will be normalised to provider delimiter).
          readonly: Whether to request read-only mode.

        Yields:
          The normalised mailbox name.
        """

        previous = self._selected
        self._select(mailbox, readonly=readonly)
        try:
            yield self._selected or mailbox
        finally:
            if previous:
                self._select(previous, readonly=False)

    def fetch_headers(self, uids: Iterable[int], data: str = "BODY.PEEK[HEADER]") -> dict:
        """Fetch message metadata for the provided ``uids``.

        What:
          Wraps ``IMAPClient.fetch`` so callers can pull headers or additional
          attributes while the client enforces UID usage.

        Why:
          Keeps the fetch interface narrow and ensures upstream code does not use
          sequence numbers by mistake.

        How:
          Delegates directly to the underlying client.

        Args:
          uids: Iterable of message UIDs to request.
          data: Fetch specifier (defaults to header peek).

        Returns:
          Mapping of UID to fetch results.
        """

        return self.client.fetch(uids, data)

    def move(self, uid: int, destination: str) -> None:
        """Move ``uid`` to the normalised ``destination`` folder.

        What:
          Ensures the destination exists, enforces rate limits, and delegates to
          ``IMAPClient.move``.

        Why:
          Moving messages is core to the rule engine; centralising the logic
          allows consistent mailbox normalisation.

        How:
          Calls :meth:`_ensure_mailbox` before invoking the client move method.

        Args:
          uid: Message UID to relocate.
          destination: User-facing folder hint.
        """

        dest = self._ensure_mailbox(destination)
        self._throttle()
        self.client.move(uid, dest)

    def copy(self, uid: int, destination: str) -> None:
        """Copy ``uid`` into ``destination`` without removing the source.

        What:
          Normalises the destination and performs a rate-limited copy.

        Why:
          Used for duplication workflows where a copy should be kept in the
          original folder.

        Args:
          uid: Message UID to copy.
          destination: Target folder name.
        """

        dest = self._ensure_mailbox(destination)
        self._throttle()
        self.client.copy(uid, dest)

    def add_label(self, uid: int, label: str) -> None:
        """Attach a Gmail label to ``uid``.

        What:
          Calls ``add_gmail_labels`` after rate limiting.

        Why:
          Gmail exposes labels separate from folders; this helper keeps the rule
          engine agnostic to provider specifics.

        Args:
          uid: Message UID to label.
          label: Label string to apply.
        """

        self._throttle()
        self.client.add_gmail_labels(uid, [label])

    def mark_read(self, uid: int, read: bool) -> None:
        """Toggle the ``\Seen`` flag depending on ``read``.

        What:
          Adds or removes the seen flag to reflect the requested read state.

        Why:
          Keeps state transitions idempotent by issuing explicit add/remove
          commands.

        Args:
          uid: Message UID to update.
          read: Desired read state.
        """

        self._throttle()
        if read:
            self.client.add_flags(uid, ["\\Seen"])
        else:
            self.client.remove_flags(uid, ["\\Seen"])

    def add_flag(self, uid: int, flag: str) -> None:
        """Set an arbitrary IMAP flag on ``uid``.

        What:
          Adds ``flag`` to the message while respecting rate limits.

        Why:
          Reused by delete semantics (``\Deleted``) and custom workflows.

        Args:
          uid: Message UID to modify.
          flag: IMAP flag name to add.
        """

        self._throttle()
        self.client.add_flags(uid, [flag])

    def set_header(self, uid: int, name: str, value: str) -> None:
        """Append a header patch to the currently selected mailbox.

        What:
          Issues an ``APPEND`` command containing an RFC 822 header fragment.

        Why:
          Some providers lack ``UID STORE`` support for arbitrary headers; this
          workaround allows MailAI to add audit headers when necessary.

        Args:
          uid: Message UID associated with the patch (for audit trails only).
          name: Header name to inject.
          value: Header value to inject.
        """

        patch = f"{name}: {value}\r\n"
        self._throttle()
        self.client.append(self._selected or self._config.folder, patch.encode("utf-8"))

    def uid_search(self, criteria: List[object]) -> List[int]:
        """Run a UID search with the provided ``criteria``.

        What:
          Issues ``IMAPClient.search`` using UID semantics and returns a list of
          integers.

        Why:
          Keeps call sites explicit about UID usage while providing a consistent
          list conversion (``imapclient`` returns tuples).

        Args:
          criteria: Search criteria as understood by ``imapclient``.

        Returns:
          List of matching UIDs (may be empty).
        """

        return list(self.client.search(criteria))

    def get_rules_email(self, subject: str) -> Optional[int]:
        """Return the UID of the latest rules mail matching ``subject``.

        What:
          Searches the control mailbox for messages with the configured subject
          and returns the highest UID.

        Why:
          The rules watcher needs a quick way to locate the canonical
          ``rules.yaml`` message without parsing the body.

        Args:
          subject: Subject line to match.

        Returns:
          UID of the most recent matching message, or ``None`` if absent.
        """

        with self.control_session(readonly=True):
            uids = self.client.search(["SUBJECT", subject])
        return max(uids) if uids else None


# TODO: Other modules in this repository still require the same What/Why/How documentation.

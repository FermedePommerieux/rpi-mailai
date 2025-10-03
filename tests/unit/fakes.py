"""In-memory IMAP backend used by unit tests.

What:
  Provide a drop-in replacement for :class:`imapclient.IMAPClient` that stores
  messages in Python data structures while exposing a subset of the IMAP API
  required by tests.

Why:
  Unit tests must exercise IMAP workflows (search, fetch, append) without
  contacting real servers. The fake backend keeps behaviour deterministic and
  debuggable.

How:
  Maintain per-mailbox dictionaries of :class:`_MessageRecord` entries. Methods
  mutate these dictionaries to emulate IMAP semantics (UID assignment, COPY,
  MOVE) while obeying the MailAI configuration defaults.

Interfaces:
  :class:`FakeImapBackend`.

Invariants & Safety:
  - UIDs increment monotonically per backend instance.
  - Mailboxes referenced by control/quarantine helpers are always present.
  - Methods avoid network calls and operate solely on in-memory data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import format_datetime, make_msgid
from typing import Dict, Iterable

from mailai.config.loader import get_runtime_config


@dataclass
class _MessageRecord:
    """Internal representation of a stored message.

    What:
      Capture the metadata and payload necessary to emulate IMAP fetches.

    Why:
      Tests need predictable responses for different ``BODY`` parts. Storing the
      full byte payload plus decoded body text allows flexible fetch behaviour.

    How:
      Tracks UID, mailbox, headers, body text, charset, and original bytes. The
      dataclass form keeps the structure lightweight and easy to instantiate.
    """

    uid: int
    mailbox: str
    subject: str
    message_id: str
    internaldate: datetime
    message_bytes: bytes
    body_text: str
    charset: str

    @property
    def header_bytes(self) -> bytes:
        """Return RFC822 header bytes derived from the stored message.

        What:
          Parses ``message_bytes`` and rebuilds the header block ending with a
          blank line.

        Why:
          Some tests fetch ``BODY[HEADER]``. Precomputing the header segment keeps
          fetch responses faithful to real IMAP servers.

        How:
          Re-parses the raw bytes with :class:`BytesParser`, iterates over
          headers, and concatenates ``name: value`` lines separated by CRLF.

        Returns:
          Header section as bytes terminated by ``\r\n\r\n``.
        """

        message = BytesParser(policy=policy.default).parsebytes(self.message_bytes)
        headers = []
        for name, value in message.items():
            headers.append(f"{name}: {value}\r\n".encode("utf-8"))
        headers.append(b"\r\n")
        return b"".join(headers)


class FakeImapBackend:
    """Minimal IMAP backend satisfying the subset MailAI tests rely upon.

    What:
      Emulate enough of :class:`imapclient.IMAPClient` for tests to exercise the
      rule engine and configuration flows without network access.

    Why:
      Provides deterministic behaviour and enables assertions on internal state
      (mailboxes, events) that would be difficult against a live server.

    How:
      Manages per-mailbox dictionaries of :class:`_MessageRecord` entries,
      assigning sequential UIDs and exposing methods that mutate those
      collections to mimic IMAP semantics.
    """

    def __init__(self) -> None:
        """Initialise the backend with default MailAI mailboxes.

        What:
          Seeds mailboxes for the default, control, and quarantine namespaces and
          prepares UID counters.

        Why:
          Ensures tests immediately have the folders referenced by production
          code, avoiding additional setup in fixtures.

        How:
          Reads runtime configuration for mailbox names, constructs dictionaries
          for each, and initialises the active ``selected`` mailbox.
        """

        settings = get_runtime_config()
        default_mailbox = settings.imap.default_mailbox
        control = settings.imap.control_namespace
        quarantine = f"{control}/{settings.imap.quarantine_subfolder}".replace("//", "/")
        self.mailboxes: Dict[str, Dict[int, _MessageRecord]] = {
            default_mailbox: {},
            control: {},
            quarantine: {},
        }
        self.selected = default_mailbox
        self.uid_counter = 1

    # Session management -------------------------------------------------
    def login(self, username: str, password: str) -> None:  # pragma: no cover - trivial
        """No-op login method to satisfy client expectations.

        What:
          Accepts credentials but performs no action.

        Why:
          :class:`MailAIImapClient` invokes ``login`` on connection startup. The
          fake backend needs to mirror that interface without side effects.

        How:
          Ignore the provided ``username``/``password`` and immediately return
          ``None``.

        Args:
          username: Ignored username.
          password: Ignored password.
        """

        return None

    def logout(self) -> None:  # pragma: no cover - trivial
        """No-op logout method.

        What:
          Exists purely to match the real client's API.

        Why:
          Ensures context managers can call ``logout`` during teardown without
          additional guards.

        How:
          Simply return ``None`` without touching backend state.
        """

        return None

    # Mailbox helpers ----------------------------------------------------
    def list_folders(self):
        """Return folder listings mimicking ``IMAPClient.list_folders`` output.

        What:
          Produces a list of ``(flags, delimiter, name)`` tuples.

        Why:
          Tests assert on folder availability and rely on this structure to match
          the real library's response format.

        How:
          Sort mailbox names alphabetically and wrap them with default
          delimiter/flag values.

        Returns:
          Sorted list of folder descriptors.
        """

        return [([], "/", name) for name in sorted(self.mailboxes)]

    def create_folder(self, name: str) -> None:
        """Ensure ``name`` exists in the backend's mailbox map.

        What:
          Adds an empty dictionary for ``name`` when absent.

        Why:
          Some operations (MOVE/COPY) implicitly create folders; tests mimic that
          behaviour.

        How:
          Insert ``name`` into :attr:`mailboxes` with ``dict.setdefault``.

        Args:
          name: Mailbox identifier.
        """

        self.mailboxes.setdefault(name, {})

    def select_folder(self, name: str, readonly: bool = False) -> None:
        """Change the active mailbox for subsequent operations.

        What:
          Sets :attr:`selected` to ``name`` creating the mailbox when missing.

        Why:
          Fetch/search operations target :attr:`selected`; switching ensures the
          backend mirrors ``IMAPClient.select_folder`` behaviour.

        How:
          Call :meth:`create_folder` to ensure existence and store ``name`` in
          :attr:`selected` while ignoring ``readonly``.

        Args:
          name: Mailbox to select.
          readonly: Ignored flag included for API parity.
        """

        self.create_folder(name)
        self.selected = name

    # Message operations -------------------------------------------------
    def search(self, criteria):
        """Search the selected mailbox using a minimal criteria subset.

        What:
          Supports ``['SUBJECT', value]`` lookups returning matching UIDs.

        Why:
          Tests only rely on subject-based queries to identify control mails. A
          narrower implementation reduces maintenance burden.

        How:
          Inspect the first search token and return UIDs whose stored subject
          matches the provided value; unsupported tokens yield an empty list.

        Args:
          criteria: Sequence describing the IMAP search query.

        Returns:
          List of UIDs whose stored subject matches ``value``.
        """

        if not criteria:
            return []
        if criteria[0] != "SUBJECT":
            return []
        subject = criteria[1]
        return [uid for uid, msg in self.mailboxes[self.selected].items() if msg.subject == subject]

    def fetch(self, uids: Iterable[int], parts: Iterable[bytes | str]):
        """Return message parts for the requested UIDs.

        What:
          Emulates ``IMAPClient.fetch`` by returning a dictionary keyed by UID
          with requested BODY/RFC822 segments.

        Why:
          MailAI's loader expects specific byte sequences; the fake must mirror
          those to avoid brittle tests.

        How:
          Iterate through ``uids``, translate requested ``parts`` into bytes, and
          assemble dictionaries with headers, bodies, and metadata drawn from
          :class:`_MessageRecord`.

        Args:
          uids: Iterable of message UIDs to fetch.
          parts: Iterable of IMAP parts (bytes or strings).

        Returns:
          Mapping of UID to a dictionary of requested parts.
        """

        response: Dict[int, Dict[bytes, object]] = {}
        requested = [part.encode() if isinstance(part, str) else part for part in parts]
        for uid in uids:
            if uid not in self.mailboxes[self.selected]:
                continue
            record = self.mailboxes[self.selected][uid]
            payload: Dict[bytes, object] = {}
            for part in requested:
                upper = part.upper()
                if upper in {b"RFC822", b"BODY[]"}:
                    payload[part] = record.message_bytes
                elif upper in {b"BODY[TEXT]", b"BODY.PEEK[TEXT]"}:
                    payload[part] = record.body_text.encode(record.charset)
                elif upper in {b"BODY[HEADER]", b"BODY.PEEK[HEADER]"}:
                    payload[part] = record.header_bytes
                elif upper == b"RFC822.SIZE":
                    payload[part] = len(record.body_text.encode(record.charset))
                elif upper == b"INTERNALDATE":
                    payload[part] = format_datetime(record.internaldate).encode("utf-8")
            response[uid] = payload
        return response

    def append(self, mailbox: str, message_bytes: bytes) -> None:
        """Store a new message and assign the next UID.

        What:
          Parses ``message_bytes`` into an :class:`EmailMessage`, extracts body
          text, and adds a :class:`_MessageRecord` to ``mailbox``.

        Why:
          Config and status mails are appended by tests to simulate incoming IMAP
          traffic.

        How:
          Create the mailbox if needed, parse the message via :class:`BytesParser`,
          derive text/charset, and persist a new :class:`_MessageRecord` while
          incrementing :attr:`uid_counter`.

        Args:
          mailbox: Destination mailbox name.
          message_bytes: Raw RFC822 bytes to store.
        """

        self.create_folder(mailbox)
        parser = BytesParser(policy=policy.default)
        message = parser.parsebytes(message_bytes)
        if not isinstance(message, EmailMessage):
            message = EmailMessage(policy=policy.default)
            message.set_content(message_bytes.decode("utf-8"))
        body = message.get_body(preferencelist=("plain",))
        charset = body.get_content_charset("utf-8") if body else "utf-8"
        text = body.get_content() if body else message.get_content()
        message_id = message["Message-ID"] or make_msgid(domain="fake.local")
        if body is None:
            charset = message.get_content_charset("utf-8")
        record = _MessageRecord(
            uid=self.uid_counter,
            mailbox=mailbox,
            subject=str(message["Subject"]),
            message_id=str(message_id),
            internaldate=datetime.now(timezone.utc),
            message_bytes=message_bytes,
            body_text=str(text),
            charset=str(charset),
        )
        self.mailboxes[mailbox][self.uid_counter] = record
        self.uid_counter += 1

    def delete_messages(self, uids: Iterable[int]) -> None:
        """Remove UIDs from the selected mailbox.

        What:
          Deletes each UID from the active mailbox, ignoring missing entries.

        Why:
          Tests simulate user deletions and expect silent success if a UID is
          absent.

        How:
          Iterate through ``uids`` and ``pop`` matching entries from the selected
          mailbox.

        Args:
          uids: Iterable of message UIDs to remove.
        """

        for uid in list(uids):
            self.mailboxes[self.selected].pop(uid, None)

    def expunge(self) -> None:  # pragma: no cover - trivial
        """No-op expunge method provided for API parity.

        What:
          Present to satisfy callers expecting ``expunge`` on IMAP clients.

        Why:
          Tests do not need to simulate expunge behaviour but rely on the method
          existing.

        How:
          Return ``None`` without mutating state.
        """

        return None

    # Action helpers -----------------------------------------------------
    def move(self, uid: int, destination: str) -> None:
        """Move ``uid`` from the selected mailbox to ``destination``.

        What:
          Removes the record from the active mailbox and inserts it into the
          destination mailbox.

        Why:
          Allows rule actions to verify behaviour when messages are relocated to
          quarantine or other folders.

        How:
          Ensure ``destination`` exists, ``pop`` the record from the selected
          mailbox, and assign it under the new mailbox.

        Args:
          uid: Message identifier to move.
          destination: Target mailbox name.
        """

        self.create_folder(destination)
        self.mailboxes[destination][uid] = self.mailboxes[self.selected].pop(uid)

    def copy(self, uid: int, destination: str) -> None:
        """Copy ``uid`` to ``destination`` without removing the original.

        What:
          Clones the record reference into ``destination``.

        Why:
          Tests for copy semantics rely on verifying the message exists in both
          mailboxes post-operation.

        How:
          Ensure ``destination`` exists and assign the same record object to the
          destination mailbox without deleting the original.

        Args:
          uid: Message identifier to duplicate.
          destination: Mailbox receiving the copy.
        """

        self.create_folder(destination)
        self.mailboxes[destination][uid] = self.mailboxes[self.selected][uid]

    def add_gmail_labels(self, uid, labels) -> None:  # pragma: no cover - unused in tests
        """No-op Gmail label helper maintained for compatibility.

        What:
          Accepts Gmail label arguments without applying changes.

        Why:
          The production client exposes this method; the fake keeps the same
          surface even though tests never invoke it meaningfully.

        How:
          Return ``None`` immediately.
        """

        return None

    def add_flags(self, uid, flags) -> None:  # pragma: no cover - unused in tests
        """No-op flag addition helper.

        What:
          Placeholder for IMAP ``add_flags`` behaviour.

        Why:
          Maintains API compatibility with :class:`imapclient.IMAPClient`.

        How:
          Immediately return without modifying message state.
        """

        return None

    def remove_flags(self, uid, flags) -> None:  # pragma: no cover - unused in tests
        """No-op flag removal helper.

        What:
          Placeholder matching the real client's ``remove_flags`` signature.

        Why:
          Ensures consumers that toggle flags can run against the fake backend
          without attribute errors.

        How:
          Return ``None`` without altering stored records.
        """

        return None


# TODO: Other modules in this repository still require the same What/Why/How documentation.

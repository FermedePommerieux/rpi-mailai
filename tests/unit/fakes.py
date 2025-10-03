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
        message = BytesParser(policy=policy.default).parsebytes(self.message_bytes)
        headers = []
        for name, value in message.items():
            headers.append(f"{name}: {value}\r\n".encode("utf-8"))
        headers.append(b"\r\n")
        return b"".join(headers)


class FakeImapBackend:
    """In-memory IMAP backend implementing a subset of `imapclient.IMAPClient`."""

    def __init__(self) -> None:
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
        return None

    def logout(self) -> None:  # pragma: no cover - trivial
        return None

    # Mailbox helpers ----------------------------------------------------
    def list_folders(self):
        return [([], "/", name) for name in sorted(self.mailboxes)]

    def create_folder(self, name: str) -> None:
        self.mailboxes.setdefault(name, {})

    def select_folder(self, name: str, readonly: bool = False) -> None:
        self.create_folder(name)
        self.selected = name

    # Message operations -------------------------------------------------
    def search(self, criteria):
        if not criteria:
            return []
        if criteria[0] != "SUBJECT":
            return []
        subject = criteria[1]
        return [uid for uid, msg in self.mailboxes[self.selected].items() if msg.subject == subject]

    def fetch(self, uids: Iterable[int], parts: Iterable[bytes | str]):
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
        for uid in list(uids):
            self.mailboxes[self.selected].pop(uid, None)

    def expunge(self) -> None:  # pragma: no cover - trivial
        return None

    # Action helpers -----------------------------------------------------
    def move(self, uid: int, destination: str) -> None:
        self.create_folder(destination)
        self.mailboxes[destination][uid] = self.mailboxes[self.selected].pop(uid)

    def copy(self, uid: int, destination: str) -> None:
        self.create_folder(destination)
        self.mailboxes[destination][uid] = self.mailboxes[self.selected][uid]

    def add_gmail_labels(self, uid, labels) -> None:  # pragma: no cover - unused in tests
        return None

    def add_flags(self, uid, flags) -> None:  # pragma: no cover - unused in tests
        return None

    def remove_flags(self, uid, flags) -> None:  # pragma: no cover - unused in tests
        return None

"""Test doubles for IMAP interactions."""
from __future__ import annotations

import email
from typing import Dict


class FakeImapBackend:
    """In-memory IMAP backend implementing a subset of `imapclient.IMAPClient`."""

    def __init__(self) -> None:
        self.mailboxes: Dict[str, Dict[int, Dict[str, object]]] = {
            "INBOX": {},
            "MailIA": {},
            "MailIA/Quarantine": {},
        }
        self.selected = "INBOX"
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
        subject = criteria[1]
        return [uid for uid, msg in self.mailboxes[self.selected].items() if msg["subject"] == subject]

    def fetch(self, uids, parts):
        data = {}
        for uid in uids:
            payload = self.mailboxes[self.selected][uid]["payload"]
            data[uid] = {b"RFC822": payload}
        return data

    def append(self, mailbox: str, message_bytes: bytes) -> None:
        self.create_folder(mailbox)
        message = email.message_from_bytes(message_bytes)
        payload = message.get_payload(decode=True)
        if payload is None:
            payload = message.get_payload().encode()
        self.mailboxes[mailbox][self.uid_counter] = {
            "subject": message["Subject"],
            "payload": payload,
        }
        self.uid_counter += 1

    def delete_messages(self, uids) -> None:
        for uid in uids:
            self.mailboxes[self.selected].pop(uid, None)

    def expunge(self) -> None:  # pragma: no cover - trivial
        return None

    # Action helpers -----------------------------------------------------
    def move(self, uid: int, destination: str) -> None:
        self.create_folder(destination)
        self.mailboxes[destination][uid] = self.mailboxes[self.selected].pop(uid)

    def copy(self, uid: int, destination: str) -> None:
        self.create_folder(destination)
        self.mailboxes[destination][uid] = dict(self.mailboxes[self.selected][uid])

    def add_gmail_labels(self, uid, labels) -> None:  # pragma: no cover - unused in tests
        return None

    def add_flags(self, uid, flags) -> None:  # pragma: no cover - unused in tests
        return None

    def remove_flags(self, uid, flags) -> None:  # pragma: no cover - unused in tests
        return None

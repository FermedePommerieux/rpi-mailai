"""Helpers for managing the `MailAI: rules.yaml` message."""
from __future__ import annotations

from email.message import EmailMessage
from typing import Optional

from ..config.loader import LoadedDocument, dump_rules, load_rules
from ..config.schema import RulesV2
from ..utils.ids import checksum
from .client import MailAIImapClient

RULES_SUBJECT = "MailAI: rules.yaml"
RULES_BACKUP_SUBJECT = "MailAI: rules.bak.yaml"
MAX_BYTES = 200_000


def read_rules(client: MailAIImapClient, *, fallback: Optional[bytes] = None) -> LoadedDocument:
    """Fetch and validate the latest rules YAML from the mailbox."""

    uid = client.get_rules_email(RULES_SUBJECT)
    if uid is None:
        if fallback is None:
            raise FileNotFoundError("rules.yaml not found in mailbox")
        document = load_rules(fallback)
        upsert_rules(client, document.model)
        return document
    data = client.client.fetch([uid], [b"RFC822"])[uid][b"RFC822"]
    return load_rules(data)


def upsert_rules(client: MailAIImapClient, model: RulesV2) -> str:
    """Upload a rules document, replacing previous copies."""

    payload = dump_rules(model)
    if len(payload) > MAX_BYTES:
        raise ValueError("rules.yaml exceeds 200KB limit")
    _delete_existing(client, RULES_SUBJECT)
    message = _build_message(RULES_SUBJECT, payload)
    client.client.append(client.config.folder, message.as_bytes())
    _delete_existing(client, RULES_BACKUP_SUBJECT)
    backup = _build_message(RULES_BACKUP_SUBJECT, payload)
    client.client.append(client.config.folder, backup.as_bytes())
    return checksum(payload)


def _delete_existing(client: MailAIImapClient, subject: str) -> None:
    uids = client.client.search(["SUBJECT", subject])
    if not uids:
        return
    client.client.delete_messages(uids)
    client.client.expunge()


def _build_message(subject: str, payload: bytes) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = "mailai@local"
    message["To"] = "mailai@local"
    message.set_content(payload.decode("utf-8"))
    return message

"""Helpers for managing the `MailAI: rules.yaml` message."""
from __future__ import annotations

from email.message import EmailMessage
from typing import Optional

from typing import Optional

from ..config.loader import LoadedDocument, dump_rules, load_rules, YamlValidationError
from ..config.schema import RulesV2
from ..utils.ids import checksum
from .client import MailAIImapClient

RULES_SUBJECT = "MailAI: rules.yaml"
RULES_BACKUP_SUBJECT = "MailAI: rules.bak.yaml"
HARD_LIMIT = 128 * 1024


def read_rules(client: MailAIImapClient, *, fallback: Optional[bytes] = None) -> LoadedDocument:
    """Fetch and validate the latest rules YAML from the mailbox."""

    uid = client.get_rules_email(RULES_SUBJECT)
    if uid is None:
        return _restore_rules(client, fallback)
    with client.control_session(readonly=True):
        payload = client.client.fetch([uid], [b"RFC822"])[uid][b"RFC822"]
    try:
        document = load_rules(payload)
    except YamlValidationError:
        return _repair_rules(client, fallback)
    _ensure_backup(client, payload)
    return document


def upsert_rules(client: MailAIImapClient, model: RulesV2) -> str:
    """Upload a rules document, replacing previous copies."""

    payload = dump_rules(model)
    if len(payload) > HARD_LIMIT:
        raise ValueError("rules.yaml exceeds 128KB hard limit")
    with client.control_session():
        _delete_existing(client, RULES_SUBJECT)
        message = _build_message(RULES_SUBJECT, payload)
        client.client.append(client.control_mailbox, message.as_bytes())
        _delete_existing(client, RULES_BACKUP_SUBJECT)
        backup = _build_message(RULES_BACKUP_SUBJECT, payload)
        client.client.append(client.control_mailbox, backup.as_bytes())
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


def _restore_rules(client: MailAIImapClient, fallback: Optional[bytes]) -> LoadedDocument:
    if fallback is not None:
        document = load_rules(fallback)
        upsert_rules(client, document.model)
        return document
    minimal = RulesV2.minimal()
    payload = dump_rules(minimal)
    upsert_rules(client, minimal)
    return load_rules(payload)


def _repair_rules(client: MailAIImapClient, fallback: Optional[bytes]) -> LoadedDocument:
    with client.control_session(readonly=True):
        backup_uid = client.client.search(["SUBJECT", RULES_BACKUP_SUBJECT])
        if backup_uid:
            data = client.client.fetch(backup_uid, [b"RFC822"])[backup_uid[-1]][b"RFC822"]
            try:
                document = load_rules(data)
            except YamlValidationError:
                document = None
            else:
                upsert_rules(client, document.model)
                return document
    return _restore_rules(client, fallback)


def _ensure_backup(client: MailAIImapClient, payload: bytes) -> None:
    digest = checksum(payload)
    backup_uid = client.get_rules_email(RULES_BACKUP_SUBJECT)
    if backup_uid is not None:
        with client.control_session(readonly=True):
            stored = client.client.fetch([backup_uid], [b"RFC822"])[backup_uid][b"RFC822"]
        if checksum(stored) == digest:
            return
    with client.control_session():
        _delete_existing(client, RULES_BACKUP_SUBJECT)
        message = _build_message(RULES_BACKUP_SUBJECT, payload)
        client.client.append(client.control_mailbox, message.as_bytes())


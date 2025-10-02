"""Helpers for managing the `MailAI: status.yaml` message."""
from __future__ import annotations

from email.message import EmailMessage

from ..config.loader import dump_status
from ..config.schema import StatusV2
from .client import MailAIImapClient

STATUS_SUBJECT = "MailAI: status.yaml"
MAX_BYTES = 200_000


def upsert_status(client: MailAIImapClient, status: StatusV2) -> None:
    """Upload the latest status YAML, truncating notes when necessary."""

    payload = dump_status(status)
    if len(payload) > MAX_BYTES:
        raise ValueError("status.yaml exceeds 200KB limit")
    _delete_existing(client)
    message = EmailMessage()
    message["Subject"] = STATUS_SUBJECT
    message["From"] = "mailai@local"
    message["To"] = "mailai@local"
    message.set_content(payload.decode("utf-8"))
    client.client.append(client.config.folder, message.as_bytes())


def _delete_existing(client: MailAIImapClient) -> None:
    uids = client.client.search(["SUBJECT", STATUS_SUBJECT])
    if not uids:
        return
    client.client.delete_messages(uids)
    client.client.expunge()

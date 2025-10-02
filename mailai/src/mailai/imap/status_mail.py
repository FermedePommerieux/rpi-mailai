"""Helpers for managing the `MailAI: status.yaml` message."""
from __future__ import annotations

from email.message import EmailMessage

from ..config.loader import dump_status
from ..config.schema import StatusV2
from .client import MailAIImapClient

STATUS_SUBJECT = "MailAI: status.yaml"
SOFT_LIMIT = 64 * 1024
HARD_LIMIT = 128 * 1024


def upsert_status(client: MailAIImapClient, status: StatusV2) -> None:
    """Upload the latest status YAML, truncating notes when necessary."""

    payload = dump_status(status)
    if len(payload) > SOFT_LIMIT:
        status = _truncate_status(status)
        payload = dump_status(status)
    if len(payload) > HARD_LIMIT:
        raise ValueError("status.yaml exceeds 128KB limit")
    with client.control_session():
        _delete_existing(client)
        message = EmailMessage()
        message["Subject"] = STATUS_SUBJECT
        message["From"] = "mailai@local"
        message["To"] = "mailai@local"
        message.set_content(payload.decode("utf-8"))
        client.client.append(client.control_mailbox, message.as_bytes())


def _delete_existing(client: MailAIImapClient) -> None:
    uids = client.client.search(["SUBJECT", STATUS_SUBJECT])
    if not uids:
        return
    client.client.delete_messages(uids)
    client.client.expunge()


def _truncate_status(status: StatusV2) -> StatusV2:
    document = status.model_dump()
    notes = document.get("notes", [])
    proposals = document.get("proposals", [])
    truncated_notes = list(notes[:20])
    if len(notes) > 20:
        truncated_notes.append("… additional notes truncated …")
    truncated_proposals = list(proposals[:8])
    document["notes"] = truncated_notes
    document["proposals"] = truncated_proposals
    return StatusV2.model_validate(document)

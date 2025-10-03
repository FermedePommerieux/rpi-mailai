"""Helpers for managing the `MailAI: status.yaml` message."""
from __future__ import annotations

from email.message import EmailMessage

from ..config.loader import dump_status, get_runtime_config
from ..config.schema import StatusV2
from .client import MailAIImapClient

def upsert_status(client: MailAIImapClient, status: StatusV2) -> None:
    """Upload the latest status YAML, truncating notes when necessary."""

    settings = get_runtime_config()
    status_cfg = settings.mail.status
    limits = status_cfg.limits
    payload = dump_status(status)
    if len(payload) > limits.soft_limit:
        status = _truncate_status(status)
        payload = dump_status(status)
    if len(payload) > limits.hard_limit:
        raise ValueError("status.yaml exceeds hard size limit")
    target_folder = status_cfg.folder or client.control_mailbox
    with client.session(target_folder, readonly=False) as mailbox:
        _delete_existing(client, status_cfg.subject)
        message = EmailMessage()
        message["Subject"] = status_cfg.subject
        message["From"] = "mailai@local"
        message["To"] = "mailai@local"
        message.set_content(payload.decode("utf-8"))
        client.client.append(mailbox, message.as_bytes())


def _delete_existing(client: MailAIImapClient, subject: str) -> None:
    uids = client.client.search(["SUBJECT", subject])
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

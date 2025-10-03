"""Maintain the ``MailAI: status.yaml`` status message over IMAP.

What:
  Serialize status documents, enforce size limits, and upsert the canonical
  ``status.yaml`` message inside the control mailbox.

Why:
  Operators rely on ``status.yaml`` for audit trails and learner proposals; the
  runtime must guarantee updates remain within soft/hard limits and that stale
  copies are removed before writing new content.

How:
  Renders :class:`StatusV2` models to YAML, truncates note/proposal lists when
  exceeding the configured soft limit, and uses :class:`MailAIImapClient`
  sessions to delete old messages before appending replacements.

Interfaces:
  :func:`upsert_status`.

Invariants & Safety:
  - Soft limit truncations append an explicit ellipsis so operators know data was
    trimmed.
  - Hard limit breaches raise errors instead of uploading partial payloads.
  - Uploads run through the control mailbox to avoid interfering with user mail.
"""
from __future__ import annotations

from email.message import EmailMessage

from ..config.loader import dump_status, get_runtime_config
from ..config.schema import StatusV2
from .client import MailAIImapClient

def upsert_status(client: MailAIImapClient, status: StatusV2) -> None:
    """Write ``status.yaml`` to the control mailbox, respecting size limits.

    What:
      Serialises ``status`` to YAML, applies soft-limit truncation, enforces the
      hard byte limit, and replaces any existing status message in the target
      folder.

    Why:
      Ensures operators always see the freshest status snapshot while preventing
      IMAP quota issues or truncated uploads that could mislead them.

    How:
      Calls :func:`dump_status`, checks lengths, optionally invokes
      :func:`_truncate_status`, deletes previous messages, and appends the new
      email using :class:`EmailMessage`.

    Args:
      client: Connected IMAP client scoped to the MailAI account.
      status: Status model to serialise.

    Raises:
      ValueError: If the final payload exceeds the configured hard limit.
    """

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
    """Remove prior ``status.yaml`` messages to keep the mailbox deduplicated.

    What:
      Searches for matching subjects, deletes them, and expunges the mailbox to
      avoid duplicates lingering in the control folder.

    Why:
      Prevents accumulation of stale status documents which could confuse
      operators and waste mailbox quota.

    Args:
      client: Connected IMAP client.
      subject: Subject string used to identify status mails.
    """

    uids = client.client.search(["SUBJECT", subject])
    if not uids:
        return
    client.client.delete_messages(uids)
    client.client.expunge()


def _truncate_status(status: StatusV2) -> StatusV2:
    """Apply soft-limit truncation to verbose fields.

    What:
      Limits ``notes`` and ``proposals`` to a manageable size, appending an
      ellipsis entry when truncation occurs.

    Why:
      Keeps the payload within soft limits so the platform rarely hits the hard
      cap while signalling to operators that extra content was omitted.

    Args:
      status: Original status model prior to truncation.

    Returns:
      New :class:`StatusV2` instance with truncated fields.
    """

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


# TODO: Other modules in this repository still require the same What/Why/How documentation.

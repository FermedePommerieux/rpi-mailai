"""Helpers for locating and repairing the `MailAI: rules.yaml` message."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import make_msgid, parsedate_to_datetime
from typing import Optional

from ..config.loader import get_runtime_config
from ..config.schema import RulesV2
from ..config import yamlshim
from ..utils.ids import checksum
from .client import MailAIImapClient


@dataclass
class RulesMailRef:
    """Reference to the canonical rules mail."""

    uid: int
    message_id: Optional[str]
    internaldate: datetime
    charset: str
    body_text: str
    checksum: str


def find_latest(
    subject: Optional[str] = None,
    folder: Optional[str] = None,
    *,
    client: MailAIImapClient,
) -> Optional[RulesMailRef]:
    """Return the most recent `rules.yaml` mail for the given subject."""

    settings = get_runtime_config()
    effective_subject = subject or settings.mail.rules.subject
    target_folder = folder or settings.mail.rules.folder
    with client.session(target_folder, readonly=True):
        uids = client.client.search(["SUBJECT", effective_subject])
        if not uids:
            return None
        uid = max(uids)
        data = client.client.fetch(
            [uid],
            [
                b"BODY.PEEK[HEADER]",
                b"BODY.PEEK[TEXT]",
                b"RFC822.SIZE",
                b"INTERNALDATE",
            ],
        )[uid]
    header_bytes = _first(data, b"BODY.PEEK[HEADER]", b"BODY[HEADER]")
    body_bytes = _first(data, b"BODY.PEEK[TEXT]", b"BODY[TEXT]")
    if header_bytes is None and body_bytes is None:
        return None
    if header_bytes is None:
        header_bytes = b""
    if body_bytes is None:
        body_bytes = b""
    combined = header_bytes + b"\r\n" + body_bytes
    message = BytesParser(policy=policy.default).parsebytes(combined)
    body = message.get_body(preferencelist=("plain",))
    charset = body.get_content_charset("utf-8") if body else message.get_content_charset("utf-8")
    text = body.get_content() if body else message.get_content()
    if not isinstance(text, str):
        text = text.decode(charset or "utf-8")  # type: ignore[union-attr]
    normalised = _normalise_text(text)
    digest = checksum(normalised.encode("utf-8"))
    message_id = message.get("Message-ID")
    internal_raw = _first(data, b"INTERNALDATE")
    internaldate = _parse_internaldate(internal_raw) or datetime.now(timezone.utc)
    return RulesMailRef(
        uid=uid,
        message_id=message_id,
        internaldate=internaldate,
        charset=charset or "utf-8",
        body_text=normalised,
        checksum=digest,
    )


def append_minimal_template(
    folder: Optional[str] = None,
    *,
    client: MailAIImapClient,
) -> RulesMailRef:
    """Append a minimal configuration template and return its reference."""

    settings = get_runtime_config()
    target_folder = folder or settings.mail.rules.folder
    minimal = RulesV2.minimal()
    yaml_text = yamlshim.dump(minimal.model_dump(mode="json"))
    message = EmailMessage()
    message["Subject"] = settings.mail.rules.subject
    message["From"] = "mailai@local"
    message["To"] = "mailai@local"
    message["Message-ID"] = make_msgid(domain="mailai.local")
    message.set_content(yaml_text, subtype="yaml")
    with client.session(target_folder, readonly=False) as mailbox:
        client.client.append(mailbox, message.as_bytes())
    return find_latest(subject=settings.mail.rules.subject, folder=target_folder, client=client)


def _normalise_text(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip() + "\n"


def _parse_internaldate(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return parsedate_to_datetime(value.decode("utf-8"))
        except (ValueError, TypeError):
            return None
    if isinstance(value, str):
        try:
            return parsedate_to_datetime(value)
        except (ValueError, TypeError):
            return None
    return None


def _first(data: dict, *keys: bytes) -> Optional[bytes]:
    for key in keys:
        if key in data:
            value = data[key]
            if isinstance(value, bytes):
                return value
            if isinstance(value, str):
                return value.encode("utf-8")
    return None

"""Helpers for parsing MIME messages safely."""
from __future__ import annotations

from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from typing import Dict, Tuple


MAX_BODY_BYTES = 1_000_000


def parse_message(raw: bytes) -> Tuple[EmailMessage, Dict[str, str], str]:
    """Parse a raw RFC822 message and return message, headers, and text body."""

    parser = BytesParser(policy=policy.default)
    message = parser.parsebytes(raw)
    headers = {k.lower(): v for k, v in message.items()}
    body_text = _extract_body_text(message)
    return message, headers, body_text


def _extract_body_text(message: EmailMessage) -> str:
    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                continue
            content_type = part.get_content_type()
            if content_type.startswith("text/"):
                payload = part.get_content()
                if isinstance(payload, bytes):
                    payload = payload.decode(part.get_content_charset("utf-8"), errors="ignore")
                return _truncate(payload)
        return ""
    payload = message.get_content()
    if isinstance(payload, bytes):
        payload = payload.decode(message.get_content_charset("utf-8"), errors="ignore")
    return _truncate(payload)


def _truncate(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_BODY_BYTES:
        return text
    truncated = encoded[:MAX_BODY_BYTES]
    return truncated.decode("utf-8", errors="ignore")

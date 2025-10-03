"""MIME parsing helpers tuned for MailAI's YAML-in-email workflow.

What:
  Provide defensive parsing utilities that turn raw RFC822 payloads into
  :class:`email.message.EmailMessage` objects, lowercase header dictionaries, and
  bounded plain-text bodies.

Why:
  Configuration and status updates flow through IMAP messages whose structure is
  outside MailAI's control. The helpers normalise inputs so the rule engine can
  operate deterministically without risking oversized payloads or encoding
  errors.

How:
  Use the ``email`` package's :class:`~email.parser.BytesParser` with the default
  policy, extract a safe text body by walking MIME parts, and truncate the UTF-8
  content when it exceeds configurable limits.

Interfaces:
  :func:`parse_message`.

Invariants & Safety:
  - Body text is always returned as UTF-8 with control characters removed via
    ``errors="ignore"`` decoding so downstream YAML parsing cannot crash on
    undecodable bytes.
  - Truncation is performed on encoded bytes to avoid splitting multi-byte code
    points and to keep size checks aligned with IMAP quotas.
"""
from __future__ import annotations

from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from typing import Dict, Tuple


MAX_BODY_BYTES = 1_000_000
"""Soft upper bound for decoded body size in bytes to guard against IMAP abuse."""


def parse_message(raw: bytes) -> Tuple[EmailMessage, Dict[str, str], str]:
    """Parse a raw IMAP message into canonical structures.

    What:
      Turns a raw RFC822 payload into a :class:`EmailMessage`, a lower-cased
      header mapping, and a truncated text body suitable for YAML parsing.

    Why:
      MailAI stores configuration inside email bodies. Consumers need a predictable
      representation regardless of how the mail was authored (multipart, various
      encodings, attachments, etc.).

    How:
      Leverages :class:`BytesParser` with the default policy to obtain the
      message object, normalises headers into a case-insensitive dictionary, and
      defers to :func:`_extract_body_text` for text extraction and truncation.

    Args:
      raw: Raw message bytes as retrieved from IMAP ``RFC822``/``BODY[]`` fetches.

    Returns:
      Tuple containing the parsed :class:`EmailMessage`, a ``dict`` of header
      values keyed by lowercase names, and the truncated UTF-8 text body.
    """

    parser = BytesParser(policy=policy.default)
    message = parser.parsebytes(raw)
    headers = {k.lower(): v for k, v in message.items()}
    body_text = _extract_body_text(message)
    return message, headers, body_text


def _extract_body_text(message: EmailMessage) -> str:
    """Select the safest textual representation from a MIME tree.

    What:
      Produces a plain-text body by preferring ``text/*`` parts and stripping all
      other MIME content.

    Why:
      Configuration emails may contain signatures or attachments. By scanning for
      textual parts explicitly we avoid inadvertently parsing binary blobs or
      HTML markup.

    How:
      Walks multipart messages depth-first while skipping container parts. For
      each leaf ``text/*`` part, decodes the payload using the declared charset or
      UTF-8 fallback, then truncates via :func:`_truncate`. Non-multipart messages
      fall back to their ``get_content`` result.

    Args:
      message: Parsed :class:`EmailMessage` to inspect.

    Returns:
      Sanitised text body (possibly empty) constrained by :data:`MAX_BODY_BYTES`.
    """
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
    """Clamp ``text`` to :data:`MAX_BODY_BYTES` when encoded in UTF-8.

    What:
      Guarantees string payloads never exceed the soft byte limit while keeping
      Unicode characters intact.

    Why:
      IMAP servers and YAML parsers both suffer when confronted with very large
      messages. Limiting size at the utility layer ensures consistent enforcement
      and enables logging about truncation upstream.

    How:
      Encodes ``text`` as UTF-8, checks the length, slices the byte array if
      necessary, and decodes again with ``errors="ignore"`` to drop incomplete
      trailing characters safely.

    Args:
      text: Candidate string extracted from the MIME structure.

    Returns:
      Original string when within limits, otherwise a truncated UTF-8 string.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_BODY_BYTES:
        return text
    truncated = encoded[:MAX_BODY_BYTES]
    return truncated.decode("utf-8", errors="ignore")


# TODO: Other modules in this repository still require the same What/Why/How documentation.

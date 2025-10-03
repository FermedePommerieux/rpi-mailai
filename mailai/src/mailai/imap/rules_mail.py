"""Manage discovery and bootstrap of the ``MailAI: rules.yaml`` message.

What:
  Locate the canonical rules mail, parse its YAML payload, compute checksums, and
  generate minimal templates when the message is missing.

Why:
  Configuration for MailAI is delivered via IMAP and treated as immutable. The
  runtime must reliably locate, validate, and (if required) recreate the
  ``rules.yaml`` anchor message to drive reload and learner workflows.

How:
  Uses :class:`MailAIImapClient` context managers to search and append mails,
  normalises message bodies for checksum stability, and extracts metadata needed
  by watcher modules.

Interfaces:
  :class:`RulesMailRef`, :func:`find_latest`, and :func:`append_minimal_template`.

Invariants & Safety:
  - Always operate in read-only sessions when fetching to avoid accidental flag
    updates.
  - Enforce UTF-8 fallbacks for body parsing so corruption does not abort the
    reload cycle.
  - YAML payloads are normalised to Unix newlines before hashing to remain stable
    across providers.
"""
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
    """Lightweight descriptor for the canonical rules mail.

    What:
      Stores UID, metadata, and a normalised body string to support change
      detection and checksum comparisons.

    Why:
      Separating metadata from the heavy email payload keeps watcher logic fast
      and avoids repeated MIME parsing during polling.

    How:
      Capture the minimal IMAP fields needed by the watcher: UID, message-id,
      internal date, charset, normalised body text, and a MailAI checksum. The
      dataclass is populated by :func:`find_latest`, which performs MIME parsing
      and newline normalisation before constructing the instance.

    Attributes:
      uid: UID of the rules message.
      message_id: Optional ``Message-ID`` header value.
      internaldate: Server-provided internal date for ordering.
      charset: Charset used to decode the body.
      body_text: Normalised body string (Unix newlines, trailing newline).
      checksum: Digest of ``body_text`` used for change detection.
    """

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
    """Fetch the latest rules mail reference from IMAP.

    What:
      Searches the target ``folder`` for mails matching ``subject`` and returns a
      :class:`RulesMailRef` describing the newest match.

    Why:
      The watcher subsystem needs metadata to detect configuration changes
      without repeatedly parsing the full message body.

    How:
      Uses :meth:`MailAIImapClient.session` in read-only mode, issues a UID
      search, fetches headers/body, and normalises content and timestamps before
      computing a checksum.

    Args:
      subject: Subject override; defaults to configured ``mail.rules.subject``.
      folder: Folder override; defaults to configured ``mail.rules.folder``.
      client: Connected :class:`MailAIImapClient`.

    Returns:
      A :class:`RulesMailRef` for the latest message or ``None`` if no match was
      found.
    """

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
    """Create a minimal ``rules.yaml`` message if one is missing.

    What:
      Appends a YAML template email and returns its :class:`RulesMailRef`.

    Why:
      Initial deployments or disaster recovery may find the rules mail missing;
      bootstrapping a template gives operators a starting point.

    How:
      Builds a minimal :class:`RulesV2` document, renders it to YAML, and appends
      it to the designated folder before delegating to :func:`find_latest` to
      return metadata.

    Args:
      folder: Optional override for the target folder.
      client: Connected IMAP client used for the append.

    Returns:
      :class:`RulesMailRef` for the newly appended message.
    """

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
    """Standardise line endings and ensure trailing newline for hashing.

    What:
      Converts CRLF to LF, strips trailing whitespace, and appends a single
      newline.

    Why:
      Keeps checksum calculations stable across providers that adjust line
      endings.

    Args:
      text: Raw body text extracted from the message.

    Returns:
      Normalised text ready for hashing and storage.
    """

    return text.replace("\r\n", "\n").rstrip() + "\n"


def _parse_internaldate(value: object) -> Optional[datetime]:
    """Parse ``INTERNALDATE`` responses into :class:`datetime` values.

    What:
      Attempts to convert common ``imapclient`` return types into timezone-aware
      datetimes.

    Why:
      Some servers return bytes while others return strings; this helper keeps
      handling consistent without leaking parsing errors to callers.

    Args:
      value: Raw value from ``IMAPClient.fetch``.

    Returns:
      Parsed :class:`datetime` or ``None`` if parsing fails.
    """

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
    """Return the first present ``data`` value for ``keys`` as bytes.

    What:
      Iterates through potential fetch keys and returns the first value coerced
      to bytes.

    Why:
      Different IMAP servers emit slightly different keys (e.g., ``BODY[HEADER]``
      vs ``BODY.PEEK[HEADER]``). This helper smooths those differences.

    Args:
      data: Fetch dictionary returned by ``IMAPClient``.
      *keys: Candidate keys ordered by priority.

    Returns:
      Matching value as bytes, or ``None`` if absent.
    """

    for key in keys:
        if key in data:
            value = data[key]
            if isinstance(value, bytes):
                return value
            if isinstance(value, str):
                return value.encode("utf-8")
    return None


# TODO: Other modules in this repository still require the same What/Why/How documentation.

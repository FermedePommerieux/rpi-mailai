"""mailai.core.features.extract

What:
  Assemble the full feature record for an email message by combining hashed
  structural features with the optional intent enrichment pipeline.

Why:
  Downstream learners and audit tooling need a single, privacy-compliant object
  that captures both traditional hashed artefacts and the higher-level intent
  scores introduced in this iteration. Centralising the orchestration keeps
  privacy guards and configuration handling consistent.

How:
  - Derive hashed structural features using :mod:`mailai.core.features`.
  - Compute sanitised metadata, text statistics, and URL summaries to feed the
    intent heuristics.
  - Invoke :func:`infer_intent_and_tone` when enabled and merge the result into a
    :class:`MailFeatureRecord`, enforcing quarantine thresholds.

Interfaces:
  - :func:`build_mail_feature_record`

Invariants & Safety:
  - Raw bodies are discarded immediately after computing aggregate statistics.
  - Intent enrichment is only executed when explicitly enabled in configuration.
  - Quarantine decisions are derived solely from bounded scores.
"""
from __future__ import annotations

import email.utils
import re
from typing import Iterable, Mapping, Optional, Tuple

from .intent_extract import infer_intent_and_tone
from .schema import (
    IntentFeatureSettings,
    IntentFeatures,
    IntentLLMSettings,
    MailFeatureRecord,
    ParsedMailMeta,
    StructuredLLM,
    TextStats,
    UrlInfo,
)
from ...utils.mime import parse_message

_CALL_TO_ACTION_KEYWORDS: Tuple[str, ...] = (
    "act now",
    "click",
    "confirm",
    "verify",
    "limited time",
    "respond",
)
_ATTACHMENT_KEYWORDS: Tuple[str, ...] = (
    "attachment",
    "attached",
    "document",
    "invoice",
)
_URL_RE = re.compile(r"https?://([^/\s]+)", re.IGNORECASE)


def build_mail_feature_record(
    raw_message: bytes,
    *,
    pepper: bytes,
    salt: bytes,
    intent_settings: Optional[IntentFeatureSettings] = None,
    llm: Optional[StructuredLLM] = None,
) -> MailFeatureRecord:
    """Build a :class:`MailFeatureRecord` from an RFC822 message.

    What:
      Hash the structural features of ``raw_message`` and, when configured,
      augment the record with intent enrichment metadata and quarantine signals.

    Why:
      Centralising this logic ensures that privacy guards, LLM invocation, and
      threshold enforcement remain consistent regardless of the caller (training
      pipeline, rule synthesis, or tests).

    How:
      - Call :func:`extract_features` to produce the hashed baseline mapping.
      - Optionally compute enrichment metadata and invoke
        :func:`infer_intent_and_tone`.
      - Convert the resulting dictionary into :class:`IntentFeatures` and derive
        the quarantine decision using configured thresholds.

    Args:
      raw_message: RFC822 message bytes.
      pepper: Secret pepper for hashing helpers.
      salt: Public salt used alongside the pepper.
      intent_settings: Optional configuration enabling intent enrichment.
      llm: Optional structured LLM object implementing ``structured_completion``.

    Returns:
      Fully populated :class:`MailFeatureRecord` with hashed and enriched data.

    Raises:
      PrivacyViolation: When the LLM returns data outside the closed vocabularies.
    """

    from . import extract_features as _extract_features

    hashed = _extract_features(raw_message, pepper=pepper, salt=salt)
    intent_features: Optional[IntentFeatures] = None
    should_quarantine = False

    if intent_settings and intent_settings.enabled:
        meta, stats, urls = _prepare_intent_inputs(
            raw_message,
            hashed,
            intent_settings.llm,
            llm if llm and hasattr(llm, "structured_completion") else None,
        )
        enriched = infer_intent_and_tone(meta, stats, urls)
        intent_features = IntentFeatures(
            intent=enriched["intent"],
            speech_acts=tuple(enriched["speech_acts"]),
            persuasion=tuple(enriched["persuasion"]),
            urgency_score=int(enriched["urgency_score"]),
            insistence_score=int(enriched["insistence_score"]),
            commercial_pressure=int(enriched["commercial_pressure"]),
            suspicion_flags=tuple(enriched["suspicion_flags"]),
            scam_singularity=int(enriched["scam_singularity"]),
        )
        thresholds = intent_settings.thresholds
        should_quarantine = (
            intent_features.scam_singularity >= thresholds.scam_singularity_quarantine
        )

    return MailFeatureRecord(
        hashed_features=hashed,
        intent_features=intent_features,
        should_quarantine=should_quarantine,
    )


def _prepare_intent_inputs(
    raw_message: bytes,
    hashed: Mapping[str, object],
    llm_settings: IntentLLMSettings,
    llm: Optional[StructuredLLM],
) -> Tuple[ParsedMailMeta, TextStats, UrlInfo]:
    """Parse the message and compute metadata required for intent extraction."""

    _, headers, body = parse_message(raw_message)
    from_domain = _extract_from_domain(headers.get("from", ""))
    subject = headers.get("subject", "")
    reply_depth = _count_tokens(subject, ("re:",))
    relance_count = _count_tokens(subject, ("fw:", "fwd:"))
    has_attachments = bool(hashed.get("has_attachment", False))
    body_lower = body.lower()
    caps_ratio = _caps_ratio(body)
    exclamation_density = body.count("!") / max(1, len(body))
    call_to_action_score = _keyword_hits(body_lower, _CALL_TO_ACTION_KEYWORDS)
    attachment_mentions = _keyword_hits(body_lower, _ATTACHMENT_KEYWORDS)
    attachment_pressure = attachment_mentions > 0 or "attach" in subject.lower()
    target_domains = tuple(dict.fromkeys(domain.lower() for domain in _URL_RE.findall(body)))

    meta = ParsedMailMeta(
        from_domain=from_domain,
        reply_depth=reply_depth,
        relance_count=relance_count,
        has_attachments=has_attachments,
        attachment_pressure=attachment_pressure,
        llm=llm,
        llm_settings=llm_settings,
    )
    stats = TextStats(
        length=len(body),
        caps_ratio=caps_ratio,
        exclamation_density=exclamation_density,
        call_to_action_score=call_to_action_score,
        attachment_mentions=attachment_mentions,
    )
    urls = UrlInfo(target_domains=target_domains)
    del body  # SAFETY: Ensure raw body text is not retained beyond this scope.
    return meta, stats, urls


def _extract_from_domain(field: str) -> str:
    """Extract the sender domain from a ``From`` header."""

    addresses = email.utils.getaddresses([field])
    if not addresses:
        return ""
    address = addresses[0][1]
    if "@" not in address:
        return ""
    return address.split("@", 1)[1].lower()


def _keyword_hits(text: str, keywords: Iterable[str]) -> int:
    """Count keyword occurrences in ``text`` without storing the matches."""

    total = 0
    for keyword in keywords:
        total += text.count(keyword)
    return total


def _count_tokens(subject: str, tokens: Tuple[str, ...]) -> int:
    """Count how many times ``tokens`` appear in ``subject`` (case-insensitive)."""

    lower = subject.lower()
    return sum(lower.count(token) for token in tokens)


def _caps_ratio(text: str) -> float:
    """Compute the uppercase ratio among alphabetic characters."""

    alphabetic = [char for char in text if char.isalpha()]
    if not alphabetic:
        return 0.0
    upper = sum(1 for char in alphabetic if char.isupper())
    return upper / len(alphabetic)

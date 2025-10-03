"""mailai.core.features

What:
  Provide deterministic feature extraction helpers that transform raw email
  messages into privacy-preserving sketches used by the learner and rule
  engine.

Why:
  MailAI must inspect message structure without storing readable content. The
  feature layer converts MIME payloads into hashed artefacts so downstream
  components can classify behaviour while upholding confidentiality promises.

How:
  - Parse MIME messages using hardened utilities that normalise headers and
    body encodings.
  - Tokenise text to a canonical lower-case form before hashing with a peppered
    hasher designed to resist dictionary attacks.
  - Derive aggregate signals such as sender domains, attachment presence, and
    simhash-based sketches that capture semantic similarity without exposing
    raw text.

Interfaces:
  - :class:`FeatureSketch` describing hashed token windows.
  - :func:`extract_features` returning the feature map for an entire message.
  - :func:`hash_text_window` producing sketches for textual snippets.

Invariants & Safety:
  - Pepper/salt pairs must be supplied by the caller so hashing remains
    reproducible but resistant to cross-run linkage without the secret pepper.
  - Only hashed or aggregate representations leave this module; raw body text
    never returns to callers.
"""
from __future__ import annotations

import email.utils
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from ...utils.mime import parse_message
from .extract import build_mail_feature_record
from ..privacy import PepperHasher
from .schema import (
    IntentFeatureSettings,
    IntentFeatures,
    IntentLLMSettings,
    IntentThresholds,
    MailFeatureRecord,
    ParsedMailMeta,
    TextStats,
    UrlInfo,
)


@dataclass
class FeatureSketch:
    """Hashed representation of a token window.

    What:
      Store a sample of hashed tokens alongside a simhash fingerprint that
      captures coarse semantic similarity of a message body or snippet.

    Why:
      Downstream classifiers require stable yet privacy-preserving sketches to
      detect near-duplicates without leaking original text.

    How:
      Retain the first 128 hashed tokens for inspection while computing an
      integer simhash value derived from the full token stream.

    Attributes:
      tokens: Prefix sample of hashed tokens for quick comparisons.
      simhash: 64-bit simhash summarising the token set.
    """

    tokens: List[str]
    simhash: int


def extract_features(raw_message: bytes, *, pepper: bytes, salt: bytes) -> Dict[str, object]:
    """Derive privacy-preserving features from a raw email message.

    What:
      Parse a MIME message and emit hashed attributes describing sender domains,
      attachment state, size, subject fingerprint, and a body sketch.

    Why:
      Feature extraction sits between sensitive email content and learning
      components, ensuring no plaintext is persisted while still enabling
      behavioural detection and heuristics.

    How:
      - Parse the message via :func:`mailai.utils.mime.parse_message` to obtain
        headers and body text.
      - Tokenise the body, hash the tokens using :class:`PepperHasher`, and
        compute a :class:`FeatureSketch`.
      - Extract address domains using ``email.utils`` to handle RFC-compliant
        lists.
      - Produce a dictionary of hashed/aggregated signals for downstream use.

    Args:
      raw_message: RFC822 message bytes fetched from IMAP.
      pepper: Secret pepper shared with :class:`PepperHasher` for hashing.
      salt: Public salt differentiating different feature pipelines.

    Returns:
      Mapping with derived features safe to persist and analyse.
    """

    _, headers, body = parse_message(raw_message)
    domains = _extract_domains(headers)
    hasher = PepperHasher(pepper=pepper, salt=salt)
    tokens = _tokenise(body)
    hashed = hasher.hash_tokens(tokens)
    sketch = FeatureSketch(tokens=hashed[:128], simhash=hasher.simhash(tokens))
    return {
        "domains": domains,
        "list_id": headers.get("list-id"),
        "has_attachment": "content-disposition" in headers and "attachment" in headers.get("content-disposition", "").lower(),
        "size": len(raw_message),
        "subject_hash": hasher.simhash(_tokenise(headers.get("subject", ""))),
        "sketch": sketch,
    }


def hash_text_window(text: str, *, pepper: bytes, salt: bytes) -> FeatureSketch:
    """Compute a hashed sketch for an arbitrary text snippet.

    What:
      Transform a short string into a :class:`FeatureSketch` using the same
      privacy-preserving hashing primitives as :func:`extract_features`.

    Why:
      Utility callers (e.g. heuristics or tests) need to inspect how snippets
      map into hashed space without reimplementing pepper/salt logic.

    How:
      Tokenise the text, hash via :class:`PepperHasher`, and return the sketch
      containing the leading hashed tokens and simhash fingerprint.

    Args:
      text: String to hash.
      pepper: Secret pepper ensuring irreversibility.
      salt: Salt differentiating hashing contexts.

    Returns:
      A :class:`FeatureSketch` ready for comparison.
    """

    hasher = PepperHasher(pepper=pepper, salt=salt)
    tokens = _tokenise(text)
    hashed = hasher.hash_tokens(tokens)
    return FeatureSketch(tokens=hashed[:128], simhash=hasher.simhash(tokens))


def _tokenise(text: str) -> List[str]:
    """Normalise and split text into lower-case tokens.

    What:
      Canonicalise whitespace and case before splitting to produce stable token
      sequences for hashing.

    Why:
      Consistent tokenisation ensures simhash stability and prevents attackers
      from bypassing hashing by injecting inconsistent casing or spacing.

    How:
      Collapse whitespace to single spaces, lowercase the content, and filter
      empty tokens.

    Args:
      text: Source text potentially containing arbitrary whitespace.

    Returns:
      List of normalised tokens.
    """

    normalized = " ".join(text.lower().split())
    return [token for token in normalized.split(" ") if token]


def _extract_domains(headers: Dict[str, str]) -> Dict[str, int]:
    """Count sender/recipient domains from RFC822 headers.

    What:
      Inspect address-bearing headers and produce a frequency map of domains.

    Why:
      Domain frequency signals feed rules and learning models that distinguish
      newsletters, vendors, or suspicious sources without storing individual
      addresses.

    How:
      Iterate over common address headers, parse RFC822 address lists, extract
      the domain portion, and increment counts in a case-insensitive manner.

    Args:
      headers: Mapping of message headers produced by the MIME parser.

    Returns:
      Dictionary keyed by lower-case domain names with occurrence counts.
    """

    domains: Counter[str] = Counter()
    for field in ("from", "to", "cc", "bcc"):
        value = headers.get(field)
        if not value:
            continue
        addresses = email.utils.getaddresses([value])
        for _, address in addresses:
            domain = address.split("@")[-1] if "@" in address else address
            if domain:
                domains[domain.lower()] += 1
    return dict(domains)


__all__ = [
    "FeatureSketch",
    "extract_features",
    "hash_text_window",
    "build_mail_feature_record",
    "IntentFeatureSettings",
    "IntentLLMSettings",
    "IntentThresholds",
    "IntentFeatures",
    "MailFeatureRecord",
    "ParsedMailMeta",
    "TextStats",
    "UrlInfo",
]


# TODO: Other modules require the same treatment (What/Why/How docstrings + module header).

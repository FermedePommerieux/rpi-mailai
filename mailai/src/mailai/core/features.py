"""Feature extraction pipelines for MailAI."""
from __future__ import annotations

import email.utils
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from ..utils.mime import parse_message
from .privacy import PepperHasher


@dataclass
class FeatureSketch:
    """Non-reversible sketch of textual content."""

    tokens: List[str]
    simhash: int


def extract_features(raw_message: bytes, *, pepper: bytes, salt: bytes) -> Dict[str, object]:
    """Extract hashed features from a raw email message."""

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
    """Produce a hashed sketch from a text snippet."""

    hasher = PepperHasher(pepper=pepper, salt=salt)
    tokens = _tokenise(text)
    hashed = hasher.hash_tokens(tokens)
    return FeatureSketch(tokens=hashed[:128], simhash=hasher.simhash(tokens))


def _tokenise(text: str) -> List[str]:
    normalized = " ".join(text.lower().split())
    return [token for token in normalized.split(" ") if token]


def _extract_domains(headers: Dict[str, str]) -> Dict[str, int]:
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

"""Generate run identifiers and stable checksums for MailAI artifacts.

What:
  Provide minimal helpers for creating unique run IDs and SHA-256 checksums used
  throughout the project for logging and deduplication.

Why:
  Centralising the logic avoids subtle inconsistencies (e.g., timestamp formats
  or hash prefixes) that would otherwise complicate audits and change detection.

How:
  Combines ISO8601 timestamps with random suffixes for IDs and wraps ``hashlib``
  with a consistent ``sha256:`` prefix for checksums.

Interfaces:
  :func:`new_run_id` and :func:`checksum`.

Invariants & Safety:
  - Run IDs always include timezone-aware timestamps for traceability.
  - Checksums are namespaced with ``sha256:`` so future algorithms can coexist.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone


def new_run_id() -> str:
    """Return a unique identifier for background runs.

    What:
      Emits an ISO8601 timestamp suffixed with a six-hex-character random token.

    Why:
      Run IDs appear in telemetry and filenames; combining time and randomness
      keeps them sortable while avoiding collisions in concurrent executions.

    How:
      Captures ``datetime.now(timezone.utc)`` for explicit timezone context and
      concatenates a ``secrets.token_hex`` suffix.

    Returns:
      Unique identifier string (e.g., ``2024-01-01T00:00:00+00:00#1a2b3c``).
    """

    timestamp = datetime.now(timezone.utc).isoformat()
    suffix = secrets.token_hex(3)
    return f"{timestamp}#{suffix}"


def checksum(data: bytes) -> str:
    """Compute a namespaced SHA-256 digest for ``data``.

    What:
      Wraps :func:`hashlib.sha256` and adds a ``sha256:`` prefix.

    Why:
      The prefix signals which hash algorithm was used, enabling future upgrades
      without ambiguity.

    How:
      Feed ``data`` into :func:`hashlib.sha256`, obtain the hexadecimal digest,
      and concatenate the ``sha256:`` namespace prefix so downstream consumers
      can distinguish the algorithm without parsing auxiliary metadata.

    Args:
      data: Bytes to hash.

    Returns:
      Hex-encoded digest string prefixed with ``sha256:``.
    """

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


# TODO: Other modules in this repository still require the same What/Why/How documentation.

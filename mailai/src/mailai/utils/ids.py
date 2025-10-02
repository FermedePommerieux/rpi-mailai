"""Helpers for creating identifiers and checksums."""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone


def new_run_id() -> str:
    """Create a unique run identifier."""

    timestamp = datetime.now(timezone.utc).isoformat()
    suffix = secrets.token_hex(3)
    return f"{timestamp}#{suffix}"


def checksum(data: bytes) -> str:
    """Return a sha256 checksum string for the provided data."""

    return f"sha256:{hashlib.sha256(data).hexdigest()}"

"""Utilities for building IMAP search queries."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple


def build_search(filters: Dict[str, object]) -> List[Tuple[str, object]]:
    """Convert a dictionary of filters into IMAP search criteria."""

    criteria: List[Tuple[str, object]] = []
    for key, value in filters.items():
        if value is None:
            continue
        if key == "since" and isinstance(value, datetime):
            criteria.append(("SINCE", value))
        elif key == "before" and isinstance(value, datetime):
            criteria.append(("BEFORE", value))
        elif key == "unseen" and isinstance(value, bool) and value:
            criteria.append(("UNSEEN", ""))
        elif key == "flagged" and isinstance(value, bool) and value:
            criteria.append(("FLAGGED", ""))
        elif key == "mailbox":
            criteria.append(("MAILBOX", value))
    return criteria

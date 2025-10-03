"""Translate high-level filter dictionaries into IMAP search criteria.

What:
  Provide a deterministic mapping from MailAI filter dictionaries to the tuple
  lists consumed by ``imapclient`` search operations.

Why:
  Keeping the translation logic centralised ensures IMAP queries remain
  consistent across modules (rule evaluation, health checks) and facilitates unit
  testing for tricky date handling.

How:
  Iterates through the filter dictionary, converting supported keys into IMAP
  keyword/value pairs while ignoring ``None`` entries.

Interfaces:
  :func:`build_search`.

Invariants & Safety:
  - Only whitelisted keys are translated to prevent injection into IMAP search
    syntax.
  - Boolean toggles emit keywords without values as expected by ``imapclient``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple


def build_search(filters: Dict[str, object]) -> List[Tuple[str, object]]:
    """Convert MailAI filter dictionaries into IMAP-compatible criteria.

    What:
      Inspects ``filters`` for supported keys (``since``, ``before``, ``unseen``,
      ``flagged``, ``mailbox``) and builds an ordered list of criteria tuples.

    Why:
      IMAP search syntax is positional and picky about argument formats. This
      helper ensures date objects are passed through unchanged and boolean flags
      emit the right keywords only when true.

    How:
      Skips ``None`` values, checks the type of each filter, and appends the
      corresponding tuple to the criteria list.

    Args:
      filters: User-specified filter mapping.

    Returns:
      List of ``(keyword, value)`` tuples suitable for ``imapclient`` search.
    """

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


# TODO: Other modules in this repository still require the same What/Why/How documentation.

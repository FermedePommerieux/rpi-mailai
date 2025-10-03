"""Configuration change detection helpers.

What:
  Provide predicates for comparing stored configuration references with fresh
  IMAP rule mails, returning both boolean change signals and human-readable
  reasons.

Why:
  MailAI treats IMAP messages as immutable; configuration updates are observed
  through UID and checksum changes instead of in-place edits. Centralizing the
  comparison logic keeps the heuristics consistent across watchers and tests.

How:
  Accepts two lightweight descriptors—one from persistent status, one from the
  mailbox scanner—and performs explicit field comparisons in priority order to
  avoid misclassifying mailbox replays.

Interfaces:
  ``has_changed`` and ``change_reason``.

Invariants & Safety:
  - ``None`` values are handled gracefully to avoid crashes on bootstrap.
  - UID changes take precedence over checksum changes because IMAP servers may
    recycle checksums but never UIDs for the same message.
"""
from __future__ import annotations

from typing import Optional

from .schema import ConfigReference
from ..imap.rules_mail import RulesMailRef


def has_changed(prev: Optional[ConfigReference], new: Optional[RulesMailRef]) -> bool:
    """Check whether the effective configuration source differs.

    What:
      Returns ``True`` when the stored ``ConfigReference`` no longer matches the
      most recent rules mail metadata.

    Why:
      Drives downstream reload workflows and LLM warmups—false negatives would
      skip required refreshes, while false positives would thrash resources.

    How:
      Evaluates ``None`` cases first, then compares UID before checksum because
      UID mutations indicate a brand-new message even if the content is
      identical.

    Args:
      prev: Previously persisted configuration reference, if any.
      new: Freshly observed rules mail descriptor from IMAP polling.

    Returns:
      ``True`` when the references differ, ``False`` otherwise.
    """

    if new is None:
        return prev is not None
    if prev is None:
        return True
    if prev.uid != new.uid:
        return True
    return prev.checksum != new.checksum


def change_reason(prev: Optional[ConfigReference], new: Optional[RulesMailRef]) -> str:
    """Explain why a configuration change was detected.

    What:
      Provides a short, operator-friendly summary used in status logs and
      telemetry dashboards.

    Why:
      Transparent reasoning helps debug false triggers and confirms that UID
      and checksum heuristics behave as expected on different IMAP providers.

    How:
      Mirrors :func:`has_changed` logic while mapping each branch to a concise
      label. ``None`` cases are distinguished to differentiate bootstrap from
      missing configuration.

    Args:
      prev: Stored configuration reference, if available.
      new: Latest IMAP rules mail descriptor, if any.

    Returns:
      Machine-parseable reason string describing the change category.
    """

    if prev is None and new is None:
        return "missing"
    if prev is None:
        return "bootstrap"
    if new is None:
        return "missing"
    if prev.uid != new.uid:
        return "uid change"
    if prev.checksum != new.checksum:
        return "checksum change"
    return "unchanged"


# TODO: Document remaining modules with the same What/Why/How structure.

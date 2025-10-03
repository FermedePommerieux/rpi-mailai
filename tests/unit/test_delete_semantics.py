"""
Module: tests/unit/test_delete_semantics.py

What:
    Validate heuristics that map soft deletion signals to structured responses
    used by the core engine for user-facing audit trails.

Why:
    The deletion semantics model guards against overzealous message removal by
    requiring consistent scoring and human-readable rationales before an IMAP
    purge is attempted.

How:
    This suite exercises the ``infer`` helper with representative signals to
    assert score ranges and descriptive text remain stable across refactors.

Interfaces:
    test_delete_semantics_interprets_signals

Invariants & Safety Rules:
    - Each signal must map to a deterministic score for consistent policy
      enforcement.
    - Reasons returned must stay descriptive to support operator audit logs.
"""

import pytest

from mailai.core import delete_semantics


def test_delete_semantics_interprets_signals():
    """
    What:
        Ensure ``delete_semantics.infer`` decorates raw signals with stable scores
        and human-readable rationales.

    Why:
        Predictable outputs are critical for audit log review and preventing
        accidental message deletion when signal interpretation changes.

    How:
        Feed the helper a representative signal list, then assert on both the
        returned textual reasons and approximate score values for each entry.

    Returns:
        None
    """
    signals = ["spam_score_high", "invite_expired", "thread_is_resolved"]
    result = delete_semantics.infer(signals)
    assert result["spam_score_high"].reason == "Spam classifier confidence high"
    assert result["invite_expired"].score == pytest.approx(0.85, rel=1e-3)
    assert "Resolved thread" in result["thread_is_resolved"].reason


# TODO: Other modules in this repository still require the same What/Why/How documentation.

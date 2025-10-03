"""Module: tests/unit/test_intent_features.py

What:
    Validate the privacy guards and heuristic behaviour of the intent enrichment
    layer.

Why:
    The enrichment pipeline must never leak free-form text and must flag risky
    patterns deterministically. These unit tests enforce the closed-vocabulary
    and bounded-score guarantees while checking representative heuristics.

How:
    - Exercise :func:`assert_closed_vocab` and :func:`assert_bounded_scores` to
      ensure violations raise :class:`PrivacyViolation`.
    - Feed sanitised metadata into :func:`infer_intent_and_tone` and assert that
      link-domain mismatches trigger the expected suspicion flag.
"""

import pytest

from mailai.core.features.intent_extract import infer_intent_and_tone
from mailai.core.features.schema import IntentLLMSettings, ParsedMailMeta, TextStats, UrlInfo
from mailai.utils.privacy import PrivacyViolation, assert_bounded_scores, assert_closed_vocab


def test_closed_vocab_enforcement() -> None:
    """What/Why/How: closed vocabulary violations must raise."""

    with pytest.raises(PrivacyViolation):
        assert_closed_vocab("intent", ["freeform"], allowed=("unknown",))


def test_bounded_scores_enforcement() -> None:
    """What/Why/How: bounded scores outside the range are rejected."""

    with pytest.raises(PrivacyViolation):
        assert_bounded_scores("urgency_score", 5, upper=3)


def test_link_mismatch_flag() -> None:
    """What/Why/How: heuristics must mark mismatched URLs as suspicious."""

    meta = ParsedMailMeta(
        from_domain="sender.example",
        reply_depth=0,
        relance_count=0,
        has_attachments=False,
        attachment_pressure=False,
        llm=None,
        llm_settings=IntentLLMSettings(),
    )
    stats = TextStats(
        length=120,
        caps_ratio=0.0,
        exclamation_density=0.0,
        call_to_action_score=0,
        attachment_mentions=0,
    )
    urls = UrlInfo(target_domains=("phish.test",))
    result = infer_intent_and_tone(meta, stats, urls)
    assert "link_mismatch" in result["suspicion_flags"]

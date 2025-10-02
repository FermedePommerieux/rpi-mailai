import pytest

from mailai.core import delete_semantics


def test_delete_semantics_interprets_signals():
    signals = ["spam_score_high", "invite_expired", "thread_is_resolved"]
    result = delete_semantics.infer(signals)
    assert result["spam_score_high"].reason == "Spam classifier confidence high"
    assert result["invite_expired"].score == pytest.approx(0.85, rel=1e-3)
    assert "Resolved thread" in result["thread_is_resolved"].reason

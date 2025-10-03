"""Module: tests/e2e/test_intent_enrichment.py

What:
    Exercise the full feature extraction stack to ensure enriched records persist
    only closed-vocabulary identifiers and bounded scores.

Why:
    The new intent pipeline must never leak plaintext while still driving the
    quarantine rule. An end-to-end check using real feature extraction guards
    against regressions when configuration or heuristics change.

How:
    - Build a synthetic RFC822 message with mismatched URLs and urgent cues.
    - Run :func:`build_mail_feature_record` with enrichment enabled and a stub LLM
      that returns whitelisted JSON.
    - Assert that the resulting record contains only IDs/scores and that the
      quarantine threshold is honoured.
"""

import json

from mailai.core.features import (
    IntentFeatureSettings,
    IntentFeatures,
    IntentLLMSettings,
    IntentThresholds,
    build_mail_feature_record,
)
class _DeterministicLLM:
    """Stub LLM returning a fixed, valid JSON payload."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def structured_completion(self, prompt: str, *, max_tokens: int, temperature: float, timeout_s: float) -> str:
        del prompt, max_tokens, temperature, timeout_s
        return self._payload


def test_enriched_record_contains_only_ids_and_scores() -> None:
    """What/Why/How: enriched records avoid plaintext and trigger quarantine."""

    message = (
        "From: urgent@sender.com\r\n"
        "To: user@example.com\r\n"
        "Subject: URGENT ACTION REQUIRED\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        "PLEASE CLICK IMMEDIATELY!!! http://phish.example/login\r\n"
    ).encode("utf-8")
    llm_response = json.dumps(
        {
            "intent": "fraud_solicitation",
            "speech_acts": ["warn"],
            "persuasion": ["fear"],
            "urgency_score": 3,
            "insistence_score": 2,
            "commercial_pressure": 1,
            "suspicion_flags": ["link_mismatch"],
            "scam_singularity": 3,
        }
    )
    llm = _DeterministicLLM(llm_response)
    settings = IntentFeatureSettings(
        enabled=True,
        llm=IntentLLMSettings(max_tokens=64, temperature=0.0, timeout_s=3.0),
        thresholds=IntentThresholds(scam_singularity_quarantine=2, urgency_warn=3),
    )

    record = build_mail_feature_record(
        message,
        pepper=b"pepper",
        salt=b"salt",
        intent_settings=settings,
        llm=llm,
    )

    assert record.intent_features is not None
    intent: IntentFeatures = record.intent_features
    assert intent.intent == "fraud_solicitation"
    assert "link_mismatch" in intent.suspicion_flags
    assert record.should_quarantine is True

    serialised = json.dumps(record.as_dict(), default=str)
    assert "CLICK" not in serialised
    assert "phish.example" not in serialised
    assert "login" not in serialised

"""Heuristics for inferring delete semantics from lightweight signals.

What:
  Offer a conservative scoring table that estimates when automation can safely
  delete or archive a message based on derived signals from the engine.

Why:
  MailAI prioritizes privacy and must avoid accidental deletion; centralizing
  heuristic weights provides a transparent, auditable decision surface that can
  be tuned without retraining models.

How:
  Iterates over signal identifiers, mapping each to a ``DeleteSignal`` with a
  human-readable reason and confidence score. Unknown signals receive a low
  score, prompting manual review rather than automatic deletion.

Interfaces:
  ``DeleteSignal`` dataclass and ``infer`` function.

Invariants & Safety:
  - Confidence scores stay within ``[0.0, 1.0]`` to align with downstream
    thresholds.
  - Reasons are plain text for operator review; no raw email content is
    persisted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class DeleteSignal:
    """Describe a delete recommendation generated from heuristics.

    What:
      Stores a natural-language reason and a floating-point score representing
      confidence in deleting a message.

    Why:
      Separating score and explanation keeps the heuristics explainable and
      allows UX layers to surface rationales alongside automated actions.

    How:
      Simple dataclass with two fields; higher-level components interpret the
      score via configurable thresholds.
    """

    reason: str
    score: float


def infer(signals: Iterable[str]) -> Dict[str, DeleteSignal]:
    """Convert engine signals into delete recommendations.

    What:
      Returns a mapping from input signal identifiers to :class:`DeleteSignal`
      objects describing deletion rationales and confidence.

    Why:
      Provides a deterministic baseline when machine learning predictions are
      unavailable or need a human-auditable fallback.

    How:
      Iterates over the signals and assigns scores based on predefined rules.
      Unknown signals are mapped to a low score to prevent accidental message
      loss.

    Args:
      signals: Iterable of normalized signal names provided by the engine.

    Returns:
      Dictionary mapping signal names to their corresponding delete guidance.
    """

    result: Dict[str, DeleteSignal] = {}
    for signal in signals:
        if signal == "thread_is_resolved":
            result[signal] = DeleteSignal(reason="Resolved thread", score=0.6)
        elif signal == "calendar_invite_past":
            result[signal] = DeleteSignal(reason="Past event", score=0.8)
        elif signal == "spam_score_high":
            result[signal] = DeleteSignal(reason="Spam classifier confidence high", score=0.95)
        elif signal == "promotion_sender":
            result[signal] = DeleteSignal(reason="Promotional blast detected", score=0.75)
        elif signal == "conversation_closed":
            result[signal] = DeleteSignal(reason="Conversation archived by user", score=0.65)
        elif signal == "invite_expired":
            result[signal] = DeleteSignal(reason="Calendar invite no longer valid", score=0.85)
        elif signal == "list_unsubscribe_present":
            result[signal] = DeleteSignal(reason="Unsubscribe available", score=0.5)
        elif signal.startswith("age_gt_days"):
            result[signal] = DeleteSignal(reason="Message aged", score=0.4)
        else:
            result[signal] = DeleteSignal(reason="Unknown", score=0.1)
    return result


# TODO: Document remaining modules with the same What/Why/How structure.

"""Heuristics for inferring delete semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class DeleteSignal:
    """Outcome of delete semantic inference."""

    reason: str
    score: float


def infer(signals: Iterable[str]) -> Dict[str, DeleteSignal]:
    """Return heuristic delete decisions for a set of signals."""

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

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
        elif signal == "list_unsubscribe_present":
            result[signal] = DeleteSignal(reason="Unsubscribe available", score=0.5)
        elif signal.startswith("age_gt_days"):
            result[signal] = DeleteSignal(reason="Message aged", score=0.4)
        else:
            result[signal] = DeleteSignal(reason="Unknown", score=0.1)
    return result

"""Rule synthesis from learned weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class RuleCandidate:
    """Proposed rule emitted by the learner."""

    id: str
    description: str
    match: Dict[str, object]
    actions: List[Dict[str, object]]
    source: str = "learner"
    enabled: bool = False
    priority: int = 40


def synthesise_rules(weights: Dict[str, float], glossary: Dict[str, str], k_max: int) -> List[RuleCandidate]:
    """Create rule candidates based on weight magnitudes."""

    top_items = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:k_max]
    candidates: List[RuleCandidate] = []
    for idx, (feature, weight) in enumerate(top_items, start=1):
        description = f"Model-derived rule for feature {feature}"
        mailbox = glossary.get(feature, "INBOX")
        candidate = RuleCandidate(
            id=f"auto-{idx}",
            description=description,
            match={
                "any": [
                    {"category_pred": {"equals": mailbox, "prob_gte": 0.85}},
                    {"header": {"name": "List-Id", "contains": feature.split("=")[-1]}},
                ]
            },
            actions=[{"add_label": mailbox}, {"move_to": mailbox}],
        )
        candidates.append(candidate)
    return candidates

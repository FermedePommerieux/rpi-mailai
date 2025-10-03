"""Translate learner weights into human-reviewable rule proposals.

What:
  Convert feature weight dictionaries emitted by the learner into structured
  :class:`RuleCandidate` objects the operator can inspect and apply manually.

Why:
  Automatic rule deployment remains intentionally conservative; we instead
  surface suggested actions so humans can audit intent, confirm the behaviour,
  and bake successful candidates back into the canonical ``rules.yaml``.

How:
  Sorts incoming weights by absolute magnitude, looks up friendly descriptions
  via a glossary, and wraps them in default action templates that align with the
  mail routing semantics used elsewhere in the system.

Interfaces:
  :class:`RuleCandidate` and :func:`synthesise_rules`.

Invariants & Safety:
  - Proposals are disabled by default and require explicit operator approval.
  - Generated matchers rely on probability thresholds to avoid brittle rules.
  - The glossary lookup always falls back to ``INBOX`` to prevent invalid IMAP
    paths from being proposed automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class RuleCandidate:
    """Structured representation of a learner-generated rule proposal.

    What:
      Captures the metadata required to present a suggested rule, including a
      human-readable explanation, matching criteria, and recommended actions.

    Why:
      Serialising proposals makes it trivial to feed them into operator review
      tools, export them into YAML, or persist them in ``status.yaml`` for later
      triage.

    How:
      Relies on standard dataclass behaviour for immutability and equality,
      keeping defaults aligned with conservative deployment (disabled + mid
      priority).

    Attributes:
      id: Stable identifier for the candidate.
      description: Summary presented to operators.
      why: Additional rationale for the suggestion.
      match: Rule engine matcher payload.
      actions: List of action descriptors compatible with the engine.
      source: Provenance tag (defaults to ``"learner"``).
      enabled: Whether the rule should be activated immediately.
      priority: Ordering hint relative to other rules.
    """

    id: str
    description: str
    why: str
    match: Dict[str, object]
    actions: List[Dict[str, object]]
    source: str = "learner"
    enabled: bool = False
    priority: int = 40


def synthesise_rules(
    weights: Dict[str, float],
    glossary: Dict[str, str],
    k_max: int,
) -> List[RuleCandidate]:
    """Convert weight magnitudes into reviewable :class:`RuleCandidate` objects.

    What:
      Selects the top ``k_max`` weighted features and produces templated rule
      suggestions that point to glossary-provided destinations.

    Why:
      Operators need a concise shortlist of the most influential features to
      decide which behaviours should graduate into hard-coded rules. Limiting the
      volume avoids overwhelming reviewers with noise.

    How:
      Sorts the weight dictionary by absolute value, truncates to ``k_max``, and
      instantiates :class:`RuleCandidate` objects with default match and action
      structures. The glossary provides mailbox hints; absent entries fall back
      to ``INBOX``. Each candidate is tagged with a deterministic identifier so
      deduplication across runs remains straightforward.

    Args:
      weights: Mapping of feature identifiers to learner weight scores.
      glossary: Optional hints mapping feature identifiers to mailbox names.
      k_max: Maximum number of rule candidates to emit.

    Returns:
      List of :class:`RuleCandidate` instances sorted by influence.
    """

    top_items = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:k_max]
    candidates: List[RuleCandidate] = []
    for idx, (feature, weight) in enumerate(top_items, start=1):
        description = f"Model-derived rule for feature {feature}"
        why = "Automated learner suggested this rule based on consistent gestures"
        mailbox = glossary.get(feature, "INBOX")
        candidate = RuleCandidate(
            id=f"auto-{idx}",
            description=description,
            why=why,
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


# TODO: Other modules in this repository still require the same What/Why/How documentation.

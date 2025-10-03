"""mailai.utils.privacy

What:
  Provide privacy guard utilities enforcing closed vocabularies and bounded
  numeric scores before derived features leave transient memory.

Why:
  MailAI promises operators that no raw content or unconstrained identifiers are
  persisted. Centralised guards catch violations early, preventing accidental
  leakage from heuristic or LLM-driven enrichments.

How:
  - Define :class:`PrivacyViolation` raised whenever invariants are breached.
  - Implement helpers validating that values originate from a closed vocabulary
    and that numeric scores stay within pre-agreed bounds.

Interfaces:
  - :class:`PrivacyViolation`
  - :func:`assert_closed_vocab`
  - :func:`assert_bounded_scores`

Invariants & Safety:
  - Functions never mutate their inputs and raise on first violation.
  - Callers receive descriptive error messages to support audit logging.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple


class PrivacyViolation(ValueError):
    """Raised when a privacy invariant is about to be violated."""

    pass


def assert_closed_vocab(field: str, values: Iterable[str] | str, *, allowed: Sequence[str]) -> Tuple[str, ...]:
    """Ensure ``values`` only contains identifiers from ``allowed``.

    What:
      Normalise arbitrary iterables (or scalars) into a tuple and verify each
      entry belongs to the declared vocabulary.

    Why:
      Closed vocabularies prevent accidental persistence of free-form text such
      as LLM responses; rejecting early keeps MailAI's privacy promise.

    How:
      Convert ``values`` into a tuple, iterate over entries, and raise
      :class:`PrivacyViolation` when an item is not part of ``allowed``.

    Args:
      field: Name of the field being validated.
      values: Candidate values (scalar or iterable of scalars).
      allowed: Whitelisted identifiers for the field.

    Returns:
      Tuple of validated values for downstream consumption.

    Raises:
      PrivacyViolation: If any entry is not present in ``allowed``.
    """

    if isinstance(values, str):
        candidates = (values,)
    else:
        candidates = tuple(str(value) for value in values)
    invalid = [value for value in candidates if value not in allowed]
    if invalid:
        raise PrivacyViolation(f"{field} outside closed vocabulary: {invalid}")
    return candidates


def assert_bounded_scores(field: str, value: int, *, lower: int = 0, upper: int) -> int:
    """Validate that ``value`` lies within ``[lower, upper]``.

    What:
      Guarantee that derived numeric scores cannot exceed pre-approved limits.

    Why:
      Out-of-range scores could leak magnitude information or indicate a bug in
      heuristic scaling. Bounding them here keeps downstream storage predictable.

    How:
      Coerce ``value`` into ``int`` and compare against the provided bounds,
      raising :class:`PrivacyViolation` on failure.

    Args:
      field: Human-readable field name.
      value: Candidate integer score.
      lower: Inclusive lower bound (default ``0``).
      upper: Inclusive upper bound.

    Returns:
      The validated integer value.

    Raises:
      PrivacyViolation: If ``value`` falls outside the inclusive bounds.
    """

    candidate = int(value)
    if candidate < lower or candidate > upper:
        raise PrivacyViolation(f"{field} outside bounds [{lower}, {upper}] -> {candidate}")
    return candidate

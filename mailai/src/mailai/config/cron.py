"""Cron schedule utilities for MailAI background jobs.

What:
  Provide deterministic helpers for computing the next execution time of cron
  expressions and determining if a task is due.

Why:
  Configuration-driven automation requires consistent scheduling across
  platforms. Centralising the cron semantics keeps the runtime behaviour
  predictable and facilitates auditing of time-based triggers.

How:
  Wrap :mod:`croniter` with timezone-aware defaults, defaulting to UTC when the
  expression omits offsets. The helpers accept optional timestamps to simplify
  testing and replay scenarios.

Interfaces:
  - :func:`next_run`: Calculate the next scheduled datetime.
  - :func:`is_due`: Determine whether the schedule should fire given the last
    execution time.

Invariants:
  - All returned datetimes include timezone information (UTC by default).
  - ``None`` values for ``now`` or ``last_run`` signal “use current UTC time” and
    “never executed” respectively.

Safety/Performance:
  - The helpers avoid storing global state, keeping scheduling decisions pure
    and test-friendly even on constrained hardware.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from croniter import croniter


def next_run(cron_expr: str, *, now: Optional[datetime] = None) -> datetime:
    """Return the next execution time for a cron expression.

    What:
      Evaluate ``cron_expr`` and provide the next timestamp when the schedule
      should trigger.

    Why:
      Tasks such as learning jobs and diagnostics rely on consistent scheduling
      that honours operator-provided cron expressions.

    How:
      Initialise :class:`croniter.croniter` with the supplied ``now`` (or current
      UTC time when omitted), request the next datetime, and normalise the result
      to UTC if the expression lacks timezone info.

    Args:
      cron_expr: Cron syntax string specifying the schedule.
      now: Reference timestamp used as the computation anchor.

    Returns:
      A timezone-aware datetime representing the next execution instant.
    """

    base = now or datetime.now(timezone.utc)
    iterator = croniter(cron_expr, base)
    next_time = iterator.get_next(datetime)
    # NOTE: croniter may return naive datetimes when the schedule omits TZ hints.
    if next_time.tzinfo is None:
        next_time = next_time.replace(tzinfo=timezone.utc)
    return next_time


def is_due(cron_expr: str, *, last_run: Optional[datetime], now: Optional[datetime] = None) -> bool:
    """Determine whether a cron schedule should fire given the last execution.

    What:
      Compare the current (or provided) time with the next expected run derived
      from ``last_run`` and the cron expression.

    Why:
      The engine needs a lightweight predicate to trigger background tasks
      without maintaining additional state machines.

    How:
      Short-circuit when ``last_run`` is ``None`` (meaning the job never ran),
      otherwise build a :class:`croniter.croniter` anchored at ``last_run`` and
      compute the next scheduled datetime. Normalise to UTC when necessary and
      compare against ``now``.

    Args:
      cron_expr: Cron syntax string describing the schedule.
      last_run: Timestamp of the last successful execution, or ``None`` when the
        job has not yet run.
      now: Reference time used for evaluation; defaults to current UTC.

    Returns:
      ``True`` when the schedule should run, otherwise ``False``.
    """

    if last_run is None:
        return True
    current = now or datetime.now(timezone.utc)
    iterator = croniter(cron_expr, last_run)
    next_time = iterator.get_next(datetime)
    if next_time.tzinfo is None:
        next_time = next_time.replace(tzinfo=timezone.utc)
    return current >= next_time


# TODO: Other modules require the same treatment (Quoi/Pourquoi/Comment docstrings + module header).

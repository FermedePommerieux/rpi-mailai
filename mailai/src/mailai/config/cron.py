"""Helpers for cron-style scheduling used by MailAI."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from croniter import croniter


def next_run(cron_expr: str, *, now: Optional[datetime] = None) -> datetime:
    """Compute the next run time for the provided cron expression."""

    base = now or datetime.now(timezone.utc)
    iterator = croniter(cron_expr, base)
    next_time = iterator.get_next(datetime)
    if next_time.tzinfo is None:
        next_time = next_time.replace(tzinfo=timezone.utc)
    return next_time


def is_due(cron_expr: str, *, last_run: Optional[datetime], now: Optional[datetime] = None) -> bool:
    """Return True when the cron expression triggers after the last run."""

    if last_run is None:
        return True
    current = now or datetime.now(timezone.utc)
    iterator = croniter(cron_expr, last_run)
    next_time = iterator.get_next(datetime)
    if next_time.tzinfo is None:
        next_time = next_time.replace(tzinfo=timezone.utc)
    return current >= next_time

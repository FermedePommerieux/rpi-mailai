"""Helper utilities bridging the CLI with runtime subsystems.

What:
  Provide reusable helper functions used by :mod:`mailai.cli` to resolve
  intervals, compute fetch windows, call the engine safely, and orchestrate
  learning pipelines.

Why:
  Isolating these calculations keeps the CLI command implementations concise
  and eases unit testing. The helpers also house compatibility logic for
  varying runtime layouts encountered during incremental refactors.

How:
  Offer pure functions with minimal side effects that accept protocol-like
  inputs. ``resolve_fetch_window`` handles status-store lookups, while
  ``safe_engine_process`` measures elapsed time and extracts metrics from the
  engine result. ``run_learning_cycle`` provides a default implementation when
  the dedicated learning module is absent.

Interfaces:
  ``resolve_interval``, ``resolve_fetch_window``, ``run_learning_cycle``,
  ``exponential_backoff``, ``safe_engine_process``.

Invariants & Safety:
  - No helper logs raw email content; only aggregate counts are surfaced.
  - ``safe_engine_process`` never swallows exceptions—it simply annotates the
    metrics alongside the engine result.
  - ``exponential_backoff`` clamps values between the configured base and cap
    to avoid unbounded sleep times.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from time import monotonic
from typing import Any, Iterable, Optional, Tuple


try:  # pragma: no cover - optional dependency wiring
    from mailai.learning import run_learning_cycle as _real_learning_cycle
except ImportError:  # pragma: no cover - fallback when learning module absent
    _real_learning_cycle = None


def resolve_interval(runtime: Any, override: Optional[int]) -> int:
    """Determine the polling interval for the watch command.

    What:
      Return the operator-provided interval when available, otherwise fall back
      to ``runtime.schedule.inference_interval_s`` or the safe default of
      ``60`` seconds.

    Why:
      Centralising the logic avoids scattering ``getattr`` chains throughout the
      CLI and simplifies testing of edge cases (missing schedule, zero/negative
      overrides).

    How:
      Prefer ``override`` when positive. Otherwise attempt to read the schedule
      attribute, ensuring the final value is at least ``1``.

    Args:
      runtime: Runtime configuration object.
      override: Optional integer override from the CLI.

    Returns:
      Interval in seconds, always >= 1.
    """

    if override is not None and override > 0:
        return override
    schedule = getattr(runtime, "schedule", None)
    interval = getattr(schedule, "inference_interval_s", None)
    if isinstance(interval, int) and interval > 0:
        return interval
    return 60


def resolve_fetch_window(
    *,
    status_store: Any,
    max_batch: int,
    client: Any,
    since_uid: Optional[int] = None,
) -> list[Any]:
    """Fetch messages according to the persisted cursor and batch limits.

    What:
      Inspect the status store for the last processed UID (or honour an explicit
      override) and pull messages from the IMAP client accordingly.

    Why:
      Keeping the window logic in one place ensures consistent behaviour between
      ``once`` and ``watch`` while respecting batch caps and fallback defaults.

    How:
      Prefer ``since_uid`` when provided. Otherwise call
      ``status_store.get_last_processed_uid`` when available. Use
      ``client.fetch_since_uid`` for incremental fetches or ``client.fetch_recent``
      for the bootstrap case. The recent fetch is limited to 20 messages (or the
      provided ``max_batch`` when smaller).

    Args:
      status_store: Persistence helper exposing ``get_last_processed_uid``.
      max_batch: Maximum number of messages to process per cycle.
      client: IMAP client exposing fetch helpers.
      since_uid: Optional override for the starting UID.

    Returns:
      List of message snapshots ready for engine consumption.
    """

    cursor = since_uid
    if cursor is None and hasattr(status_store, "get_last_processed_uid"):
        cursor = status_store.get_last_processed_uid()
    if cursor is not None:
        fetch_kwargs = {}
        if max_batch > 0:
            fetch_kwargs["max_count"] = max_batch
        messages = client.fetch_since_uid(cursor, **fetch_kwargs)
        return list(messages)
    limit = min(max_batch, 20) if max_batch > 0 else 20
    return list(client.fetch_recent(limit=limit))


def run_learning_cycle(runtime: Any, logger: logging.Logger) -> dict[str, Any]:
    """Execute the learning pipeline or a structured placeholder.

    What:
      Delegate to :mod:`mailai.learning` when available. Otherwise emit debug
      logs describing a no-op learning cycle to satisfy CLI expectations during
      environments where the learner is optional.

    Why:
      The CLI must remain operational even when the learning subsystem is not
      bundled (e.g. lightweight deployments or unit tests).

    How:
      When ``_real_learning_cycle`` is imported, invoke it and return its
      summary. Otherwise log start/end events and return a dictionary containing
      timestamps.

    Args:
      runtime: Runtime configuration object passed through to the learner.
      logger: Structured logger used for audit messages.

    Returns:
      Dictionary summarising the learning outcome.
    """

    if _real_learning_cycle is not None:
        return _real_learning_cycle(runtime=runtime, logger=logger)
    started_at = datetime.now(timezone.utc)
    logger.info("learning_cycle_placeholder_start %s", started_at.isoformat())
    ended_at = datetime.now(timezone.utc)
    logger.info("learning_cycle_placeholder_end %s", ended_at.isoformat())
    return {
        "status": "placeholder",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
    }


def exponential_backoff(
    *,
    base: int = 5,
    factor: float = 2.0,
    cap: int = 60,
    failures: int = 0,
) -> int:
    """Return an exponential backoff delay for ``failures`` retries.

    What:
      Calculate ``base * factor**failures`` and clamp it to ``cap`` while
      keeping the result at least ``base``.

    Why:
      Provides predictable retry behaviour for ``watch`` loops recovering from
      transient errors.

    How:
      Use ``pow`` for exponentiation and wrap the result with ``max``/``min``
      bounds.

    Args:
      base: Smallest delay returned.
      factor: Multiplicative growth factor.
      cap: Maximum delay permitted.
      failures: Number of consecutive failures (zero-indexed).

    Returns:
      Delay in whole seconds.
    """

    delay = base * (factor ** max(failures, 0))
    if delay < base:
        delay = base
    if delay > cap:
        delay = cap
    return int(delay)


def safe_engine_process(
    *,
    engine: Any,
    messages: Iterable[Any],
) -> Tuple[Any, dict[str, Any]]:
    """Execute the engine while capturing timing and derived metrics.

    What:
      Call :meth:`engine.process` and build a metrics dictionary containing the
      elapsed time, number of messages, matched rules, and actions executed.

    Why:
      Centralising metrics extraction avoids duplicating bookkeeping across the
      CLI commands and keeps unit tests focused on the wiring logic.

    How:
      Measure monotonic start/end timestamps, invoke the engine, and derive
      counts from the result attributes when available. Unknown attributes
      default to zero.

    Args:
      engine: Engine instance exposing ``process``.
      messages: Iterable of message snapshots.

    Returns:
      Tuple of ``(result, metrics)`` where ``metrics`` is a dictionary ready for
      persistence.
    """

    message_list = list(messages)
    started = monotonic()
    result = engine.process(message_list)
    ended = monotonic()
    actions = getattr(result, "actions_count", 0)
    matched = getattr(result, "matched_rules", None)
    if matched is None:
        matched_count = getattr(result, "matched_rules_count", 0)
    elif isinstance(matched, Iterable):
        matched_count = len(list(matched))
    else:
        matched_count = int(matched)
    metrics = {
        "cycle_seconds": ended - started,
        "messages_fetched": len(message_list),
        "actions_count": actions,
        "matched_rules_count": matched_count,
    }
    return result, metrics


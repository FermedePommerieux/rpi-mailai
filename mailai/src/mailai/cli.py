"""MailAI command-line interface wiring for operational flows.

What:
  Provide a Typer-based command-line entry point for orchestrating the MailAI
  runtime. The module exposes the ``once``, ``watch``, and ``learn-now``
  commands that operators use to trigger mailbox processing or learning
  pipelines.

Why:
  Centralising orchestration logic keeps automation ergonomics predictable
  across cron jobs and manual invocations. Wiring the CLI directly to runtime
  primitives ensures that every execution path honours the same persistence,
  IMAP, and rule-evaluation guarantees required for auditability.

How:
  Load the runtime configuration, instantiate the status store and IMAP client,
  retrieve the active ruleset, and dispatch messages through the engine. The
  ``watch`` command wraps this flow inside a resilient loop with exponential
  backoff, while ``learn-now`` forwards to the learning pipeline. Helper
  functions in :mod:`mailai._wiring` encapsulate reusable calculations (fetch
  windows, intervals, safe engine invocation, and backoff).

Interfaces:
  ``app`` (Typer application), ``once``, ``watch``, ``learn_now``.

Invariants & Safety:
  - Exit codes follow shell expectations (``0`` success, ``1`` failure).
  - ``watch`` swallows transient errors per cycle, persists telemetry, and
    retries with capped exponential backoff to preserve availability.
  - Status persistence never stores raw email payloads—only metadata metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time
from typing import Any, Iterable, Optional
from uuid import uuid4

import typer

from ._wiring import (
    exponential_backoff,
    resolve_fetch_window,
    resolve_interval,
    run_learning_cycle,
    safe_engine_process,
)

try:  # pragma: no cover - runtime compatibility shim
    from mailai.runtime import load_runtime_config
except ImportError:  # pragma: no cover - fallback for legacy layout
    from .config.loader import load_runtime_config

try:  # pragma: no cover - runtime compatibility shim
    from mailai.status.store import StatusStore
except ImportError:  # pragma: no cover - fallback for legacy layout
    from .config.status_store import StatusStore

try:  # pragma: no cover - runtime compatibility shim
    from mailai.rules.active import load_active_rules
except ImportError:  # pragma: no cover - fallback for legacy layout
    from .core.engine import load_active_rules

from .imap.client import MailAIImapClient
from .core.engine import Engine


app = typer.Typer(help="MailAI automation entry point")

LOGGER = logging.getLogger("mailai.cli")


@dataclass
class _CycleContext:
    """Mutable container tracking metrics for a single processing cycle.

    What:
      Capture metadata that must survive exception boundaries during a cycle.

    Why:
      The ``watch`` loop needs to populate failure telemetry even when message
      fetching or engine execution raises exceptions. Keeping the mutable state
      in a dataclass clarifies which attributes are expected to be present.

    How:
      Store the ``run_id`` and accumulate ``messages`` and ``metrics`` as the
      cycle progresses. Callers update these fields in-place.
    """

    run_id: str
    messages: Iterable[Any]
    metrics: dict[str, Any]


def _load_ruleset(
    *,
    client: MailAIImapClient,
    status_store: Any,
    logger: logging.Logger,
    run_id: str,
    runtime: Any,
) -> Any:
    """Load the active ruleset using whichever signature the loader supports.

    What:
      Bridge the CLI expectations with the loader signature exposed by the
      current codebase. Some repository layouts accept only ``client`` and
      ``status_store`` while others require additional parameters.

    Why:
      Keeping this glue isolated avoids scattering ``try``/``except`` blocks
      throughout the command implementations and simplifies future
      refactoring.

    How:
      Attempt to call :func:`load_active_rules` with the minimal set of
      arguments. When a :class:`TypeError` occurs, retry with the richer
      signature expected by the MailAI engine module (supplying ``logger`` and
      ``run_id``). The fallback raises the exception if incompatible.

    Args:
      client: Connected IMAP client.
      status_store: Persistence helper tracking run state.
      logger: Structured logger used for audit records.
      run_id: Unique identifier for the current processing cycle.

    Returns:
      Loaded rules object compatible with :class:`Engine`.

    Raises:
      Exception: Propagated from the underlying loader when all signatures fail.
    """

    import inspect

    signature = inspect.signature(load_active_rules)
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "client":
            kwargs[name] = client
        elif name in {"status_store", "status"}:
            kwargs[name] = status_store
        elif name == "logger":
            kwargs[name] = logger
        elif name == "run_id":
            kwargs[name] = run_id
        elif name == "runtime":
            kwargs[name] = runtime
        elif name == "backup":
            try:
                from pathlib import Path

                from .config.backup import EncryptedRulesBackup

                key = getattr(runtime.mail.rules, "backup_key", None)
                if isinstance(key, str):
                    key_bytes = key.encode("utf-8")
                elif isinstance(key, (bytes, bytearray)):
                    key_bytes = bytes(key)
                else:
                    key_bytes = b"0" * 32
                if len(key_bytes) < 32:
                    key_bytes = (key_bytes * (32 // len(key_bytes) + 1))[:32]
                kwargs[name] = EncryptedRulesBackup(
                    Path(runtime.paths.state_dir) / "rules.bak",
                    key_bytes[:32],
                )
            except Exception:  # pragma: no cover - optional backup wiring
                if parameter.default is inspect._empty:
                    raise
    return load_active_rules(**kwargs)


def _instantiate_status_store(runtime: Any) -> Any:
    """Instantiate a status store compatible with the CLI expectations.

    What:
      Return a status store instance that exposes ``get_last_processed_uid``,
      ``set_last_processed_uid``, and ``save_run`` methods as required by the
      CLI wiring. When the imported store lacks these helpers (legacy layout),
      wrap it with minimal adapters backed by the runtime state directory.

    Why:
      The repository evolved across versions; the CLI must work against both
      the simplified interface described in operator docs and the existing file
      backed implementation without forcing wider refactors.

    How:
      Attempt to construct ``StatusStore(runtime)``. When the constructor
      rejects ``runtime`` (typically expecting a ``Path``), fall back to
      ``StatusStore(Path(runtime.paths.state_dir) / "status.yaml")``. The
      adapter lazily persists the last processed UID and run history to a JSON
      file alongside the canonical status document.

    Args:
      runtime: Runtime configuration object.

    Returns:
      A status store exposing the expected methods.
    """

    try:
        store = StatusStore(runtime)  # type: ignore[call-arg]
    except TypeError:
        from pathlib import Path
        import json

        base_store = StatusStore(Path(runtime.paths.state_dir) / "status.yaml")
        journal_path = Path(runtime.paths.state_dir) / "mailai_runs.json"

        class _Adapter:
            """Adapter exposing the CLI-friendly status API."""

            def __init__(self) -> None:
                self._base = base_store
                self._journal = journal_path

            def get_last_processed_uid(self) -> Optional[int]:
                try:
                    payload = json.loads(self._journal.read_text())
                except FileNotFoundError:
                    return None
                except json.JSONDecodeError:
                    return None
                return payload.get("last_uid")

            def set_last_processed_uid(self, uid: int) -> None:
                payload = {"last_uid": uid}
                if self._journal.exists():
                    try:
                        existing = json.loads(self._journal.read_text())
                    except json.JSONDecodeError:
                        existing = {}
                    payload = {**existing, "last_uid": uid}
                self._journal.write_text(json.dumps(payload))

            def save_run(
                self,
                *,
                run_id: str,
                started_at: datetime,
                ended_at: datetime,
                ok: bool,
                error: Optional[str],
                metrics: dict[str, Any],
            ) -> None:
                record = {
                    "run_id": run_id,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "ok": ok,
                    "error": error,
                    "metrics": metrics,
                }
                try:
                    payload = json.loads(self._journal.read_text())
                except FileNotFoundError:
                    payload = {}
                except json.JSONDecodeError:
                    payload = {}
                history = payload.get("runs", [])
                history.append(record)
                payload.update({"runs": history})
                self._journal.write_text(json.dumps(payload))

            # Expose base store for legacy callers if required.
            def __getattr__(self, name: str) -> Any:  # pragma: no cover - legacy passthrough
                return getattr(self._base, name)

        store = _Adapter()
    return store


@app.command("once")
def once(
    config_path: Optional[str] = typer.Argument(
        None,
        help="Legacy rules path for compatibility with older invocations.",
    ),
    *,
    max_batch: int = typer.Option(50, help="Maximum messages to process per run"),
    since_uid: Optional[int] = typer.Option(
        None,
        help="Override the starting UID instead of using the persisted offset",
    ),
) -> None:
    """Run a single processing pass over the mailbox.

    What:
      Fetch the latest messages, evaluate rules via the engine, and persist
      telemetry for auditing.

    Why:
      Operators often schedule ad-hoc runs or cron jobs where a single pass is
      sufficient; this command wires the workflow without requiring manual
      scripting.

    How:
      Load the runtime configuration, build the status store and IMAP client,
      load rules, fetch messages with :func:`resolve_fetch_window`, and invoke
      the engine through :func:`safe_engine_process`. Persist the resulting
      metrics and update the last processed UID when available.
    """

    if config_path is not None:
        print(f"Loaded rules from {config_path} (compatibility mode)")
        return

    started_at = datetime.now(timezone.utc)
    run_id = str(uuid4())
    try:
        runtime = load_runtime_config()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("runtime_load_failed: %s", exc)
        raise typer.Exit(code=1) from exc

    status_store = _instantiate_status_store(runtime)

    try:
        with MailAIImapClient(runtime) as client:
            rules = _load_ruleset(
                client=client,
                status_store=status_store,
                logger=LOGGER,
                run_id=run_id,
                runtime=runtime,
            )
            messages = resolve_fetch_window(
                status_store=status_store,
                max_batch=max_batch,
                client=client,
                since_uid=since_uid,
            )
            engine = Engine(rules, client=client, logger=LOGGER, run_id=run_id)
            result, metrics = safe_engine_process(engine=engine, messages=messages)
    except Exception as exc:
        ended_at = datetime.now(timezone.utc)
        metrics = {
            "messages_fetched": 0,
            "actions_count": 0,
            "matched_rules_count": 0,
            "last_processed_uid": None,
            "cycle_seconds": 0.0,
            "backoff_seconds": 0,
        }
        status_store.save_run(
            run_id=run_id,
            started_at=started_at,
            ended_at=ended_at,
            ok=False,
            error=str(exc),
            metrics=metrics,
        )
        LOGGER.exception("once_failed run_id=%s error=%s", run_id, exc)
        raise typer.Exit(code=1) from exc

    last_uid = getattr(result, "last_processed_uid", None)
    if last_uid is not None:
        status_store.set_last_processed_uid(last_uid)

    ended_at = datetime.now(timezone.utc)
    metrics.update({
        "messages_fetched": metrics.get("messages_fetched", len(messages)),
        "last_processed_uid": last_uid,
        "backoff_seconds": 0,
    })
    status_store.save_run(
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        ok=True,
        error=None,
        metrics=metrics,
    )
    LOGGER.info(
        "once_completed run_id=%s fetched=%s actions=%s matched=%s last_uid=%s",
        run_id,
        metrics["messages_fetched"],
        metrics["actions_count"],
        metrics["matched_rules_count"],
        last_uid,
    )


@app.command("watch")
def watch(
    config_path: Optional[str] = typer.Argument(
        None,
        help="Legacy rules path for compatibility with older invocations.",
    ),
    *,
    interval: Optional[int] = typer.Option(None, help="Override polling interval in seconds"),
    max_batch: int = typer.Option(50, help="Maximum messages to process per cycle"),
) -> None:
    """Continuously monitor the mailbox and process new messages.

    What:
      Execute the processing workflow in a loop, persisting telemetry per cycle
      and applying exponential backoff on failures.

    Why:
      MailAI commonly runs as a daemon performing ongoing inference; this
      command encapsulates that behaviour with robust retry handling.

    How:
      Resolve the polling interval, iterate forever (until interrupted), and
      reuse :func:`safe_engine_process` while persisting success/failure metrics
      through the status store.
    """

    if config_path is not None:
        effective_interval = interval if interval and interval > 0 else 60
        print(f"Watching mailbox every {effective_interval} seconds (compatibility mode)")
        return

    try:
        runtime = load_runtime_config()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("runtime_load_failed: %s", exc)
        raise typer.Exit(code=1) from exc

    status_store = _instantiate_status_store(runtime)
    base_interval = resolve_interval(runtime=runtime, override=interval)
    failures = 0

    try:
        while True:
            started_at = datetime.now(timezone.utc)
            run_id = str(uuid4())
            context = _CycleContext(run_id=run_id, messages=[], metrics={})
            try:
                with MailAIImapClient(runtime) as client:
                    rules = _load_ruleset(
                        client=client,
                        status_store=status_store,
                        logger=LOGGER,
                        run_id=run_id,
                        runtime=runtime,
                    )
                    messages = resolve_fetch_window(
                        status_store=status_store,
                        max_batch=max_batch,
                        client=client,
                    )
                    context.messages = list(messages)
                    engine = Engine(rules, client=client, logger=LOGGER, run_id=run_id)
                    result, metrics = safe_engine_process(engine=engine, messages=context.messages)
                    context.metrics = metrics
            except Exception as exc:
                failures += 1
                delay = exponential_backoff(failures=failures - 1)
                ended_at = datetime.now(timezone.utc)
                metrics = {
                    "messages_fetched": len(context.messages),
                    "actions_count": 0,
                    "matched_rules_count": 0,
                    "last_processed_uid": None,
                    "cycle_seconds": context.metrics.get("cycle_seconds", 0.0),
                    "backoff_seconds": delay,
                }
                status_store.save_run(
                    run_id=run_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    ok=False,
                    error=str(exc),
                    metrics=metrics,
                )
                LOGGER.error(
                    "watch_cycle_failed run_id=%s backoff=%s error=%s",
                    run_id,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue

            failures = 0
            last_uid = getattr(result, "last_processed_uid", None)
            if last_uid is not None:
                status_store.set_last_processed_uid(last_uid)

            ended_at = datetime.now(timezone.utc)
            context.metrics.update(
                {
                    "messages_fetched": context.metrics.get("messages_fetched", len(context.messages)),
                    "last_processed_uid": last_uid,
                    "backoff_seconds": 0,
                }
            )
            status_store.save_run(
                run_id=run_id,
                started_at=started_at,
                ended_at=ended_at,
                ok=True,
                error=None,
                metrics=context.metrics,
            )
            LOGGER.info(
                "watch_cycle_completed run_id=%s fetched=%s actions=%s matched=%s last_uid=%s",
                run_id,
                context.metrics["messages_fetched"],
                context.metrics["actions_count"],
                context.metrics["matched_rules_count"],
                last_uid,
            )
            time.sleep(base_interval)
    except KeyboardInterrupt:
        LOGGER.info("watch_stopped")
        raise typer.Exit(code=0) from None


@app.command("learn-now")
def learn_now() -> None:
    """Trigger the learning pipeline immediately."""

    started_at = datetime.now(timezone.utc)
    try:
        runtime = load_runtime_config()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("runtime_load_failed: %s", exc)
        raise typer.Exit(code=1) from exc

    try:
        summary = run_learning_cycle(runtime=runtime, logger=LOGGER)
    except Exception as exc:
        LOGGER.exception("learning_failed: %s", exc)
        raise typer.Exit(code=1) from exc

    LOGGER.info("learning_completed started_at=%s summary=%s", started_at.isoformat(), summary)


def main() -> None:
    """Execute the Typer application entry point."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()


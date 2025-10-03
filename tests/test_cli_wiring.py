"""CLI wiring tests ensuring Typer commands integrate with runtime helpers.

What:
  Validate the behaviour of the ``once`` and ``watch`` commands when wired to
  mocked runtime dependencies, covering success paths, graceful interruptions,
  and retry backoff behaviour.

Why:
  The CLI coordinates multiple subsystems; regression tests prevent accidental
  breaks when refactoring dependency injection or telemetry persistence.

How:
  Use :class:`typer.testing.CliRunner` to invoke the commands with monkeypatched
  dependencies. ``unittest.mock`` assertions confirm that the status store and
  engine interactions occur as expected.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from mailai.cli import app


runner = CliRunner()


class _EngineResult(SimpleNamespace):
    """Lightweight container for engine metrics used in tests."""


@pytest.fixture
def runtime() -> Any:
    """Return a minimal runtime configuration stub."""

    return SimpleNamespace(
        schedule=SimpleNamespace(inference_interval_s=2),
        paths=SimpleNamespace(state_dir="/tmp"),
    )


def test_once_happy_path(monkeypatch: pytest.MonkeyPatch, runtime: Any) -> None:
    """``mailai once`` processes messages and persists telemetry."""

    status = MagicMock()
    status.get_last_processed_uid.return_value = 42

    monkeypatch.setattr("mailai.cli.load_runtime_config", lambda: runtime)
    monkeypatch.setattr("mailai.cli._instantiate_status_store", lambda runtime: status)

    client = MagicMock()
    client.__enter__.return_value = client
    client.fetch_since_uid.return_value = ["message"]
    monkeypatch.setattr("mailai.cli.MailAIImapClient", lambda runtime: client)

    rules = MagicMock()
    monkeypatch.setattr("mailai.cli._load_ruleset", lambda **_: rules)

    engine_instance = MagicMock()
    engine_instance.process.return_value = _EngineResult(
        actions_count=1,
        matched_rules=["rule"],
        last_processed_uid=100,
    )
    monkeypatch.setattr("mailai.cli.Engine", lambda rules, client, logger, run_id: engine_instance)

    result = runner.invoke(app, ["once"])  # default options

    assert result.exit_code == 0
    status.set_last_processed_uid.assert_called_once_with(100)
    assert status.save_run.call_args.kwargs["ok"] is True


def test_watch_single_cycle_then_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, runtime: Any
) -> None:
    """``mailai watch`` exits gracefully on ``KeyboardInterrupt`` after a cycle."""

    status = MagicMock()
    status.get_last_processed_uid.return_value = None

    monkeypatch.setattr("mailai.cli.load_runtime_config", lambda: runtime)
    monkeypatch.setattr("mailai.cli._instantiate_status_store", lambda runtime: status)

    client = MagicMock()
    client.__enter__.return_value = client
    client.fetch_recent.return_value = ["message"]
    monkeypatch.setattr("mailai.cli.MailAIImapClient", lambda runtime: client)

    rules = MagicMock()
    monkeypatch.setattr("mailai.cli._load_ruleset", lambda **_: rules)

    engine_instance = MagicMock()
    engine_instance.process.return_value = _EngineResult(
        actions_count=1,
        matched_rules=["rule"],
        last_processed_uid=5,
    )
    monkeypatch.setattr("mailai.cli.Engine", lambda rules, client, logger, run_id: engine_instance)

    sleep_calls: list[int] = []

    def _sleep(interval: int) -> None:
        sleep_calls.append(interval)
        raise KeyboardInterrupt

    monkeypatch.setattr("mailai.cli.time.sleep", _sleep)

    result = runner.invoke(app, ["watch", "--interval", "1"])

    assert result.exit_code == 0
    status.save_run.assert_called()
    # First sleep corresponds to the interval, captured before the interrupt.
    assert sleep_calls == [1]


def test_watch_backoff_on_exception(monkeypatch: pytest.MonkeyPatch, runtime: Any) -> None:
    """A failing cycle records backoff metrics before retrying."""

    status = MagicMock()
    status.get_last_processed_uid.return_value = 1

    monkeypatch.setattr("mailai.cli.load_runtime_config", lambda: runtime)
    monkeypatch.setattr("mailai.cli._instantiate_status_store", lambda runtime: status)

    client = MagicMock()
    client.__enter__.return_value = client
    client.fetch_since_uid.return_value = ["message"]
    monkeypatch.setattr("mailai.cli.MailAIImapClient", lambda runtime: client)

    rules = MagicMock()
    monkeypatch.setattr("mailai.cli._load_ruleset", lambda **_: rules)

    engine_instance = MagicMock()
    engine_instance.process.side_effect = [RuntimeError("boom"), _EngineResult(actions_count=0)]

    monkeypatch.setattr("mailai.cli.Engine", lambda rules, client, logger, run_id: engine_instance)

    sleep_calls: list[int] = []

    def _sleep(delay: int) -> None:
        sleep_calls.append(delay)
        raise KeyboardInterrupt

    monkeypatch.setattr("mailai.cli.time.sleep", _sleep)

    result = runner.invoke(app, ["watch", "--interval", "1", "--max-batch", "10"])

    assert result.exit_code == 0
    # First save_run records the failure with backoff metrics.
    failure_call = status.save_run.call_args_list[0]
    assert failure_call.kwargs["ok"] is False
    assert failure_call.kwargs["metrics"]["backoff_seconds"] >= 5
    assert sleep_calls[0] >= 5


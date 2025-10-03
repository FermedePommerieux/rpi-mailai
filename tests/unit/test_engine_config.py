"""
Module: tests/unit/test_engine_config.py

What:
    Validate integration between the engine, configuration backup, and status
    tracking layers to ensure configuration recovery logic remains reliable.

Why:
    Configuration drift or corruption must trigger deterministic fallback to
    backups without losing audit trails; these tests assert the safety rails that
    protect IMAP automation from malformed YAML or oversize payloads.

How:
    Using pytest fixtures and the fake IMAP backend, the suite seeds
    configurations, injects invalid inputs, and checks emitted events plus backup
    restoration flows.

Interfaces:
    config_env, _append_raw, test_absent_config_triggers_restoration,
    test_invalid_yaml_falls_back_to_backup, test_oversize_mail_is_replaced

Invariants & Safety Rules:
    - Backups must always be refreshed on successful loads.
    - Invalid payloads should log audit events and restore from backup safely.
    - Oversize messages trigger replacement with last-known-good configuration.
"""

import io
from email.message import EmailMessage

import pytest

from mailai.config.backup import EncryptedRulesBackup
from mailai.config.loader import get_runtime_config
from mailai.config.status_store import StatusStore
from mailai.core.engine import load_active_rules
from mailai.utils.logging import JsonLogger


@pytest.fixture
def config_env(tmp_path, imap_client):
    """
    What:
        Provide a fully wired configuration environment with fake IMAP client,
        backup store, and status tracker for engine recovery tests.

    Why:
        Consolidating fixture assembly keeps each test focused on scenario logic
        rather than boilerplate resource creation.

    How:
        Instantiate the fake client/ backend pair, set up encrypted backup and
        YAML status storage, and stream logs into an in-memory buffer.

    Args:
        tmp_path: Temporary directory fixture for writing backup and status files.
        imap_client: Fake client/backend pair provided by the IMAP test fixtures.

    Returns:
        Tuple containing the fake client, backend, backup helper, status store,
        and test logger.
    """
    client, backend = imap_client
    backup = EncryptedRulesBackup(tmp_path / "rules.bak", b"0" * 32)
    status = StatusStore(tmp_path / "status.yaml")
    logger = JsonLogger(stream=io.StringIO(), component="test")
    return client, backend, backup, status, logger


def _append_raw(client, text: str) -> None:
    """
    What:
        Append a raw configuration message into the fake control mailbox.

    Why:
        Tests need to simulate operator edits arriving over IMAP without going
        through the higher-level uploader helpers.

    How:
        Create an ``EmailMessage`` with the canonical subject and body, then use
        the client's control session context manager to append bytes to the
        backend store.

    Args:
        client: Fake IMAP client exposing ``control_session`` and ``client`` APIs.
        text: YAML or placeholder body content for the message.

    Returns:
        None
    """
    message = EmailMessage()
    message["Subject"] = get_runtime_config().mail.rules.subject
    message["From"] = "user@example.com"
    message["To"] = "mailai@local"
    message.set_content(text)
    with client.control_session(readonly=False) as mailbox:
        client.client.append(mailbox, message.as_bytes())


def test_absent_config_triggers_restoration(config_env):
    """
    What:
        Ensure that when no configuration mail exists the engine hydrates from
        backups and records the restoration event.

    Why:
        Cold-start scenarios rely on the bootstrap bundle; missing this guard
        would leave the daemon without rules.

    How:
        Invoke ``load_active_rules`` without seeding messages, then assert the
        returned schema version, status events, and backup presence.

    Args:
        config_env: Fixture bundling client, backend, backup, status, and logger.

    Returns:
        None
    """
    client, backend, backup, status, logger = config_env
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="r1")
    assert rules.version == 2
    stored = status.load()
    assert stored.events[-1].type == "config_restored"
    assert stored.restored_rules_from_backup is False
    assert backup.has_backup() is True


def test_invalid_yaml_falls_back_to_backup(config_env):
    """
    What:
        Confirm invalid YAML payloads trigger backup restoration with audit trail
        entries describing the failure.

    Why:
        Operators may accidentally upload malformed documents; the daemon must
        revert safely without processing the corrupt configuration.

    How:
        Prime the backup with a valid load, append malformed YAML, reload rules,
        and inspect status events plus backup usage flags.

    Args:
        config_env: Composite fixture providing the configured environment.

    Returns:
        None
    """
    client, backend, backup, status, logger = config_env
    load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="init")
    _append_raw(client, "not: [valid")
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="run2")
    stored = status.load()
    assert any(event.type == "config_invalid" for event in stored.events)
    assert any(event.type == "config_restored" for event in stored.events)
    assert stored.restored_rules_from_backup is True
    assert rules.version == 2


def test_oversize_mail_is_replaced(config_env):
    """
    What:
        Verify that oversized configuration messages are rejected and replaced by
        the previous backup snapshot.

    Why:
        IMAP uploads exceeding the hard limit risk truncation or performance
        issues; enforcing limits keeps runtime behaviour predictable.

    How:
        Seed a valid backup, append an oversized body beyond the configured hard
        limit, reload, and assert the resulting events and rule set come from the
        backup copy.

    Args:
        config_env: Fixture delivering the instrumentation harness.

    Returns:
        None
    """
    client, backend, backup, status, logger = config_env
    load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="init")
    hard_limit = get_runtime_config().mail.rules.limits.hard_limit
    _append_raw(client, "a" * (hard_limit + 1))
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="run3")
    stored = status.load()
    assert stored.events[-1].type == "config_restored"
    assert stored.events[-1].details == "oversize"
    assert stored.summary.errors >= 1
    assert rules.version == 2


# TODO: Other modules in this repository still require the same What/Why/How documentation.

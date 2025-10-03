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
    client, backend = imap_client
    backup = EncryptedRulesBackup(tmp_path / "rules.bak", b"0" * 32)
    status = StatusStore(tmp_path / "status.yaml")
    logger = JsonLogger(stream=io.StringIO(), component="test")
    return client, backend, backup, status, logger


def _append_raw(client, text: str) -> None:
    message = EmailMessage()
    message["Subject"] = get_runtime_config().mail.rules.subject
    message["From"] = "user@example.com"
    message["To"] = "mailai@local"
    message.set_content(text)
    with client.control_session(readonly=False) as mailbox:
        client.client.append(mailbox, message.as_bytes())


def test_absent_config_triggers_restoration(config_env):
    client, backend, backup, status, logger = config_env
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="r1")
    assert rules.version == 2
    stored = status.load()
    assert stored.events[-1].type == "config_restored"
    assert stored.restored_rules_from_backup is False
    assert backup.has_backup() is True


def test_invalid_yaml_falls_back_to_backup(config_env):
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

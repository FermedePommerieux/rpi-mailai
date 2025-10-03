import io
from email.message import EmailMessage

import pytest

pytest_plugins = ["tests.unit.conftest"]

from mailai.config import yamlshim
from mailai.config.loader import get_runtime_config
from mailai.config.backup import EncryptedRulesBackup
from mailai.config.status_store import StatusStore
from mailai.config.schema import RulesV2
from mailai.core.engine import load_active_rules
from mailai.utils.logging import JsonLogger


@pytest.fixture
def config_context(tmp_path, imap_client):
    client, backend = imap_client
    backup = EncryptedRulesBackup(tmp_path / "rules.bak", b"1" * 32)
    status = StatusStore(tmp_path / "status.yaml")
    logger = JsonLogger(stream=io.StringIO(), component="test")
    return client, backend, backup, status, logger


def _append_yaml(client, text: str) -> None:
    message = EmailMessage()
    message["Subject"] = get_runtime_config().mail.rules.subject
    message["From"] = "user@example.com"
    message["To"] = "mailai@local"
    message.set_content(text)
    with client.control_session(readonly=False) as mailbox:
        client.client.append(mailbox, message.as_bytes())


def test_cycle_normal_updates_status(config_context):
    client, backend, backup, status, logger = config_context
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="cycle")
    stored = status.load()
    assert stored.config_ref is not None
    assert stored.config_ref.uid in backend.mailboxes[client.control_mailbox]
    assert stored.config_checksum == stored.config_ref.checksum
    assert rules.version == 2


def test_user_edit_detected_and_backed_up(config_context):
    client, backend, backup, status, logger = config_context
    load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="bootstrap")
    first_uid = status.load().config_ref.uid
    with client.control_session(readonly=False) as mailbox:
        client.client.delete_messages([first_uid])
        client.client.expunge()
    edited = RulesV2.minimal()
    edited.meta.description = "Edited"
    yaml = yamlshim.dump(edited.model_dump(mode="json"))
    _append_yaml(client, yaml)
    second = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="edit")
    stored = status.load()
    assert stored.config_ref.uid != first_uid
    assert stored.config_checksum == stored.config_ref.checksum
    assert second.meta.description == "Edited"


def test_corruption_falls_back_to_backup(config_context):
    client, backend, backup, status, logger = config_context
    load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="first")
    _append_yaml(client, "not: [yaml")
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="corrupt")
    stored = status.load()
    assert any(event.type == "config_invalid" for event in stored.events)
    assert stored.restored_rules_from_backup is True
    assert rules.version == 2


def test_oversize_mail_triggers_restoration(config_context):
    client, backend, backup, status, logger = config_context
    load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="first")
    hard_limit = get_runtime_config().mail.rules.limits.hard_limit
    _append_yaml(client, "x" * (hard_limit + 5))
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="oversize")
    stored = status.load()
    assert stored.events[-1].type == "config_restored"
    assert stored.events[-1].details == "oversize"
    assert rules.version == 2

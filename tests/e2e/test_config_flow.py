"""End-to-end configuration cycle tests exercising IMAP integrations.

What:
  Validate the full configuration ingestion pipeline: fetching YAML from the
  IMAP control mailbox, persisting state, detecting edits, and falling back to
  encrypted backups when necessary.

Why:
  These scenarios represent the most critical operator workflows. Regressions in
  config handling could disable automated triage, so the suite ensures high-level
  invariants hold even after refactors.

How:
  Spin up the :class:`~tests.unit.fakes.FakeImapBackend` fixture, append crafted
  emails, and drive :func:`mailai.core.engine.load_active_rules` while inspecting
  the resulting status snapshots and rule sets.

Interfaces:
  ``config_context`` fixture, helper ``_append_yaml``, and four test cases covering
  normal updates, user edits, corruption, and oversize mails.

Invariants & Safety:
  - Each test uses isolated temporary directories for backup and status files.
  - The helper ensures appended YAML uses the runtime-configured subject to mimic
    production traffic.
"""

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
    """Provide reusable dependencies for exercising the config pipeline.

    What:
      Constructs a tuple containing the IMAP client/backend pair plus fresh
      backup/status stores and a JSON logger writing to memory.

    Why:
      All tests require the same scaffolding; centralising the setup keeps
      assertions focused on behavioural differences instead of boilerplate.

    How:
      Leverages the ``imap_client`` fixture, instantiates
      :class:`EncryptedRulesBackup` and :class:`StatusStore` in the temporary
      directory, and returns them alongside an in-memory :class:`JsonLogger`.

    Args:
      tmp_path: Pytest temporary directory for persistent artifacts.
      imap_client: Tuple containing the :class:`MailAIImapClient` and backend.

    Returns:
      Tuple of ``(client, backend, backup, status, logger)`` consumed by tests.
    """

    client, backend = imap_client
    backup = EncryptedRulesBackup(tmp_path / "rules.bak", b"1" * 32)
    status = StatusStore(tmp_path / "status.yaml")
    logger = JsonLogger(stream=io.StringIO(), component="test")
    return client, backend, backup, status, logger


def _append_yaml(client, text: str) -> None:
    """Append YAML text to the fake IMAP control mailbox.

    What:
      Crafts an :class:`EmailMessage` with the configured subject and enqueues it
      into the control mailbox used by :func:`load_active_rules`.

    Why:
      Simulates how operators update configuration in production, ensuring the
      tests exercise the same control path (subject line, from/to headers).

    How:
      Builds the message, sets the body to ``text``, enters a writable control
      session via :meth:`MailAIImapClient.control_session`, and invokes the fake
      backend ``append`` method.

    Args:
      client: Fake IMAP client fixture.
      text: YAML string to append as the message body.
    """

    message = EmailMessage()
    message["Subject"] = get_runtime_config().mail.rules.subject
    message["From"] = "user@example.com"
    message["To"] = "mailai@local"
    message.set_content(text)
    with client.control_session(readonly=False) as mailbox:
        client.client.append(mailbox, message.as_bytes())


def test_cycle_normal_updates_status(config_context) -> None:
    """Bootstrap rules from IMAP and persist the resulting metadata.

    What:
      Runs :func:`load_active_rules` once and inspects the stored status record.

    Why:
      Establishes the baseline expectation that a normal configuration fetch
      persists UID/checksum references and returns version 2 rules.

    How:
      Unpacks the shared fixtures, calls the engine, and asserts the status store
      contains matching UID/checksum values referencing the control mailbox.
    """

    client, backend, backup, status, logger = config_context
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="cycle")
    stored = status.load()
    assert stored.config_ref is not None
    assert stored.config_ref.uid in backend.mailboxes[client.control_mailbox]
    assert stored.config_checksum == stored.config_ref.checksum
    assert rules.version == 2


def test_user_edit_detected_and_backed_up(config_context) -> None:
    """Detect a user edit and ensure new configuration is backed up.

    What:
      Simulates deleting the previous rules mail, appending an edited variant,
      and verifying that the engine records the new UID plus checksum.

    Why:
      Guarantees MailAI recognises configuration updates by UID changes and keeps
      the encrypted backup in sync.

    How:
      Perform an initial bootstrap, delete the original message, append edited
      YAML, call :func:`load_active_rules` again, and assert the stored reference
      reflects the new UID with matching checksum and updated description.
    """

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


def test_corruption_falls_back_to_backup(config_context) -> None:
    """Handle malformed YAML by restoring from backup.

    What:
      Appends invalid YAML, triggers rule loading, and checks the backup
      restoration path.

    Why:
      Confirms that corrupted configuration does not break rule evaluation and
      that the status store records the ``config_invalid`` event.

    How:
      After an initial successful sync, append broken YAML, invoke
      :func:`load_active_rules`, and assert the events list reports the error and
      ``restored_rules_from_backup`` is set.
    """

    client, backend, backup, status, logger = config_context
    load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="first")
    _append_yaml(client, "not: [yaml")
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="corrupt")
    stored = status.load()
    assert any(event.type == "config_invalid" for event in stored.events)
    assert stored.restored_rules_from_backup is True
    assert rules.version == 2


def test_oversize_mail_triggers_restoration(config_context) -> None:
    """Truncate oversize YAML and recover from backup.

    What:
      Appends a body exceeding the configured hard limit and ensures the engine
      restores from the encrypted backup.

    Why:
      Protects against IMAP abuse where large messages could exhaust resources or
      prevent configuration updates.

    How:
      Perform an initial sync, append an oversized payload, run the loader, and
      assert that the final status event records ``oversize`` and that rules fall
      back to version 2 from the backup.
    """

    client, backend, backup, status, logger = config_context
    load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="first")
    hard_limit = get_runtime_config().mail.rules.limits.hard_limit
    _append_yaml(client, "x" * (hard_limit + 5))
    rules = load_active_rules(client=client, status=status, backup=backup, logger=logger, run_id="oversize")
    stored = status.load()
    assert stored.events[-1].type == "config_restored"
    assert stored.events[-1].details == "oversize"
    assert rules.version == 2


# TODO: Other modules in this repository still require the same What/Why/How documentation.

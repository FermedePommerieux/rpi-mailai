import pytest

from mailai.config.loader import load_rules
from mailai.config.schema import RulesV2
from mailai.imap.rules_mail import RULES_SUBJECT, read_rules, upsert_rules


def test_read_rules_bootstraps_minimal(imap_client):
    client, backend = imap_client
    document = read_rules(client)
    assert isinstance(document.model, RulesV2)
    assert document.model.meta.description == "Default MailAI policy"
    assert RULES_SUBJECT in [msg["subject"] for msg in backend.mailboxes[client.control_mailbox].values()]


def test_read_rules_repairs_corruption(imap_client):
    client, backend = imap_client
    document = read_rules(client)
    assert document.model.version == 2
    # Corrupt the primary rules message while keeping the backup intact.
    mailbox = backend.mailboxes[client.control_mailbox]
    for uid, entry in mailbox.items():
        if entry["subject"] == RULES_SUBJECT:
            entry["payload"] = b"not: yaml"
            break
    restored = read_rules(client)
    assert restored.model.version == 2
    payloads = [entry["payload"] for entry in backend.mailboxes[client.control_mailbox].values()]
    assert all(payload != b"not: yaml" for payload in payloads)


def test_upsert_rules_enforces_limits(imap_client):
    client, backend = imap_client
    model = RulesV2.minimal()
    upsert_rules(client, model)
    stored = next(entry["payload"] for entry in backend.mailboxes[client.control_mailbox].values() if entry["subject"] == RULES_SUBJECT)
    reloaded = load_rules(stored)
    assert reloaded.model.meta.description == "Default MailAI policy"

from datetime import datetime, timezone

from mailai.config.loader import get_runtime_config, parse_and_validate
from mailai.imap.rules_mail import append_minimal_template, find_latest


def test_find_latest_returns_none_when_missing(imap_client):
    client, _ = imap_client
    assert find_latest(client=client) is None


def test_append_minimal_template_creates_message(imap_client):
    client, backend = imap_client
    ref = append_minimal_template(client=client)
    assert ref is not None
    assert ref.uid in backend.mailboxes[client.control_mailbox]
    assert ref.message_id is not None
    assert ref.internaldate <= datetime.now(timezone.utc)
    parsed = parse_and_validate(ref.body_text)
    assert parsed.version == 2
    subject = get_runtime_config().mail.rules.subject
    assert any(entry.subject == subject for entry in backend.mailboxes[client.control_mailbox].values())

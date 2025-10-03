from mailai.config.loader import get_runtime_config, load_status
from mailai.config.schema import Proposal, StatusV2
from mailai.imap.status_mail import upsert_status


def test_upsert_status_truncates_large_payload(imap_client):
    client, backend = imap_client
    status = StatusV2.minimal()
    status.notes = [f"note {idx} " + "x" * 1024 for idx in range(150)]
    status.proposals = [
        Proposal(rule_id=f"auto-{idx}", diff="+ sample", why="test proposal")
        for idx in range(20)
    ]
    upsert_status(client, status)
    subject = get_runtime_config().mail.status.subject
    hard_limit = get_runtime_config().mail.status.limits.hard_limit
    record = next(
        entry
        for entry in backend.mailboxes[client.control_mailbox].values()
        if entry.subject == subject
    )
    payload = record.body_text.encode(record.charset)
    assert len(payload) <= hard_limit
    parsed = load_status(payload)
    assert len(parsed.model.notes) <= 21  # 20 original + truncation marker
    assert len(parsed.model.proposals) <= 8

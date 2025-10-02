from mailai.config.loader import load_status
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
    stored = next(
        entry["payload"]
        for entry in backend.mailboxes[client.control_mailbox].values()
        if entry["subject"] == "MailAI: status.yaml"
    )
    assert len(stored) <= 128 * 1024
    parsed = load_status(stored)
    assert len(parsed.model.notes) <= 21  # 20 original + truncation marker
    assert len(parsed.model.proposals) <= 8

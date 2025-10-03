"""
Module: tests/unit/test_rules_mail.py

What:
    Validate the IMAP configuration mail helpers responsible for discovering the
    latest rules message and bootstrapping a minimal template when absent.

Why:
    Accurate discovery prevents stale configuration loads, and template creation
    ensures first-run setups populate mailboxes with a valid rules payload.

How:
    Use the fake IMAP backend to verify absence handling and template append
    behaviour, checking metadata, timestamps, and schema parsing.

Interfaces:
    test_find_latest_returns_none_when_missing,
    test_append_minimal_template_creates_message

Invariants & Safety Rules:
    - Template append must stamp current timestamps without exceeding now.
    - Parsed body should validate as the current rules schema version.
"""

from datetime import datetime, timezone

from mailai.config.loader import get_runtime_config, parse_and_validate
from mailai.imap.rules_mail import append_minimal_template, find_latest


def test_find_latest_returns_none_when_missing(imap_client):
    """
    What:
        Confirm ``find_latest`` returns ``None`` when no configuration mail is
        present.

    Why:
        The engine relies on this sentinel to decide whether to bootstrap rules
        from disk; misreporting would skip necessary template creation.

    How:
        Invoke the helper on the empty fake mailbox and assert the result is
        ``None``.

    Args:
        imap_client: Fixture providing fake IMAP client/backend pair.

    Returns:
        None
    """
    client, _ = imap_client
    assert find_latest(client=client) is None


def test_append_minimal_template_creates_message(imap_client):
    """
    What:
        Ensure the minimal template append populates the control mailbox with a
        valid configuration reference.

    Why:
        Onboarding requires a valid rules message; this verifies metadata and
        schema integrity for the generated template.

    How:
        Append the template via the helper, inspect the backend mailbox for the
        new message, parse the body, and confirm timestamps/subjects align.

    Args:
        imap_client: Fake IMAP client/backend fixture.

    Returns:
        None
    """
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


# TODO: Other modules in this repository still require the same What/Why/How documentation.

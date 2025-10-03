"""
Module: tests/unit/test_engine.py

What:
    Exercise the rule engine orchestration layer to guarantee ordered evaluation,
    idempotent application, and action dispatch semantics remain stable.

Why:
    The engine coordinates IMAP actions across potentially destructive
    operations; regression in sequencing or idempotence would lead to duplicate
    label application or repeated side effects on already-processed messages.

How:
    A lightweight fake client records issued actions while deterministic rules
    derived from the schema ensure predictable flows. Tests assert ordered
    execution, header stamping, and idempotent behaviour across consecutive runs.

Interfaces:
    FakeClient, test_engine_applies_rules_in_order, test_engine_idempotence_skips_processed_rule

Invariants & Safety Rules:
    - Actions must follow rule order while respecting stop-on-first semantics.
    - Processed messages must not trigger duplicate IMAP mutations on re-run.
"""

import io

from mailai.config.schema import RulesV2
from mailai.core.engine import Engine, Message
from mailai.utils.logging import JsonLogger


class FakeClient:
    """
    What:
        Minimal IMAP client double capturing issued actions for assertion.

    Why:
        Tests need to validate action ordering and duplication without touching a
        live mailbox; recording calls enables deterministic verification.

    How:
        Maintain an in-memory ``calls`` list appended by each method, mirroring
        the action name and parameters invoked by the engine.
    """

    def __init__(self):
        """
        What:
            Initialise the fake client with an empty action log.

        Why:
            Ensuring a clean slate for each test run avoids cross-test bleed of
            recorded operations.

        How:
            Assign an empty list to ``self.calls`` for subsequent method writes.

        Returns:
            None
        """
        self.calls = []

    def move(self, uid, destination):
        """
        What:
            Record a ``move`` invocation issued by the engine.

        Why:
            Movement operations need auditing to confirm the engine routes
            messages to the expected folders.

        How:
            Append a tuple containing the action name, UID, and destination to the
            ``calls`` log.

        Args:
            uid: The IMAP UID targeted by the move operation.
            destination: Destination mailbox path provided by the engine.

        Returns:
            None
        """
        self.calls.append(("move", uid, destination))

    def copy(self, uid, destination):
        """
        What:
            Capture a ``copy`` directive emitted during engine processing.

        Why:
            Copies are used for quarantining; ensuring correct mailbox targeting
            prevents data loss.

        How:
            Append the action details to ``self.calls`` for later assertions.

        Args:
            uid: IMAP UID subject to the copy action.
            destination: Folder receiving the copied message.

        Returns:
            None
        """
        self.calls.append(("copy", uid, destination))

    def add_label(self, uid, label):
        """
        What:
            Track label additions requested by the engine.

        Why:
            Label order matters for audit expectations; recording ensures labels
            align with configured actions.

        How:
            Append the action signature to ``self.calls``.

        Args:
            uid: UID receiving the label.
            label: Label string applied.

        Returns:
            None
        """
        self.calls.append(("add_label", uid, label))

    def mark_read(self, uid, read):
        """
        What:
            Capture ``mark_read`` instructions from the engine.

        Why:
            Ensures unread state transitions happen exactly once per rule match.

        How:
            Append a tuple describing the action and desired read state.

        Args:
            uid: UID whose read status changes.
            read: Boolean indicating the desired read state.

        Returns:
            None
        """
        self.calls.append(("mark_read", uid, read))

    def add_flag(self, uid, flag):
        """
        What:
            Record ``add_flag`` usage.

        Why:
            Flags influence server-side state; logging them allows verifying the
            engine respects action definitions.

        How:
            Append the flag assignment to ``self.calls``.

        Args:
            uid: UID receiving the flag.
            flag: Flag token added to the message.

        Returns:
            None
        """
        self.calls.append(("add_flag", uid, flag))

    def set_header(self, uid, name, value):
        """
        What:
            Capture header stamping operations.

        Why:
            Header annotations underpin idempotence by marking processed messages.

        How:
            Append the action tuple to ``self.calls`` for later verification.

        Args:
            uid: Target message UID.
            name: Header field set by the engine.
            value: Value written for the header.

        Returns:
            None
        """
        self.calls.append(("set_header", uid, name, value))


def _build_rules(stop_on_first: bool = False):
    """
    What:
        Produce a minimal ``RulesV2`` schema instance tailored for engine tests.

    Why:
        Constructing the model inline avoids filesystem dependencies and allows
        toggling ``stop_on_first_match`` for idempotence verification.

    How:
        Assemble a dictionary mirroring the schema structure and validate it via
        Pydantic's ``model_validate`` helper.

    Args:
        stop_on_first: Flag controlling early termination after a matching rule.

    Returns:
        RulesV2: Parsed configuration object for use in the engine.
    """
    data = {
        "version": 2,
        "meta": {
            "description": "test",
            "owner": "tester@example.com",
            "updated_at": "2025-01-01T00:00:00Z",
        },
        "schedule": {"learn_cron": "0 0 * * *", "inference_interval_s": 60},
        "privacy": {
            "feature_store_path": "/tmp/features.db",
            "encryption": {"enabled": True, "key_path": "/tmp/key.bin"},
            "hashing_pepper_path": "/tmp/pepper.bin",
            "hashing_salt_path": "/tmp/salt.bin",
            "max_plaintext_window_chars": 0,
        },
        "learning": {
            "enabled": True,
            "window_days": 30,
            "min_samples_per_class": 5,
            "llm": {
                "provider": "llama.cpp",
                "model": "dummy",
                "max_tokens": 128,
                "temperature": 0.2,
            },
            "embeddings": {"enabled": False},
            "rule_synthesis": {
                "enabled": True,
                "max_rules_per_pass": 3,
                "require_user_confirmation": True,
            },
            "delete_semantics": {"infer_meaning": True, "signals": []},
        },
        "defaults": {
            "case_sensitive": False,
            "stop_on_first_match": stop_on_first,
            "dry_run": False,
        },
        "rules": [
            {
                "id": "rule1",
                "description": "mark updates",
                "why": "Ensures weekly update threads are prioritised",
                "source": "deterministic",
                "priority": 5,
                "enabled": True,
                "match": {
                    "all": [
                        {"subject": {"contains": "update"}},
                    ],
                    "none": [
                        {"from": {"equals": "blocked@example.com"}}
                    ],
                },
                "actions": [
                    {"mark_read": True},
                    {"add_label": "Updates"},
                    {"stop_processing": True},
                ],
            },
            {
                "id": "rule2",
                "description": "fallback",
                "why": "Catch unmatched updates for auditing",
                "source": "deterministic",
                "priority": 10,
                "enabled": True,
                "match": {"any": [{"subject": {"contains": "update"}}]},
                "actions": [
                    {"add_label": "Fallback"},
                ],
            },
        ],
    }
    return RulesV2.model_validate(data)


def test_engine_applies_rules_in_order():
    """
    What:
        Validate that rules execute in priority order and trigger the expected IMAP
        actions once per match.

    Why:
        Ordered execution ensures deterministic behaviour, especially when rules
        request header stamping or action chaining that depends on precedence.

    How:
        Instantiate the engine with deterministic rules and process a message,
        then assert on stats and the recorded action log for ordering and header
        stamping.

    Returns:
        None
    """
    rules = _build_rules()
    client = FakeClient()
    logger = JsonLogger(stream=io.StringIO())
    engine = Engine(rules, client=client, logger=logger, run_id="test")
    message = Message(
        uid=1,
        headers={"subject": "Weekly Update", "from": "news@example.com"},
        body="body",
        mailbox="INBOX",
        size=123,
    )
    stats = engine.process([message])
    assert stats.matched_messages == 1
    action_names = [call[0] for call in client.calls]
    assert action_names[:2] == ["mark_read", "add_label"]
    assert any(call[0] == "set_header" for call in client.calls)
    assert all(call[0] != "add_label" or call[2] != "Fallback" for call in client.calls)


def test_engine_idempotence_skips_processed_rule():
    """
    What:
        Ensure processed messages do not re-trigger actions during subsequent
        engine runs.

    Why:
        Idempotence is critical when the daemon reprocesses mailboxes; duplicate
        actions can violate IMAP quotas and confuse operators.

    How:
        Run the engine twice on the same message and compare the fake client's
        call count to confirm no additional operations occur on the second pass.

    Returns:
        None
    """
    rules = _build_rules()
    client = FakeClient()
    logger = JsonLogger(stream=io.StringIO())
    engine = Engine(rules, client=client, logger=logger, run_id="test")
    message = Message(
        uid=2,
        headers={"subject": "Weekly Update", "from": "news@example.com"},
        body="body",
        mailbox="INBOX",
        size=123,
    )
    engine.process([message])
    initial = len(client.calls)
    engine.process([message])
    assert len(client.calls) == initial


# TODO: Other modules in this repository still require the same What/Why/How documentation.

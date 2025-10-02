import io

from mailai.config.schema import RulesV2
from mailai.core.engine import Engine, Message
from mailai.utils.logging import JsonLogger


class FakeClient:
    def __init__(self):
        self.calls = []

    def move(self, uid, destination):
        self.calls.append(("move", uid, destination))

    def copy(self, uid, destination):
        self.calls.append(("copy", uid, destination))

    def add_label(self, uid, label):
        self.calls.append(("add_label", uid, label))

    def mark_read(self, uid, read):
        self.calls.append(("mark_read", uid, read))

    def add_flag(self, uid, flag):
        self.calls.append(("add_flag", uid, flag))

    def set_header(self, uid, name, value):
        self.calls.append(("set_header", uid, name, value))


def _build_rules(stop_on_first: bool = False):
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

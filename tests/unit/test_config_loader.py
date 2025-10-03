import json
import pathlib

import pytest

from mailai.config.loader import (
    ConfigLoadError,
    RuntimeConfigError,
    get_runtime_config,
    load_runtime_config,
    parse_and_validate,
    reset_runtime_config,
)


def test_parse_and_validate_roundtrip():
    text = pathlib.Path("examples/rules.yaml").read_text()
    model = parse_and_validate(text)
    assert model.version == 2
    assert model.rules


def test_invalid_rules_raise_config_load_error():
    with pytest.raises(ConfigLoadError):
        parse_and_validate("version: 1")


def test_load_runtime_config_from_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.cfg"
    config_path.write_text(
        """
version: 1
paths:
  state_dir: /var/lib/mailai
  config_dir: /etc/mailai
  models_dir: /var/lib/mailai/models
imap:
  default_mailbox: Primary
  control_namespace: Drafts
  quarantine_subfolder: Quarantine
mail:
  rules:
    subject: "MailAI: custom rules"
    folder: Drafts
    limits:
      soft_limit: 1024
      hard_limit: 2048
  status:
    subject: "MailAI: status.yaml"
    folder: Drafts
    limits:
      soft_limit: 1024
      hard_limit: 2048
llm:
  model_path: /models/llm.gguf
  threads: 4
  ctx_size: 1024
  sentinel_path: /state/llm.json
  max_age: 3600
  load_timeout_s: 180
  warmup_completion_timeout_s: 20
  healthcheck_timeout_s: 15
feedback:
  enabled: true
  mailbox: Drafts/Feedback
  subject_prefix: "Feedback:"
"""
    )
    monkeypatch.setenv("MAILAI_CONFIG_PATH", str(config_path))
    reset_runtime_config()
    runtime = get_runtime_config()
    assert runtime.mail.rules.subject == "MailAI: custom rules"
    assert runtime.imap.default_mailbox == "Primary"
    assert runtime.llm.model_path == "/models/llm.gguf"
    assert runtime.llm.load_timeout_s == 180
    assert runtime.llm.warmup_completion_timeout_s == 20
    assert runtime.llm.healthcheck_timeout_s == 15


def test_load_runtime_config_from_json(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    payload = {
        "version": 1,
        "paths": {
            "state_dir": "/var/lib/mailai",
            "config_dir": "/etc/mailai",
            "models_dir": "/var/lib/mailai/models",
        },
        "imap": {
            "default_mailbox": "Inbox",
            "control_namespace": "Drafts",
            "quarantine_subfolder": "Quarantine",
        },
        "mail": {
            "rules": {
                "subject": "MailAI: rules.yaml",
                "folder": "Drafts",
                "limits": {"soft_limit": 10, "hard_limit": 20},
            },
            "status": {
                "subject": "MailAI: status.yaml",
                "folder": "Drafts",
                "limits": {"soft_limit": 10, "hard_limit": 20},
            },
        },
        "llm": {
            "model_path": "/models/model.gguf",
            "threads": 2,
            "ctx_size": 512,
            "sentinel_path": "/state/sentinel.json",
            "max_age": 7200,
            "load_timeout_s": 90,
            "warmup_completion_timeout_s": 12,
            "healthcheck_timeout_s": 8,
        },
        "feedback": {"enabled": False, "mailbox": None, "subject_prefix": None},
    }
    config_path.write_text(json.dumps(payload))
    monkeypatch.setenv("MAILAI_CONFIG_PATH", str(config_path))
    reset_runtime_config()
    runtime = load_runtime_config()
    assert runtime.llm.threads == 2
    assert runtime.mail.rules.limits.hard_limit == 20
    assert runtime.llm.load_timeout_s == 90
    assert runtime.llm.warmup_completion_timeout_s == 12
    assert runtime.llm.healthcheck_timeout_s == 8


def test_missing_config_raises(tmp_path, monkeypatch):
    missing = tmp_path / "nope.cfg"
    monkeypatch.setenv("MAILAI_CONFIG_PATH", str(missing))
    reset_runtime_config()
    with pytest.raises(RuntimeConfigError):
        load_runtime_config()

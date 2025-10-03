"""
Module: tests/unit/test_config_loader.py

What:
    Validate the configuration loader helpers by exercising happy-path parsing,
    multi-format ingestion, and error signalling routines exposed to callers.

Why:
    These tests ensure the runtime configuration bootstrap defends against
    malformed payloads and environment drift so that the daemon never operates
    with partially initialised settings or stale cached state.

How:
    Each test feeds representative YAML/JSON payloads or missing files through
    the loader façade, asserting on structured schema objects and raised
    exceptions to guarantee invariants around timeouts, mailbox mappings, and
    runtime cache resets.

Interfaces:
    test_parse_and_validate_roundtrip, test_invalid_rules_raise_config_load_error,
    test_load_runtime_config_from_yaml, test_load_runtime_config_from_json,
    test_missing_config_raises

Invariants & Safety Rules:
    - Runtime cache must be cleared before each load to avoid cross-test bleed.
    - Loader should surface ConfigLoadError/RuntimeConfigError precisely when the
      on-disk payload is invalid or missing.
"""

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
    """ 
    What:
        Confirm that a valid YAML rules payload round-trips through the parser
        into a strongly typed configuration model.

    Why:
        The runtime relies on schema validation to prevent partial rule sets or
        stale versions from entering the system; this guard catches regression in
        that validation layer.

    How:
        Load the canonical example rules file, parse it via ``parse_and_validate``,
        and assert key schema attributes such as version and non-empty rule list.

    Returns:
        None
    """
    text = pathlib.Path("examples/rules.yaml").read_text()
    model = parse_and_validate(text)
    assert model.version == 2
    assert model.rules


def test_invalid_rules_raise_config_load_error():
    """
    What:
        Ensure the parser rejects malformed rule payloads.

    Why:
        Invalid configuration must fail fast so that the daemon refuses to start
        with unsupported schema versions or incomplete rule definitions.

    How:
        Attempt to parse a deliberately incorrect version declaration and assert
        that ``ConfigLoadError`` is raised.

    Returns:
        None

    Raises:
        AssertionError: If ``ConfigLoadError`` is not raised as expected.
    """
    with pytest.raises(ConfigLoadError):
        parse_and_validate("version: 1")


def test_load_runtime_config_from_yaml(tmp_path, monkeypatch):
    """
    What:
        Exercise the YAML configuration loader path from disk to cached runtime
        object.

    Why:
        YAML is the primary configuration format for deployments; verifying the
        loader preserves timeouts and mailbox metadata protects LLM warmup limits
        and IMAP namespace assumptions.

    How:
        Write a representative YAML payload to a temporary file, point the
        environment override to it, reset the runtime cache, and assert the
        resulting ``RuntimeConfig`` mirrors the source values.

    Args:
        tmp_path: Pytest fixture providing a temporary directory for config files.
        monkeypatch: Pytest fixture used to manipulate environment variables.

    Returns:
        None
    """
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
    """
    What:
        Validate the JSON configuration loader pathway mirrors the YAML behaviour.

    Why:
        Operators may provide JSON instead of YAML; parity ensures the
        configuration validator remains format-agnostic while still enforcing
        LLM timeout and mailbox invariants.

    How:
        Serialize a dictionary to JSON, write it to disk, point the loader at the
        file, reset cache state, and confirm the resulting model echoes the
        encoded values.

    Args:
        tmp_path: Temporary directory fixture for constructing the JSON file.
        monkeypatch: Fixture for setting environment variables in isolation.

    Returns:
        None
    """
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
    """
    What:
        Confirm the loader surfaces ``RuntimeConfigError`` when no configuration
        file exists at the expected path.

    Why:
        Missing configuration must halt startup to avoid defaulting to unsafe
        values or performing IMAP actions without explicit operator consent.

    How:
        Set the configuration environment variable to a non-existent file, reset
        cached state, and assert ``RuntimeConfigError`` is raised during load.

    Args:
        tmp_path: Fixture providing a temporary directory for constructing paths.
        monkeypatch: Fixture for mutating environment variables.

    Returns:
        None

    Raises:
        AssertionError: If ``RuntimeConfigError`` is not triggered.
    """
    missing = tmp_path / "nope.cfg"
    monkeypatch.setenv("MAILAI_CONFIG_PATH", str(missing))
    reset_runtime_config()
    with pytest.raises(RuntimeConfigError):
        load_runtime_config()


# TODO: Other modules in this repository still require the same What/Why/How documentation.

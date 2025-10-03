"""Unit tests for the ``mailai.accountctl`` module.

What:
  Validate CLI workflows that create, list, show, and remove account entries in
  ``accounts.yaml`` documents.

Why:
  Ensures the new management tool behaves deterministically, writes YAML safely,
  and guards against regressions when editing account metadata.

How:
  Invoke :func:`mailai.accountctl.main` with argument lists to simulate CLI
  usage, then inspect filesystem side effects and captured output.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from mailai import accountctl


def test_set_creates_accounts_document(tmp_path: pathlib.Path) -> None:
    """Ensure ``set`` creates a new document with validated content.

    What:
      Run the ``set`` subcommand and inspect the resulting YAML document.

    Why:
      Confirms the command creates missing files and persists the expected
      structure for downstream tooling.

    How:
      Invoke :func:`accountctl.main` with a dedicated temporary path, then read
      back the generated YAML and compare to the expected dictionary.
    """

    accounts_path = tmp_path / "accounts.yaml"
    exit_code = accountctl.main(
        [
            "--accounts",
            str(accounts_path),
            "set",
            "personal",
            "--host",
            "imap.example.net",
            "--port",
            "993",
            "--username",
            "doe@example.net",
            "--password-file",
            "/run/secrets/personal_password",
            "--control-namespace",
            "Drafts",
            "--quarantine-subfolder",
            "Quarantine",
            "--hash-salt",
            "/run/secrets/hash_salt",
            "--pepper",
            "/run/secrets/pepper",
            "--sqlcipher-key",
            "/run/secrets/sqlcipher_key",
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(accounts_path.read_text())
    assert payload == {
        "accounts": [
            {
                "imap": {
                    "control_namespace": "Drafts",
                    "host": "imap.example.net",
                    "password_file": "/run/secrets/personal_password",
                    "port": 993,
                    "quarantine_subfolder": "Quarantine",
                    "ssl": True,
                    "username": "doe@example.net",
                },
                "name": "personal",
                "secrets": {
                    "hash_salt": "/run/secrets/hash_salt",
                    "pepper": "/run/secrets/pepper",
                    "sqlcipher_key": "/run/secrets/sqlcipher_key",
                },
            }
        ]
    }


def test_list_and_show_commands(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise ``list`` and ``show`` output helpers.

    What:
      Populate two accounts, then verify the listing order and JSON rendering of
      individual entries.

    Why:
      Validates deterministic sorting and redaction behaviour in ``show``.

    How:
      - Create two accounts via consecutive ``set`` invocations.
      - Run ``list`` and capture stdout.
      - Run ``show`` and parse the JSON payload.
    """

    accounts_path = tmp_path / "accounts.yaml"
    for name in ("personal", "work"):
        accountctl.main(
            [
                "--accounts",
                str(accounts_path),
                "set",
                name,
                "--host",
                f"{name}.imap",
                "--port",
                "993",
                "--username",
                f"{name}@example.net",
                "--password-file",
                f"/run/secrets/{name}_password",
                "--control-namespace",
                "Drafts",
                "--quarantine-subfolder",
                "Quarantine",
                "--hash-salt",
                "/run/secrets/hash_salt",
                "--pepper",
                f"/run/secrets/{name}_pepper",
                "--sqlcipher-key",
                "/run/secrets/sqlcipher_key",
            ]
        )
        capsys.readouterr()

    exit_code = accountctl.main(["--accounts", str(accounts_path), "list"])
    assert exit_code == 0
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured == ["personal", "work"]

    exit_code = accountctl.main(["--accounts", str(accounts_path), "show", "personal"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "personal"
    assert payload["imap"]["host"] == "personal.imap"
    assert payload["secrets"]["pepper"] == "personal_pepper"


def test_remove_missing_account(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ensure deleting an absent account surfaces a structured error.

    What:
      Invoke ``remove`` when no account exists.

    Why:
      Confirms the command fails gracefully without mutating files.

    How:
      Call :func:`accountctl.main` with a non-existent account and inspect the
      emitted JSON error.
    """

    accounts_path = tmp_path / "accounts.yaml"
    exit_code = accountctl.main(["--accounts", str(accounts_path), "remove", "missing"])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "not_found"

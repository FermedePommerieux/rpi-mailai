"""Module: mailai/accountctl.py

What:
  Provide a non-interactive command-line utility for configuring MailAI IMAP
  accounts stored inside ``accounts.yaml`` documents. Operators can list,
  inspect, create, update, and delete account definitions that the runtime uses
  when connecting to control mailboxes.

Why:
  MailAI deployments frequently rotate credentials and secret file paths while
  keeping other infrastructure stable. A dedicated tool ensures administrators
  can manage account metadata deterministically without hand-editing YAML files
  and risking schema drift or indentation mistakes that would break automated
  provisioning.

How:
  - Parse ``accounts.yaml`` into strict Pydantic models enforcing the expected
    structure for IMAP and secret settings.
  - Expose ``list``, ``show``, ``set``, and ``remove`` subcommands through
    ``argparse`` with script-friendly arguments.
  - Normalise boolean handling (e.g. SSL flags) and write the validated
    document back to disk using ``yaml.safe_dump`` while preserving ordering for
    readability.
  - Return deterministic exit codes and print JSON-compatible output so shell
    scripts can consume the results safely.

Interfaces:
  - main(argv: Optional[List[str]] = None) -> int

Invariants:
  - Secret values are never printed; the tool only handles filesystem paths to
    secret material.
  - ``accounts.yaml`` is rewritten atomically by writing to a temporary file and
    renaming it into place, preventing partial writes during crashes.
  - Account names are unique; ``set`` replaces existing entries atomically when
    a name collision occurs.

Security/Perf:
  - All inputs are validated by Pydantic models with ``extra="forbid"`` to
    prevent unrecognised keys from slipping into the runtime configuration.
  - The implementation avoids interactive prompts, keeping the tool scriptable
    for automated rotations and infrastructure-as-code workflows.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from typing import List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AccountSecrets(BaseModel):
    """Filesystem paths pointing at secret material for an account.

    What:
      Track the locations of salted hashing material, pepper files, and the
      SQLCipher key required by the MailAI feature store.

    Why:
      Centralising these paths ensures the runtime can ingest encrypted
      artefacts without leaking plaintext; administrators rotate these files via
      secret stores such as Docker secrets or tmpfs mounts.

    How:
      Enforce presence of three string fields representing absolute or relative
      filesystem paths. Additional keys are rejected thanks to the strict
      Pydantic model configuration.

    Attributes:
      hash_salt: Path to the salted hashing secret.
      pepper: Path containing the per-account pepper material.
      sqlcipher_key: Path to the SQLCipher key used by the feature store.
    """

    model_config = ConfigDict(extra="forbid")

    hash_salt: str
    pepper: str
    sqlcipher_key: str


class AccountImapSettings(BaseModel):
    """Connection metadata for a MailAI-controlled IMAP account.

    What:
      Define the host, credentials, and mailbox naming conventions for the
      control namespace that stores configuration and status messages.

    Why:
      The runtime requires deterministic IMAP parameters to locate control
      mailboxes, authenticate safely, and isolate quarantine folders.

    How:
      Store primitive types for connection details alongside folder naming
      conventions. Enforce port ranges and restrict unexpected keys through
      ``extra="forbid"``.

    Attributes:
      host: IMAP server hostname or IP.
      port: TCP port (typically 993 for IMAPS).
      ssl: Whether to establish the session using TLS from the start.
      username: IMAP login name.
      password_file: Path to the password file consumed by secret managers.
      control_namespace: Mailbox holding control messages (defaults to Drafts).
      quarantine_subfolder: Folder where the engine moves quarantined mail.
    """

    model_config = ConfigDict(extra="forbid")

    host: str
    port: int = Field(ge=1, le=65535)
    ssl: bool = True
    username: str
    password_file: str
    control_namespace: str
    quarantine_subfolder: str


class AccountEntry(BaseModel):
    """Single account entry stored in ``accounts.yaml``.

    What:
      Compose IMAP connection settings with associated secret file paths under a
      human-readable account name.

    Why:
      MailAI supports multi-account deployments; naming each entry allows CLI
      tooling and the runtime to select the target environment explicitly.

    How:
      Combine :class:`AccountImapSettings` and :class:`AccountSecrets` within a
      strict Pydantic model that forbids extra keys. The ``name`` field is used
      for de-duplication and CLI lookups.

    Attributes:
      name: Logical account identifier unique within the document.
      imap: Connection settings for the account.
      secrets: Secret file references for the account.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    imap: AccountImapSettings
    secrets: AccountSecrets


class AccountsDocument(BaseModel):
    """Top-level ``accounts.yaml`` document containing one or more accounts.

    What:
      Represent the structure persisted on disk including the ``accounts`` list.

    Why:
      Wrapping account entries provides a stable anchor for future extensions
      (e.g. global defaults) while enabling validation before writing to disk.

    How:
      Maintain a list of :class:`AccountEntry` instances with ``extra="forbid"``
      semantics for safety.

    Attributes:
      accounts: Collection of configured MailAI accounts.
    """

    model_config = ConfigDict(extra="forbid")

    accounts: List[AccountEntry] = Field(default_factory=list)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``mailai-accountctl`` command-line utility.

    What:
      Parse arguments, route to the appropriate subcommand, and return a process
      exit code indicating success or failure.

    Why:
      Centralising argument parsing keeps error handling and documentation
      consistent while enabling reuse in unit tests by invoking the function
      directly.

    How:
      - Build an ``argparse`` parser with ``list``, ``show``, ``set``, and
        ``remove`` subcommands.
      - Normalise the accounts file path and dispatch to dedicated helper
        functions that perform validation and file operations.
      - Catch :class:`ValidationError` raised by Pydantic to emit user-friendly
        error messages without exposing stack traces.

    Args:
      argv: Optional list of argument strings; defaults to ``sys.argv[1:]`` when
        ``None``.

    Returns:
      ``0`` on success and ``1`` on validation or lookup failures.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    path = pathlib.Path(args.accounts)
    try:
        if args.command == "list":
            return _cmd_list(path)
        if args.command == "show":
            return _cmd_show(path, args.name)
        if args.command == "set":
            return _cmd_set(
                path=path,
                name=args.name,
                host=args.host,
                port=args.port,
                ssl=args.ssl,
                username=args.username,
                password_file=args.password_file,
                control_namespace=args.control_namespace,
                quarantine_subfolder=args.quarantine_subfolder,
                hash_salt=args.hash_salt,
                pepper=args.pepper,
                sqlcipher_key=args.sqlcipher_key,
            )
        if args.command == "remove":
            return _cmd_remove(path, args.name)
    except ValidationError as exc:  # pragma: no cover - exercised via unit tests
        _print_error("validation_error", exc.errors())
        return 1
    parser.error("Unknown command")
    return 1


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the account control CLI.

    What:
      Define the CLI surface exposed by ``mailai-accountctl`` including global
      and subcommand-specific options.

    Why:
      Keeping parser construction in a separate helper simplifies unit testing
      and keeps :func:`main` focused on dispatching logic.

    How:
      Instantiate :class:`argparse.ArgumentParser`, register subcommands, and
      declare the arguments required to manage accounts deterministically.

    Returns:
      Configured :class:`argparse.ArgumentParser` instance.
    """

    parser = argparse.ArgumentParser(description="Manage MailAI IMAP account definitions")
    parser.add_argument(
        "--accounts",
        type=str,
        default="accounts.yaml",
        help="Path to the accounts YAML document (defaults to ./accounts.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List configured account names")

    show_parser = subparsers.add_parser("show", help="Show a single account as JSON")
    show_parser.add_argument("name", help="Account name to inspect")

    set_parser = subparsers.add_parser("set", help="Create or update an account definition")
    set_parser.add_argument("name", help="Logical account name")
    set_parser.add_argument("--host", required=True, help="IMAP server hostname")
    set_parser.add_argument("--port", type=int, default=993, help="IMAP server port")
    set_parser.add_argument("--ssl", dest="ssl", action="store_true", default=True, help="Enable TLS")
    set_parser.add_argument("--no-ssl", dest="ssl", action="store_false", help="Disable TLS")
    set_parser.add_argument("--username", required=True, help="IMAP username")
    set_parser.add_argument(
        "--password-file",
        required=True,
        help="Filesystem path containing the IMAP password",
    )
    set_parser.add_argument(
        "--control-namespace",
        required=True,
        help="Mailbox storing rules/status documents",
    )
    set_parser.add_argument(
        "--quarantine-subfolder",
        required=True,
        help="Mailbox used for quarantined messages",
    )
    set_parser.add_argument("--hash-salt", required=True, help="Path to the hashing salt secret")
    set_parser.add_argument("--pepper", required=True, help="Path to the pepper secret")
    set_parser.add_argument(
        "--sqlcipher-key",
        required=True,
        help="Path to the SQLCipher key for the feature store",
    )

    remove_parser = subparsers.add_parser("remove", help="Delete an account definition")
    remove_parser.add_argument("name", help="Account name to delete")

    return parser


def _cmd_list(path: pathlib.Path) -> int:
    """List configured account names.

    What:
      Load the accounts document (if present) and print each account name on its
      own line.

    Why:
      Administrators frequently need to audit which accounts are registered
      without revealing secret material. Listing keeps automation simple by
      emitting newline-delimited identifiers.

    How:
      - Load the existing document via :func:`_load_accounts`.
      - Iterate over the ``accounts`` list and print names in deterministic
        sorted order.

    Args:
      path: Location of the accounts YAML file.

    Returns:
      ``0`` on success.
    """

    document = _load_accounts(path)
    for entry in sorted(document.accounts, key=lambda item: item.name):
        print(entry.name)
    return 0


def _cmd_show(path: pathlib.Path, name: str) -> int:
    """Render a single account as JSON for inspection.

    What:
      Retrieve a named account entry and emit the serialised representation with
      secrets redacted.

    Why:
      Operators can confirm configuration values during rollouts without
      leaking actual credential material.

    How:
      - Load the document and search for the named account.
      - If present, convert to a dictionary and replace secret paths with their
        basename to avoid revealing directory layouts.
      - Print JSON to stdout and return ``0``. Emit a structured error when the
        account is missing.

    Args:
      path: Accounts file path.
      name: Account name to display.

    Returns:
      ``0`` on success, ``1`` when the account is not found.
    """

    document = _load_accounts(path)
    for entry in document.accounts:
        if entry.name == name:
            payload = entry.model_dump()
            payload["secrets"] = {
                key: pathlib.Path(value).name for key, value in payload["secrets"].items()
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
    _print_error("not_found", {"name": name})
    return 1


def _cmd_set(
    *,
    path: pathlib.Path,
    name: str,
    host: str,
    port: int,
    ssl: bool,
    username: str,
    password_file: str,
    control_namespace: str,
    quarantine_subfolder: str,
    hash_salt: str,
    pepper: str,
    sqlcipher_key: str,
) -> int:
    """Create or update an account definition in ``accounts.yaml``.

    What:
      Insert or replace an account entry with validated IMAP and secret
      settings.

    Why:
      Keeping account mutations idempotent avoids manual YAML editing errors and
      enables infrastructure tooling to run the command repeatedly.

    How:
      - Load the existing document (creating an empty one when missing).
      - Build an :class:`AccountEntry` and either append or replace an entry with
        the same name.
      - Persist the document atomically via :func:`_write_accounts`.

    Args:
      path: Accounts file path.
      name: Logical account identifier.
      host: IMAP server hostname.
      port: IMAP server port.
      ssl: Whether to establish an SSL session immediately.
      username: IMAP login username.
      password_file: Filesystem path containing the IMAP password.
      control_namespace: Mailbox storing configuration/status documents.
      quarantine_subfolder: Mailbox where quarantined messages are placed.
      hash_salt: Path to the hashing salt secret.
      pepper: Path to the pepper secret.
      sqlcipher_key: Path to the SQLCipher key.

    Returns:
      ``0`` on success.
    """

    document = _load_accounts(path)
    entry = AccountEntry(
        name=name,
        imap=AccountImapSettings(
            host=host,
            port=port,
            ssl=ssl,
            username=username,
            password_file=password_file,
            control_namespace=control_namespace,
            quarantine_subfolder=quarantine_subfolder,
        ),
        secrets=AccountSecrets(hash_salt=hash_salt, pepper=pepper, sqlcipher_key=sqlcipher_key),
    )

    updated = [item for item in document.accounts if item.name != name]
    updated.append(entry)
    document.accounts = sorted(updated, key=lambda item: item.name)
    _write_accounts(path, document)
    print(json.dumps({"status": "updated", "name": name}))
    return 0


def _cmd_remove(path: pathlib.Path, name: str) -> int:
    """Remove an account definition from ``accounts.yaml``.

    What:
      Delete the specified account entry if it exists.

    Why:
      Simplifies credential decommissioning without hand-editing the YAML
      document.

    How:
      - Load the document; if the entry exists filter it out and rewrite the
        file.
      - Emit a structured error when attempting to delete a non-existent
        account.

    Args:
      path: Accounts file path.
      name: Account name to remove.

    Returns:
      ``0`` on success, ``1`` if the account was missing.
    """

    document = _load_accounts(path)
    remaining = [item for item in document.accounts if item.name != name]
    if len(remaining) == len(document.accounts):
        _print_error("not_found", {"name": name})
        return 1
    document.accounts = remaining
    _write_accounts(path, document)
    print(json.dumps({"status": "removed", "name": name}))
    return 0


def _load_accounts(path: pathlib.Path) -> AccountsDocument:
    """Load the accounts document from disk.

    What:
      Parse ``accounts.yaml`` into a validated :class:`AccountsDocument`.

    Why:
      Validation ensures upstream commands operate on a consistent structure and
      alerts operators to schema drift immediately.

    How:
      - Return an empty document when the file is absent.
      - Otherwise read the YAML, default to an empty dict when the file is
        blank, and validate via Pydantic.

    Args:
      path: Accounts file path.

    Returns:
      Validated :class:`AccountsDocument` instance.
    """

    if not path.exists():
        return AccountsDocument()
    data = yaml.safe_load(path.read_text()) or {}
    return AccountsDocument.model_validate(data)


def _write_accounts(path: pathlib.Path, document: AccountsDocument) -> None:
    """Persist the accounts document to disk atomically.

    What:
      Serialize the :class:`AccountsDocument` into YAML and write it safely.

    Why:
      Atomic writes avoid partial file corruption during power loss or crashes,
      which is critical for headless Raspberry Pi deployments.

    How:
      - Create parent directories when missing.
      - Serialise the document into YAML with sorted keys for deterministic diffs.
      - Write to a temporary file in the target directory and rename it into
        place.

    Args:
      path: Destination path for ``accounts.yaml``.
      document: Validated accounts document to persist.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(document.model_dump(mode="json"), sort_keys=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as handle:
        handle.write(payload)
        temp_path = pathlib.Path(handle.name)
    temp_path.replace(path)


def _print_error(kind: str, detail: object) -> None:
    """Emit structured JSON errors for CLI consumption.

    What:
      Print a JSON object describing the error type and details.

    Why:
      Structured output keeps shell scripts from relying on fragile string
      matching while avoiding the need for logging frameworks.

    How:
      Serialize a dictionary containing ``error`` and ``detail`` keys to stdout.

    Args:
      kind: Machine-readable error identifier.
      detail: Additional context about the failure.
    """

    print(json.dumps({"error": kind, "detail": detail}, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


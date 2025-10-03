"""Module: mailai/cli.py

What:
  Command-line interface that exposes the MailAI automation entry points for
  one-off runs, continuous monitoring, learning triggers, and diagnostics.

Why:
  Provides a simple operational control surface for administrators scripting or
  manually orchestrating MailAI on Raspberry Pi deployments without importing
  Python modules directly. Centralizing the CLI logic keeps argument parsing and
  logging configuration consistent across commands.

How:
  - Builds an ``argparse`` parser with subcommands that map directly to the
    supported operational modes.
  - Validates the configuration path upfront to avoid dispatching commands with
    missing inputs.
  - Loads the YAML-based rules document through the shared configuration loader
    to ensure checksum tracking remains uniform across commands.
  - Emits structured logs and user-facing console output before returning
    process status codes to the caller.

Interfaces:
  main(argv: Optional[List[str]] = None) -> int

Invariants:
  - Commands never perform network side effects beyond configuration loading;
    behavioral actions are stubs until the engine is wired in.
  - The CLI must exit with deterministic codes (0 success, 1 parser failure)
    for shell automation friendliness.
  - Logging always includes the checksum when a configuration document is
    successfully loaded, providing traceability for audits.

Security/Perf:
  - Accepts file paths only from the invoking shell, avoiding implicit
    discovery of configuration files to reduce accidental disclosure.
  - All diagnostics honor the ``redact`` flag to prevent leaking sensitive
    metadata in shared terminals.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import List, Optional

from .config.loader import load_rules
from .utils.logging import get_logger


def main(argv: Optional[List[str]] = None) -> int:
    """Execute the MailAI command-line interface.

    What:
      Parses command-line arguments, loads the requested MailAI configuration,
      and dispatches the selected operational command while returning an exit
      status suitable for shell scripts.

    Why:
      Offers a unified entry point for operators to trigger MailAI capabilities
      without importing modules, enabling cron jobs and manual debugging to
      share the same behavior and logging pipeline.

    How:
      - Builds the argument parser with subcommands covering once, watch,
        learn-now, and diagnostics modes.
      - Validates required configuration paths and reads the YAML rules through
        ``load_rules`` to leverage checksum validation.
      - Logs each invocation with contextual metadata and prints concise status
        messages before returning ``0`` or propagating parser errors.

    Args:
      argv: Optional iterable of command-line strings; defaults to
        ``sys.argv[1:]`` when ``None``.

    Returns:
      Process exit code ``0`` for success, or raises ``SystemExit`` on parser
      errors which equates to a non-zero exit in shell contexts.

    Raises:
      SystemExit: Propagated when ``argparse`` encounters validation failures
        or when ``parser.error`` is explicitly invoked for invalid commands.
    """
    parser = argparse.ArgumentParser(description="MailAI offline email triage agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    once_parser = subparsers.add_parser("once", help="Run a single inference pass")
    once_parser.add_argument("config_path", type=pathlib.Path)

    watch_parser = subparsers.add_parser("watch", help="Continuously process the mailbox")
    watch_parser.add_argument("config_path", type=pathlib.Path)
    watch_parser.add_argument("--interval", type=int, default=None, help="Override inference interval")

    learn_parser = subparsers.add_parser("learn-now", help="Trigger the learning pipeline")
    learn_parser.add_argument("config_path", type=pathlib.Path)

    diag_parser = subparsers.add_parser("diag", help="Emit diagnostics")
    diag_parser.add_argument("--redact", action="store_true", default=True)
    diag_parser.add_argument("--no-redact", dest="redact", action="store_false")

    args = parser.parse_args(argv)

    if args.command == "diag":
        return _cmd_diag(redact=args.redact)

    path: pathlib.Path = args.config_path
    if not path.exists():
        parser.error(f"Config file not found: {path}")
    document = load_rules(path.read_bytes())
    logger = get_logger("cli")

    if args.command == "once":
        logger.info("run_once", checksum=document.checksum)
        print(f"Loaded rules with checksum {document.checksum}")
        return 0
    if args.command == "watch":
        interval = args.interval or document.model.schedule.inference_interval_s
        logger.info("run_watch", checksum=document.checksum, interval=interval)
        print(f"Watching mailbox every {interval} seconds (dry-run stub)")
        return 0
    if args.command == "learn-now":
        logger.info("learn_now", checksum=document.checksum)
        print("Triggered learning cycle (stub)")
        return 0
    parser.error("Unknown command")
    return 1


def _cmd_diag(redact: bool) -> int:
    """Handle the ``diag`` subcommand.

    What:
      Emits a diagnostic payload describing the CLI health status and redaction
      mode while logging the event for audit purposes.

    Why:
      Enables quick checks that logging, configuration, and runtime wiring are
      functional without performing IMAP or LLM operations, which is helpful
      during deployments or troubleshooting.

    How:
      - Retrieves the shared CLI logger and records the diagnostic invocation
        with the requested redaction setting.
      - Prints a JSON-like summary to stdout for human or scripted inspection.
      - Returns ``0`` to signal successful completion to the shell.

    Args:
      redact: Whether sensitive values should be suppressed from diagnostics.

    Returns:
      Integer ``0`` to indicate success.
    """
    logger = get_logger("cli")
    logger.info("diagnostic", redact=redact)
    print({"status": "ok", "redaction": redact})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# TODO: Other modules require the same treatment (Quoi/Pourquoi/Comment docstrings + module header).

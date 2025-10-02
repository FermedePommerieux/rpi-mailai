"""CLI entrypoint for MailAI."""
from __future__ import annotations

import argparse
import pathlib
from typing import List, Optional

from .config.loader import load_rules
from .utils.logging import get_logger


def main(argv: Optional[List[str]] = None) -> int:
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
    logger = get_logger("cli")
    logger.info("diagnostic", redact=redact)
    print({"status": "ok", "redaction": redact})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

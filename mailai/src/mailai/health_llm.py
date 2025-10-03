"""Runtime health probe for the embedded llama-cpp-python model."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Final

_DEFAULT_SENTINEL: Final[str] = "/var/lib/mailai/.cache/llm_ready.json"


def _int_from_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text()
    except FileNotFoundError:
        raise SystemExit(f"sentinel missing: {path}") from None
    except OSError as exc:  # pragma: no cover - filesystem surface
        raise SystemExit(f"unable to read sentinel {path}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"sentinel {path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):  # pragma: no cover - defensive guard
        raise SystemExit(f"sentinel {path} must contain an object")

    return payload


def _touch_model(model_file: Path) -> None:
    try:
        with model_file.open("rb") as handle:
            handle.read(1)
    except OSError as exc:  # pragma: no cover - filesystem surface
        raise SystemExit(f"unable to read model: {exc}") from exc


def main() -> int:
    model_path = os.environ.get("LLM_MODEL_PATH")
    if not model_path:
        print("LLM_MODEL_PATH is not set", file=sys.stderr)
        return 1

    model_file = Path(model_path)
    if not model_file.is_file():
        print(f"model file not found: {model_file}", file=sys.stderr)
        return 1

    sentinel_path = Path(os.environ.get("LLM_HEALTH_SENTINEL", _DEFAULT_SENTINEL))
    max_age = _int_from_env("LLM_HEALTH_MAX_AGE", 86400)

    try:
        payload = _load_json(sentinel_path)
    except SystemExit as exc:
        print(exc.args[0], file=sys.stderr)
        return 1

    expected_model = payload.get("model_path")
    if expected_model != str(model_file):
        print(
            f"sentinel model mismatch: expected {expected_model}, actual {model_file}",
            file=sys.stderr,
        )
        return 1

    threads = _int_from_env("LLM_N_THREADS", 3)
    ctx_size = _int_from_env("LLM_CTX_SIZE", 2048)

    sentinel_threads = payload.get("threads")
    sentinel_ctx = payload.get("ctx_size")
    if sentinel_threads != threads or sentinel_ctx != ctx_size:
        print(
            "sentinel parameters differ from environment", file=sys.stderr
        )
        return 1

    completed_at = payload.get("completed_at")
    if not isinstance(completed_at, (int, float)):
        print("sentinel missing completion timestamp", file=sys.stderr)
        return 1

    if completed_at < time.time() - max_age:
        print("sentinel is stale", file=sys.stderr)
        return 1

    try:
        sentinel_mtime = sentinel_path.stat().st_mtime
    except OSError as exc:  # pragma: no cover - filesystem surface
        print(f"unable to stat sentinel: {exc}", file=sys.stderr)
        return 1

    try:
        model_mtime = model_file.stat().st_mtime
    except OSError as exc:  # pragma: no cover - filesystem surface
        print(f"unable to stat model: {exc}", file=sys.stderr)
        return 1

    if sentinel_mtime < model_mtime:
        print("model updated since warmup", file=sys.stderr)
        return 1

    _touch_model(model_file)

    response = payload.get("response")
    if not isinstance(response, str) or not response.strip():
        print("sentinel completion response missing", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - module executable entry point
    raise SystemExit(main())

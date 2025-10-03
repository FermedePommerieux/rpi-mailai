"""MailAI LLM health probe for validating warmup sentinels and model readiness.

What:
    Provide the CLI entry point that checks the on-disk warmup sentinel produced
    by the `llm_local` bootstrap, ensuring the configured llama-cpp-python model
    still matches the parameters captured during the last warmup run.
Why:
    The health check is invoked by container orchestrators and CI pipelines to
    guarantee the Raspberry Pi deployment only reports "ready" when the local
    LLM can answer completions. We centralize the validation rules here so both
    manual operators and automated supervisors receive the same deterministic
    verdict.
How:
    Load runtime configuration, resolve environment overrides, and compare the
    warmup sentinel metadata with the active filesystem state. The module keeps
    I/O minimal—only the sentinel JSON and the model file are touched—to avoid
    thrashing SD storage. Any mismatch or stale value terminates the process
    with a non-zero code.
Interfaces:
    main() -> int
Invariants:
    - Never attempt IMAP or network calls; the probe must stay fast and local.
    - Refuse to report success when the sentinel predates the model timestamp.
    - Require non-empty completion text to confirm llama-cpp-python responded.
Security/Performance:
    - Avoid leaking model content by never logging completion bodies.
    - Fail-fast with actionable stderr messages to aid unattended monitoring.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .config.loader import RuntimeConfigError, get_runtime_config


def _int_from_env(name: str, default: int) -> int:
    """Return an integer from ``os.environ`` or ``default`` on absence.

    What:
      Parse the named environment variable as an integer, using ``default`` when
      unset.

    Why:
      Health probes allow operators to override timeouts via environment
      variables; this helper ensures type safety and clear error messages.

    How:
      Look up ``name`` in the environment, attempt to convert to ``int``, and
      abort with :class:`SystemExit` on invalid input.

    Args:
      name: Environment variable to read.
      default: Fallback value when the variable is missing.

    Returns:
      Parsed integer value.
    """

    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc


def _load_json(path: Path) -> dict[str, object]:
    """Read and decode a JSON file into a dictionary.

    What:
      Load the warmup sentinel describing the expected model parameters.

    Why:
      The sentinel drives comparisons between runtime configuration and
      filesystem state; malformed or missing files should abort the probe.

    How:
      Read the file, decode JSON, ensure the payload is a dictionary, and raise
      :class:`SystemExit` with helpful messages on failure.

    Args:
      path: Location of the sentinel file.

    Returns:
      Dictionary representing the sentinel contents.
    """

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
    """Attempt to open the model file to confirm readability.

    What:
      Perform a minimal read to ensure the model still exists and is accessible.

    Why:
      Health checks should fail fast if the model was removed or permissions
      changed, preventing misleading "ready" responses.

    How:
      Open the file in binary mode and read a single byte, raising
      :class:`SystemExit` if an :class:`OSError` occurs.

    Args:
      model_file: Path to the llama-cpp-python model file.
    """

    try:
        with model_file.open("rb") as handle:
            handle.read(1)
    except OSError as exc:  # pragma: no cover - filesystem surface
        raise SystemExit(f"unable to read model: {exc}") from exc


def main() -> int:
    """Check local LLM readiness based on the warmup sentinel.

    What:
        Validate that the llama-cpp-python model previously warmed up is still
        available, unchanged, and responsive according to the JSON sentinel.
        Returns an exit status suitable for container health probes.
    Why:
        Health endpoints must guard against silent model replacement, SD card
        corruption, or configuration drift. Returning success without checking
        these invariants would allow the orchestrator to route traffic to an
        unready node, leading to failed completions for operators.
    How:
        Resolve runtime configuration, apply environment overrides, load the
        sentinel, and compare model path, threading parameters, timestamp, and
        completion payload. Any discrepancy produces stderr diagnostics and a
        non-zero exit code; a clean pass returns zero.

    Returns:
        int: ``0`` when the sentinel and filesystem state agree; ``1`` when a
        mismatch or I/O error prevents validation.
    """
    try:
        runtime = get_runtime_config()
    except RuntimeConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    llm_cfg = runtime.llm
    model_path = os.environ.get("LLM_MODEL_PATH", llm_cfg.model_path)
    if not model_path:
        print("LLM model path is not configured", file=sys.stderr)
        return 1

    model_file = Path(model_path)
    if not model_file.is_file():
        print(f"model file not found: {model_file}", file=sys.stderr)
        return 1

    sentinel_path = Path(os.environ.get("LLM_HEALTH_SENTINEL", llm_cfg.sentinel_path))
    max_age = _int_from_env("LLM_HEALTH_MAX_AGE", llm_cfg.max_age)

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

    threads = _int_from_env("LLM_N_THREADS", llm_cfg.threads)
    ctx_size = _int_from_env("LLM_CTX_SIZE", llm_cfg.ctx_size)

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

# TODO: Remaining modules still require What/Why/How documentation.

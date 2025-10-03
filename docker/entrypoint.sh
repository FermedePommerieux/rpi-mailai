#!/bin/sh
set -euo pipefail

log() {
    printf '[mailai-entrypoint] %s\n' "$*" >&2
}

fail() {
    log "fatal: $*"
    exit 1
}

require_dir() {
    dir=$1
    mode=$2
    if [ ! -d "$dir" ]; then
        fail "required directory $dir is missing"
    fi
    if [ ! -r "$dir" ]; then
        fail "required directory $dir must be readable"
    fi
    case "$mode" in
        rw)
            if [ ! -w "$dir" ]; then
                fail "required directory $dir must be writable"
            fi
            ;;
        ro)
            if [ -w "$dir" ]; then
                fail "directory $dir must be mounted read-only"
            fi
            ;;
        *)
            fail "unknown mode $mode for $dir"
            ;;
    esac
}

check_secret_file() {
    file=$1
    if [ -f "$file" ]; then
        perm=$(stat -c '%a' "$file")
        if [ "$perm" != "600" ]; then
            fail "secret $file must have permissions 0600 (found $perm)"
        fi
    fi
}

check_secrets() {
    base=/etc/mailai
    if [ ! -d "$base" ]; then
        return 0
    fi
    check_secret_file "${MAILAI_ACCOUNTS:-$base/accounts.yaml}"
    find "$base" -maxdepth 1 -type f \
        \( -name '*.key' -o -name '*.pem' -o -name 'pepper' -o -name 'pepper.txt' \) \
        -print0 | while IFS= read -r -d '' secret; do
            check_secret_file "$secret"
        done
}

warmup_llm() {
    log "warming up local LLM"
    if ! python - <<'PY'; then
        fail "LLM warmup failed"
    fi
PY
import json
import os
import signal
import stat
import time
from pathlib import Path

from llama_cpp import Llama

from mailai.config.loader import RuntimeConfigError, get_runtime_config


def _int_from_env(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc


class _TimeoutAlarm:
    def __init__(self, seconds, message):
        self._seconds = int(seconds)
        self._message = message
        self._previous = None

    def __enter__(self):
        if self._seconds <= 0:
            return self

        def _handle(_signum, _frame):
            raise TimeoutError(self._message)

        self._previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle)
        signal.alarm(self._seconds)
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        signal.alarm(0)
        if self._previous is not None:
            signal.signal(signal.SIGALRM, self._previous)
        return False


def _load_settings():
    try:
        runtime = get_runtime_config()
    except RuntimeConfigError as exc:
        raise SystemExit(f"unable to load runtime config: {exc}") from exc

    llm_cfg = runtime.llm
    model_path = os.environ.get("LLM_MODEL_PATH") or llm_cfg.model_path
    if not model_path:
        raise SystemExit("LLM_MODEL_PATH is not set")
    model_file = Path(model_path)
    if not model_file.is_file():
        raise SystemExit(f"model file not found: {model_file}")

    threads = _int_from_env("LLM_N_THREADS", llm_cfg.threads)
    ctx_size = _int_from_env("LLM_CTX_SIZE", llm_cfg.ctx_size)
    sentinel_path = Path(os.environ.get("LLM_HEALTH_SENTINEL") or llm_cfg.sentinel_path)
    load_timeout_s = _int_from_env("LLM_LOAD_TIMEOUT_S", llm_cfg.load_timeout_s)
    completion_timeout_s = _int_from_env(
        "LLM_WARMUP_COMPLETION_TIMEOUT_S", llm_cfg.warmup_completion_timeout_s
    )
    healthcheck_timeout_s = _int_from_env(
        "LLM_HEALTHCHECK_TIMEOUT_S", llm_cfg.healthcheck_timeout_s
    )

    return {
        "model_file": model_file,
        "threads": threads,
        "ctx_size": ctx_size,
        "sentinel_path": sentinel_path,
        "load_timeout_s": load_timeout_s,
        "completion_timeout_s": completion_timeout_s,
        "healthcheck_timeout_s": healthcheck_timeout_s,
    }

def _run_warmup(settings):
    attempts = 3
    backoff = 1.0
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            model_file = settings["model_file"]
            threads = settings["threads"]
            ctx_size = settings["ctx_size"]
            load_timeout_s = settings["load_timeout_s"]
            completion_timeout_s = settings["completion_timeout_s"]

            llm = None
            result = None
            load_started = time.monotonic()
            with _TimeoutAlarm(load_timeout_s, f"LLM load timed out after {load_timeout_s}s"):
                llm = Llama(
                    model_path=str(model_file),
                    n_ctx=ctx_size,
                    n_threads=threads,
                    logits_all=False,
                    embedding=False,
                    verbose=False,
                )
            load_duration = time.monotonic() - load_started

            try:
                completion_started = time.monotonic()
                with _TimeoutAlarm(
                    completion_timeout_s,
                    f"LLM warmup completion timed out after {completion_timeout_s}s",
                ):
                    result = llm(
                        "ok?",
                        max_tokens=4,
                        temperature=0.0,
                        top_p=1.0,
                        repeat_penalty=1.0,
                        return_dict=True,
                    )
                completion_duration = time.monotonic() - completion_started
            finally:
                if llm is not None:
                    del llm

            choices = result.get("choices") if isinstance(result, dict) else None
            if not choices:
                raise RuntimeError("warmup returned no choices")

            first_choice = choices[0]
            text = first_choice.get("text") if isinstance(first_choice, dict) else None

            print(
                json.dumps(
                    {
                        "event": "llm_warmup_attempt",
                        "attempt": attempt,
                        "status": "success",
                        "load_duration_s": load_duration,
                        "completion_duration_s": completion_duration,
                    }
                ),
                flush=True,
            )

            return {
                **settings,
                "response": text,
                "attempt": attempt,
                "load_duration_s": load_duration,
                "completion_duration_s": completion_duration,
            }
        except TimeoutError as exc:
            last_error = ("timeout", str(exc))
            print(
                json.dumps(
                    {
                        "event": "llm_warmup_attempt",
                        "attempt": attempt,
                        "status": "timeout",
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            last_error = ("error", str(exc))
            print(
                json.dumps(
                    {
                        "event": "llm_warmup_attempt",
                        "attempt": attempt,
                        "status": "error",
                        "error": str(exc),
                    }
                ),
                flush=True,
            )

        if attempt < attempts:
            time.sleep(backoff)
            backoff = min(backoff * 2, 4.0)

    if last_error is None:
        raise SystemExit("LLM warmup failed")
    reason, message = last_error
    if reason == "timeout":
        raise SystemExit(message)
    raise SystemExit(f"LLM warmup failed after {attempts} attempts: {message}")


def main():
    settings = _load_settings()
    result = _run_warmup(settings)

    sentinel_path = result["sentinel_path"]
    if not isinstance(sentinel_path, Path):
        sentinel_path = Path(str(sentinel_path))
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_path": str(result["model_file"]),
        "threads": result["threads"],
        "ctx_size": result["ctx_size"],
        "completed_at": time.time(),
        "response": result.get("response"),
        "load_duration_s": result.get("load_duration_s"),
        "completion_duration_s": result.get("completion_duration_s"),
        "healthcheck_timeout_s": result.get("healthcheck_timeout_s"),
    }
    sentinel_path.write_text(json.dumps(payload, ensure_ascii=False))
    sentinel_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise SystemExit(str(exc)) from exc
PY
    log "LLM warmup succeeded"
}

MAILAI_VALIDATE=${MAILAI_VALIDATE:-0}
: "${LLM_MODEL_PATH:=}"
: "${LLM_N_THREADS:=}"
: "${LLM_CTX_SIZE:=}"
: "${LLM_HEALTH_SENTINEL:=}"
export LLM_MODEL_PATH LLM_N_THREADS LLM_CTX_SIZE MAILAI_VALIDATE
export LLM_HEALTH_SENTINEL

require_dir /etc/mailai ro
require_dir /var/lib/mailai rw
require_dir /models ro

check_secrets

if [ -n "$LLM_MODEL_PATH" ] && [ ! -f "$LLM_MODEL_PATH" ]; then
    fail "LLM model file $LLM_MODEL_PATH not found"
fi

warmup_llm

if [ "$MAILAI_VALIDATE" = "1" ]; then
    log "running mailai diagnostics"
    if ! mailai diag --redact; then
        fail "mailai diagnostics failed"
    fi
fi

exec "$@"

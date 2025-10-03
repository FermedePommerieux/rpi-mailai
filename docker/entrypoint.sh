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
    if ! timeout 5 python - <<'PY'; then
        fail "LLM warmup failed"
    fi
PY
import os
from pathlib import Path

from llama_cpp import Llama

model_path = os.environ.get("LLM_MODEL_PATH")
if not model_path:
    raise SystemExit("LLM_MODEL_PATH is not set")
model_file = Path(model_path)
if not model_file.is_file():
    raise SystemExit(f"model file not found: {model_file}")

threads = int(os.environ.get("LLM_N_THREADS", "3") or 3)
ctx_size = int(os.environ.get("LLM_CTX_SIZE", "2048") or 2048)

llm = Llama(
    model_path=str(model_file),
    n_ctx=ctx_size,
    n_threads=threads,
    logits_all=False,
    embedding=False,
    verbose=False,
)
try:
    result = llm(
        "ok?",
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        repeat_penalty=1.0,
        return_dict=True,
    )
finally:
    del llm

choices = result.get("choices") if isinstance(result, dict) else None
if not choices:
    raise SystemExit("warmup returned no choices")
PY
    log "LLM warmup succeeded"
}

MAILAI_VALIDATE=${MAILAI_VALIDATE:-0}
LLM_MODEL_PATH=${LLM_MODEL_PATH:-}
LLM_N_THREADS=${LLM_N_THREADS:-3}
LLM_CTX_SIZE=${LLM_CTX_SIZE:-2048}
export LLM_MODEL_PATH LLM_N_THREADS LLM_CTX_SIZE MAILAI_VALIDATE

require_dir /etc/mailai ro
require_dir /var/lib/mailai rw
require_dir /models ro

check_secrets

if [ -z "$LLM_MODEL_PATH" ]; then
    fail "LLM_MODEL_PATH is required"
fi
if [ ! -f "$LLM_MODEL_PATH" ]; then
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

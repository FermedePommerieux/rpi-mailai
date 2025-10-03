#!/bin/sh
set -eu
# shellcheck disable=SC3040
if set -o pipefail 2>/dev/null; then
    :
fi

log() {
    printf '%s\n' "$*" >&2
}

fail() {
    log "[fatal] $*"
    exit 1
}

check_directory() {
    dir_path=$1
    mode=$2
    if [ ! -d "$dir_path" ]; then
        fail "required directory $dir_path is missing"
    fi
    if [ ! -r "$dir_path" ]; then
        fail "required directory $dir_path must be readable"
    fi
    if [ "$mode" = "rw" ] && [ ! -w "$dir_path" ]; then
        fail "required directory $dir_path must be writable"
    fi
    if [ "$mode" = "ro" ] && [ -w "$dir_path" ]; then
        fail "directory $dir_path must be mounted read-only"
    fi
}

check_secret_permissions() {
    base_dir=$1
    if [ ! -d "$base_dir" ]; then
        return 0
    fi
    find "$base_dir" -maxdepth 1 -type f \
        \( -iname '*secret*' -o -iname '*password*' -o -iname '*token*' -o -iname '*key*' -o -iname '*pepper*' \) \
        -exec sh -c '
            for file do
                perm=$(stat -c "%a" "$file")
                if [ "$perm" != "600" ]; then
                    printf "[fatal] secret %s must have permissions 0600 (found %s)\n" "$file" "$perm" >&2
                    exit 1
                fi
            done
        ' sh {} +
}

query_llm() {
    python - "$LLM_BASE_URL" "$LLM_HEALTH_MODEL" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base = (sys.argv[1] or "http://llama-server:8080/v1").rstrip("/")
model_hint = sys.argv[2]

def resolve_model():
    if model_hint:
        return model_hint
    try:
        with urllib.request.urlopen(base + "/models", timeout=5) as response:
            payload = json.load(response)
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"unable to query LLM models: {exc}")
    data = payload.get("data")
    if not data:
        raise SystemExit("no models returned by LLM")
    first = data[0]
    model_id = first.get("id")
    if not model_id:
        raise SystemExit("LLM response missing model id")
    return model_id

model_id = resolve_model()
body = json.dumps({
    "model": model_id,
    "messages": [{"role": "user", "content": "ok?"}],
    "max_tokens": 4,
}).encode()
request = urllib.request.Request(
    base + "/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.status != 200:
            raise SystemExit(f"unexpected status {response.status}")
        json.load(response)
except urllib.error.URLError as exc:
    raise SystemExit(f"LLM request failed: {exc}")
PY
}

check_llm() {
    retries=3
    attempt=1
    while [ "$attempt" -le "$retries" ]; do
        if query_llm >/dev/null 2>&1; then
            return 0
        fi
        log "waiting for LLM to become ready (attempt $attempt/$retries)"
        attempt=$((attempt + 1))
        sleep 2
    done
    query_llm
}

LLM_BASE_URL=${LLM_BASE_URL:-http://llama-server:8080/v1}
LLM_HEALTH_MODEL=${LLM_HEALTH_MODEL:-}
export LLM_BASE_URL
export LLM_HEALTH_MODEL

check_directory /etc/mailai ro
check_directory /var/lib/mailai rw
check_directory /models ro

check_secret_permissions /etc/mailai

if ! check_llm; then
    fail "LLM service is not reachable"
fi

if [ "${MAILAI_VALIDATE:-0}" = "1" ]; then
    log "running mailai diagnostics"
    mailai diag --redact
fi

exec "$@"

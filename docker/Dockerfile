# syntax=docker/dockerfile:1.6
FROM --platform=linux/arm64 python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libffi-dev \
       libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

COPY mailai /tmp/build/mailai

RUN pip wheel --wheel-dir /tmp/wheels /tmp/build/mailai

FROM --platform=linux/arm64 python:3.11-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/mailai/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/mailai \
    && /opt/mailai/bin/pip install --upgrade pip wheel setuptools

COPY --from=builder /tmp/wheels /tmp/wheels

RUN /opt/mailai/bin/pip install /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

RUN groupadd --system --gid 10001 mailai \
    && useradd --system --gid 10001 --uid 10001 --create-home --home-dir /var/lib/mailai mailai

RUN mkdir -p /etc/mailai /models \
    && chown -R mailai:mailai /var/lib/mailai \
    && chown mailai:mailai /etc/mailai /models

COPY docker/entrypoint.sh /opt/mailai/bin/entrypoint.sh

RUN chmod 0550 /opt/mailai/bin/entrypoint.sh \
    && chown mailai:mailai /opt/mailai/bin/entrypoint.sh

USER mailai:mailai

WORKDIR /var/lib/mailai

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/mailai/bin/entrypoint.sh"]

CMD ["mailai", "watch"]

HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=30s CMD python - <<'PY'
import json
import os
import urllib.error
import urllib.request

base = os.environ.get("LLM_BASE_URL", "http://llama-server:8080/v1").rstrip("/")
model_hint = os.environ.get("LLM_HEALTH_MODEL", "")

def get_model_id():
    if model_hint:
        return model_hint
    try:
        with urllib.request.urlopen(base + "/models", timeout=5) as resp:
            payload = json.load(resp)
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"failed to query models: {exc}")
    data = payload.get("data")
    if not data:
        raise SystemExit("no models reported by LLM")
    first = data[0]
    model_id = first.get("id")
    if not model_id:
        raise SystemExit("model id missing in LLM response")
    return model_id

model_id = get_model_id()
body = json.dumps({
    "model": model_id,
    "messages": [{"role": "user", "content": "ok?"}],
    "max_tokens": 4,
}).encode()
req = urllib.request.Request(
    base + "/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        if resp.status != 200:
            raise SystemExit(f"unexpected status: {resp.status}")
        json.load(resp)
except urllib.error.URLError as exc:
    raise SystemExit(f"healthcheck failed: {exc}")
PY

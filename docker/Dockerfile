# syntax=docker/dockerfile:1.6

FROM --platform=linux/arm64 python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        make \
        g++ \
        python3-dev \
        libopenblas-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip build wheel

WORKDIR /build

COPY requirements.txt ./
COPY mailai ./mailai

RUN pip wheel --wheel-dir /wheels -r requirements.txt
RUN pip wheel --wheel-dir /wheels llama-cpp-python

FROM --platform=linux/arm64 python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        tzdata \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

ENV LLM_N_THREADS=3 \
    LLM_CTX_SIZE=2048 \
    MAILAI_ACCOUNTS=/etc/mailai/accounts.yaml \
    MAILAI_LOG_LEVEL=INFO \
    MAILAI_IMAP_TIMEOUT=30 \
    TZ=Europe/Paris

RUN groupadd --system --gid 10001 mailai \
    && useradd --system --gid 10001 --uid 10001 --create-home --home-dir /var/lib/mailai mailai

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

COPY --chown=mailai:mailai mailai /app/mailai
COPY --chown=mailai:mailai README.md /app/README.md
COPY --chown=mailai:mailai docker/entrypoint.sh /entrypoint.sh

RUN chmod 0755 /entrypoint.sh

USER mailai:mailai

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
CMD ["mailai", "watch"]

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD python -m mailai.health_llm

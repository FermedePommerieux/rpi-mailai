# syntax=docker/dockerfile:1.7

FROM --platform=$BUILDPLATFORM python:3.11-bookworm AS base
WORKDIR /app
COPY app/ /app/

# Harden pip a bit (optional)
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=10 PIP_INDEX_URL=${PIP_INDEX_URL}

RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir \
      imapclient mail-parser beautifulsoup4 lxml pyyaml joblib numpy \
      sentence-transformers scikit-learn

# Warm up small encoder
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
PY

FROM python:3.11-bookworm
WORKDIR /app
COPY --from=base /usr/local /usr/local
COPY --from=base /app /app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /app/mailai.py /app/bin/accountctl

VOLUME ["/config", "/data"]

ENV APP_CONFIG=/config/config.yml \
    DATA_DIR=/data \
    POLL_EVERY=600 \
    RETRAIN_TIME=03:10:00

HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
  CMD pgrep -f "python /app/mailai.py" > /dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]

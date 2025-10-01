#!/usr/bin/env bash
set -euo pipefail

APP_CONFIG="${APP_CONFIG:-/config/config.yml}"
DATA_DIR="${DATA_DIR:-/data}"
POLL_EVERY="${POLL_EVERY:-600}"
RETRAIN_TIME="${RETRAIN_TIME:-03:10:00}"

echo "[entrypoint] Config: $APP_CONFIG"
echo "[entrypoint] DATA_DIR: $DATA_DIR, POLL_EVERY: $POLL_EVERY, RETRAIN_TIME: $RETRAIN_TIME"

mkdir -p "$DATA_DIR/db" "$DATA_DIR/models"
if [ ! -f "$APP_CONFIG" ]; then
  echo "[entrypoint] Creating default config..."
  mkdir -p "$(dirname "$APP_CONFIG")"
  cat > "$APP_CONFIG" <<YAML
data_dir: $DATA_DIR
model:
  encoder: sentence-transformers/all-MiniLM-L6-v2
  enable_heavy_encoder_on_next_retrain: false
  heavy_encoder: intfloat/e5-base-v2
  enable_cross_encoder: false
  cross_encoder_name: cross-encoder/ms-marco-MiniLM-L-6-v2
  ambiguous_lower: 0.60
  ambiguous_upper: 0.85
  min_auto_move_confidence: 0.85
  min_spam_confidence: 0.97
  quarantine_folder: "AI/A-REVIEW"
scheduler:
  poll_every_seconds: $POLL_EVERY
  nightly_retrain_hour: "$(echo "$RETRAIN_TIME" | cut -d: -f1-2)"
accounts: []
YAML
fi

run_loop() { python /app/mailai.py --config "$APP_CONFIG" loop || true; }
run_retrain() { echo "[entrypoint] Nightly retrain..."; python /app/mailai.py --config "$APP_CONFIG" retrain || true; }

next_retrain_epoch() {
  local hh=${RETRAIN_TIME:0:2}; local mm=${RETRAIN_TIME:3:2}; local ss=${RETRAIN_TIME:6:2}
  local today="$(date +%Y-%m-%d)"; local target="$today $hh:$mm:$ss"
  local now_epoch=$(date +%s); local tgt_epoch=$(date -d "$target" +%s)
  if [ "$tgt_epoch" -le "$now_epoch" ]; then
    tgt_epoch=$(date -d "$today + 1 day ${hh}:${mm}:${ss}" +%s)
  fi
  echo "$tgt_epoch"
}

NEXT_RETRAIN=$(next_retrain_epoch)
echo "[entrypoint] Next retrain at: $(date -d @$NEXT_RETRAIN)"

while true; do
  run_loop
  now=$(date +%s)
  if [ "$now" -ge "$NEXT_RETRAIN" ]; then
    run_retrain
    NEXT_RETRAIN=$(next_retrain_epoch)
    echo "[entrypoint] Next retrain at: $(date -d @$NEXT_RETRAIN)"
  fi
  sleep "$POLL_EVERY"
done

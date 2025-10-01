#!/usr/bin/env bash
set -euo pipefail
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJ_DIR"
echo "===> Building image with host network (DNS-friendly)"
docker build --network=host -t rpi-mailai:latest .

echo "===> Starting service via docker compose"
docker compose up -d
docker ps --filter "name=rpi-mailai"
echo "Done. Logs: docker logs -f rpi-mailai"

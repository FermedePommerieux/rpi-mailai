# RPi-MailAI

Self-hosted IMAP mail triage with:
- Multi-account (IMAP)
- Cold mode → per-account Auto-move & optional auto-spam
- Nightly retraining from your corrections
- Light encoder by default; optional heavier models & cross-encoder
- Single Docker service + CLI

## Quick start

```bash
docker build --network=host -t rpi-mailai:latest .
docker compose up -d
docker logs -f rpi-mailai

#Add an account (folders normalized/created if needed):
```bash
docker exec -it rpi-mailai /app/bin/accountctl add pro \
  --user you@example.com \
  --host mail.example.com --port 993 \
  --spam "Junk" \
  --newsletter "Promotions" \
  --basse "SocialNetwork" \
  --important "Important" \
  --projet "Projects" \
  --quarantine "AI/A-REVIEW" \
  --auto-move
#Manual pipeline:
```bash
docker exec -it rpi-mailai python /app/mailai.py --config /config/config.yml snapshot
docker exec -it rpi-mailai python /app/mailai.py --config /config/config.yml predict
#Heavier models
```bash
sed -i 's/enable_cross_encoder: false/enable_cross_encoder: true/' config/config.yml
sed -i 's/enable_heavy_encoder_on_next_retrain: false/enable_heavy_encoder_on_next_retrain: true/' config/config.yml
docker restart rpi-mailai
#DNS tips
If container DNS fails, either:
use network_mode: host in docker-compose.yml, or
add dns: (1.1.1.1, 8.8.8.8) under the service, or
```bash
set /etc/docker/daemon.json with "dns": ["1.1.1.1","8.8.8.8"] and restart Docker.
#License
MIT
MD

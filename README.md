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
```
## Add an account (folders normalized/created if needed):
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
```

### Mail type configuration per account

Each account now ships with a JSON configuration stored under `/config/account_types/<account>.json` (override the directory with `MAIL_TYPES_DIR`).
The file lists the types of messages the assistant is allowed to triage for this account. When only one rule is enabled, only that mail type will be processed.

Sample content created by `accountctl`:

```json
{
  "account": "pro",
  "types": [
    {
      "key": "important",
      "label": "Important",
      "enabled": true,
      "target_folder": "INBOX/Important",
      "prompt": "Classer dans le dossier Important tout message nécessitant une action rapide, lié à des responsables, des clients ou des demandes urgentes."
    },
    {
      "key": "factures",
      "label": "Factures à traiter",
      "enabled": false,
      "target_folder": "INBOX/Projects",
      "prompt": "Repérer les mails contenant une facture à traiter : expéditeur de type fournisseur, mention explicite de facture ou facturation et présence d'une pièce jointe PDF."
    }
  ]
}
```

Toggle `enabled` to activate/deactivate a rule, adjust `target_folder`, and customise the French “prompt” to describe how the sorter should recognise the type. Generic prompts are provided for common categories (important, newsletters, projects, faible priorité, quarantaine, factures) and auto-move only occurs for enabled rules.

## Manual pipeline:
```bash
docker exec -it rpi-mailai python /app/mailai.py --config /config/config.yml snapshot
docker exec -it rpi-mailai python /app/mailai.py --config /config/config.yml predict
```
## Heavier models
```bash
sed -i 's/enable_cross_encoder: false/enable_cross_encoder: true/' config/config.yml
sed -i 's/enable_heavy_encoder_on_next_retrain: false/enable_heavy_encoder_on_next_retrain: true/' config/config.yml
docker restart rpi-mailai
```
## DNS tips
If container DNS fails, either:
use network_mode: host in docker-compose.yml, or
add dns: (1.1.1.1, 8.8.8.8) under the service, or
```bash
set /etc/docker/daemon.json with "dns": ["1.1.1.1","8.8.8.8"] and restart Docker.
```
#License
MIT
MD

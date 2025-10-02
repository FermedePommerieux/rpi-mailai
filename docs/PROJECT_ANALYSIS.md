# RPi-MailAI Project Review

## Overview
RPi-MailAI is a self-hosted IMAP triage service designed to run as a single Docker container. The Docker image bundles a Python 3.11 application that continuously ingests messages from configured accounts, stores message snapshots in SQLite, and trains a scikit-learn classifier backed by SentenceTransformers embeddings. Runtime behavior is orchestrated by `entrypoint.sh`, which creates a default configuration, launches the main loop, and schedules nightly retraining runs based on environment variables exposed in `docker-compose.yml`. Users manage accounts through the CLI tool `app/bin/accountctl`, which provisions IMAP folders and stores credentials in the `/config/secrets` directory. The main workflow in `app/mailai.py` implements four pipeline steps: `snapshot` (ingest), `retrain` (train a logistic regression on labeled messages), `predict` (classify unlabeled rows and optionally move them), and `loop` (polling orchestrator).

## Functional Flow
1. **Container lifecycle** – Docker builds install IMAP and ML dependencies and pre-download the default MiniLM encoder. At runtime the entrypoint ensures configuration and data directories exist, then calls `python /app/mailai.py --config ... loop` in a perpetual loop, interleaving nightly retrain jobs.
2. **Account provisioning** – `accountctl` normalizes folder names according to the IMAP server's namespace, persists credentials, and augments the YAML config with account metadata used by the main app.
3. **Message ingestion** – `snapshot` logs into each account, enumerates every UID returned by `srv.search()`, fetches full message bodies, parses them with `mailparser`, and records subjects and plain-text bodies into SQLite with a uniqueness constraint on `(account,msgid)`.
4. **Model training** – `retrain` loads all labeled rows, derives embeddings through the configured `SentenceTransformer`, trains `LogisticRegression`, and saves both model and encoder via `joblib` to `/data/models` while recording metadata in the `models` table.
5. **Prediction & actions** – `predict` reloads the persisted encoder and classifier, produces probabilities for unlabeled records, writes predictions back into SQLite, and, when `auto_move` is enabled, attempts to move messages to target folders via IMAP.

## Key Issues Identified
1. **Auto-move uses database IDs instead of IMAP UIDs** – The ingestion pipeline never stores the IMAP UID and `predict` passes the SQLite primary key to `srv.move`, so the server receives an invalid sequence number. Moves will therefore fail silently for all accounts. Storing the UID during `snapshot` and persisting it alongside each message is required.
2. **SentenceTransformer persistence via joblib** – Dumping the encoder instance with `joblib` can serialize device-specific state and bloats the models directory. SentenceTransformers provide `.save()`/`.from_pretrained()` helpers and rely on on-disk configuration; reloading with `SentenceTransformer(model_name_or_path)` would be more robust and keep checkpoints compatible across machines.
3. **Message-ID decoding robustness** – `snapshot` decodes ENVELOPE fields with the default codec, which raises `UnicodeDecodeError` for non-ASCII data. Falling back to UTF-8 with error handling (or using `mailparser` metadata) would prevent ingestion crashes.
4. **Plaintext secret management** – Account passwords are stored as world-readable YAML paths to plaintext files under `/config/secrets`. Tightening file permissions (already 0600) and documenting secret storage risks is advisable, but integrating with a secrets manager would be safer for multi-user hosts.
5. **Full-body fetch on every poll** – Re-fetching entire message bodies for all messages during each snapshot run is expensive and bandwidth-heavy. Tracking `uidvalidity` and `uid`s to only fetch new mail (and storing header-only data when appropriate) would dramatically reduce load.
6. **Lack of config under version control** – The repository ships an empty `/config` directory and relies on runtime generation, which hinders reproducible defaults in development environments. Supplying a sample config (without secrets) would improve onboarding and lint/test setups.
7. **Limited error handling for IMAP operations** – Failures in `snapshot` (e.g., connection drops, parsing errors) are not isolated per message; one exception can break the entire account iteration. Wrapping fetch/parse loops and logging problematic messages would make the service more resilient.

## Suggested Next Steps
- Extend the schema to capture `uid` and `folder` at ingest time and update the auto-move logic to operate on correct server identifiers.
- Refactor model persistence to rely on SentenceTransformer’s native save/load utilities and guard against missing artifacts before attempting predictions. Document that large checkpoints can be redirected to network-attached storage (NAS) so resource-constrained hosts like Raspberry Pi 4B avoid filling local media.
- Harden text decoding and parsing paths, including explicit charset handling for ENVELOPE fields and fallback strategies when `mailparser` fails.
- Introduce incremental snapshot logic (track the highest seen UID per account) to avoid repeatedly downloading unchanged messages.
- Provide documented configuration samples and possibly a CLI command to bootstrap example accounts without secrets for testing.
- Consider optional integration with cross-encoders and heavy encoders described in the config once persistence/reload issues are resolved.

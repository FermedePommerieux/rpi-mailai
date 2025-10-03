# MailAI Monorepo Wrapper

This repository hosts the MailAI offline email triage agent. The Python package
lives under the [`mailai/`](mailai/) directory where you will find the project
source, documentation, and packaging metadata. Refer to
[`mailai/README.md`](mailai/README.md) for installation and usage instructions.

## Docker (ARM64 + mandatory in-container LLM)

MailAI ships with an ARM64-only container workflow that embeds
`llama-cpp-python` inside the MailAI image. The project targets Raspberry Pi 4B
class hardware and does **not** provide any "LLM disabled" fallback path. The
container refuses to start unless it can load the local GGUF model and produce a
short completion during the boot sequence.

### Prerequisites

1. Use a 64-bit host capable of running `linux/arm64` containers (for example a
   Raspberry Pi 4B running a 64-bit OS).
2. Download a quantised GGUF model compatible with `llama.cpp` and store it as
   `./models/model.gguf` on the host.
3. Provide configuration secrets in `./config/` and persistent state in
   `./data/`. Secrets inside `./config/` must be owned by your host user and
   have permissions `0600` before starting the container.

### Quick start

```bash
mkdir -p models data config
# Place your GGUF model at models/model.gguf
cp /path/to/your-model.gguf models/model.gguf

# Launch MailAI (LLM warmup will run automatically)
docker compose -f compose/docker-compose.yml up -d
```

The supplied `Makefile` exposes helper targets for the hardened runtime:

```bash
# Build the linux/arm64 image with Buildx
make buildx-arm64

# Build and push the linux/arm64 image to a registry
make push-arm64

# Start the service via Docker Compose (single container)
make run
```

`docker compose` uses the [`compose/docker-compose.yml`](compose/docker-compose.yml)
definition which declares a single `mailai` service running the in-process LLM.
The container mounts the following volumes:

- `./data` → `/var/lib/mailai` (read-write persistent state)
- `./config` → `/etc/mailai` (read-only secrets and configuration)
- `./models` → `/models` (read-only GGUF model directory)

### Required environment variables

The container sets sensible defaults for most parameters. Provide the GGUF
model via `LLM_MODEL_PATH` and override other variables as needed:

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_MODEL_PATH` | _(required)_ | Absolute path inside the container to the GGUF model (defaults to `/models/model.gguf` in Compose). |
| `LLM_N_THREADS` | `3` | Number of CPU threads used by `llama-cpp-python` (tuned for Raspberry Pi 4B). |
| `LLM_CTX_SIZE` | `2048` | Context window size for the embedded model. |
| `MAILAI_ACCOUNTS` | `/etc/mailai/accounts.yaml` | Location of the IMAP account definitions. |
| `MAILAI_LOG_LEVEL` | `INFO` | Log level emitted to stdout (JSON). |
| `MAILAI_IMAP_TIMEOUT` | `30` | IMAP network timeout in seconds. |
| `TZ` | `Europe/Paris` | Time zone for the container. |

### Runtime guarantees

- The entrypoint validates required volumes, enforces `0600` permissions on
  secrets, and fails fast when directories are missing or misconfigured.
- A warmup routine loads the GGUF model via `llama-cpp-python`, performs a tiny
  completion (`"ok?"` with four tokens), and aborts the container if it cannot
  complete within five seconds.
- The healthcheck reuses the in-process integration and fails whenever the model
  cannot be loaded or respond, ensuring orchestrators detect LLM regressions.
- The root filesystem is kept read-only at runtime; only `/var/lib/mailai`
  remains writable for stateful data. Logs are emitted to stdout in JSON format
  without exposing secrets or email content.

### Intent enrichment signals

MailAI now records additional closed-vocabulary metadata describing the intent
and tone of every processed message. The enrichment stage stores only
identifiers and bounded scores (0..3) covering:

- `intent` – coarse purpose such as `transactional`, `marketing`, or
  `fraud_solicitation`.
- `speech_acts` and `persuasion` – sets of whitelisted tactics detected via
  heuristics and the local LLM.
- `urgency_score`, `insistence_score`, `commercial_pressure`, and
  `scam_singularity` – bounded integers driving routing/quarantine decisions.
- `suspicion_flags` – closed vocabulary including `link_mismatch`,
  `attachment_push`, and other high-signal heuristics.

The optional enrichment pipeline is enabled through the new
`intent_features` section in `config.cfg` (see `examples/config.yaml`). Messages
exceeding the configured `scam_singularity_quarantine` threshold are isolated in
the quarantine mailbox without ever persisting plaintext content.

If the GGUF model is missing, renamed, or incompatible, MailAI terminates with a
non-zero exit status. Restore the model to `/models/model.gguf` and restart the
container to resume operation.

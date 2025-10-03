# MailAI Monorepo Wrapper

This repository hosts the MailAI offline email triage agent. The Python package
lives under the [`mailai/`](mailai/) directory where you will find the project
source, documentation, and packaging metadata. Refer to
[`mailai/README.md`](mailai/README.md) for installation and usage instructions.

## Docker (ARM64 + mandatory LLM)

MailAI ships with an ARM64-only container workflow that requires a running
local LLM at all times. The project targets Raspberry Pi 4B class hardware and
does **not** provide any "LLM disabled" fallback path.

### Prerequisites

1. Use a 64-bit host capable of running `linux/arm64` containers (for example a
   Raspberry Pi 4B running a 64-bit OS).
2. Download at least one GGUF model (e.g.
   `phi-3-mini-q4_0.gguf`) into `./models/`.
3. Provide configuration secrets in `./config/` and persistent state in
   `./data/`. Secrets inside `./config/` must be owned by your host user and
   have permissions `0600` before starting the containers.

### Building and running

The supplied `Makefile` exposes helper targets for the hardened runtime:

```bash
# Build the linux/arm64 image locally
make build

# Build with Buildx (linux/arm64) or push to a registry
make buildx-arm64
make push-arm64

# Launch the LLM server and MailAI runtime together
make run
```

`docker compose` uses the [`compose/docker-compose.yml`](compose/docker-compose.yml)
definition which declares two services:

- `llama-server` – based on `ghcr.io/ggerganov/llama.cpp:server`, mounting the
  read-only `./models` directory and exposing the OpenAI-compatible HTTP API.
- `mailai` – the MailAI runtime image that depends on `llama-server` and reads
  the model via `LLM_BASE_URL=http://llama-server:8080/v1`. Set
  `LLM_HEALTH_MODEL` to the model identifier exposed by the server (defaults to
  `phi-3-mini-q4_0.gguf` in the provided Compose file).

The MailAI entrypoint validates the mounted directories, enforces `0600`
permissions on secrets, probes the LLM with a short completion request, and
aborts the container if the LLM is unreachable. The container healthcheck also
fails whenever the LLM endpoint stops responding, so stopping `llama-server`
will mark MailAI as unhealthy and prevent successful restarts until the model
is back online.

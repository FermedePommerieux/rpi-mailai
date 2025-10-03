# MailAI

MailAI is an offline-first email triage agent designed for Raspberry Pi 4B devices.
It consumes configuration from IMAP mailboxes, evaluates deterministic rules,
learns from user gestures via a lightweight local language model, and updates a
status message after every pass. The implementation focuses on privacy by design
and never persists cleartext email data.

## Features

- Strict YAML configuration (`rules.yaml`) and status reporting (`status.yaml`).
- Deterministic rule engine with idempotent IMAP actions.
- Local learning pipeline using hashed features and optional local embeddings.
- Offline LLM integration via `llama.cpp` bindings for semantic inference and
  rule synthesis.
- Encrypted feature store with peppered hashing and redact-only logging.
- CLI with `once`, `watch`, `learn-now`, and `diag` subcommands.

## Installation (Raspberry Pi 4B)

1. Install system dependencies:
   ```bash
   sudo apt update
   sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential \
       libffi-dev libssl-dev libsodium-dev
   ```
2. Create a virtual environment and install MailAI:
   ```bash
   python3.11 -m venv ~/mailai-venv
   source ~/mailai-venv/bin/activate
   pip install --upgrade pip
   pip install .
   ```
3. Provision a local `llama.cpp` build and place your quantised model
   (e.g. `qwen2.5-3b-instruct-q4_0.gguf`) under `/var/lib/mailai/models/`.

## Usage

- `mailai once`: run a single evaluation pass.
- `mailai watch`: continuously watch the mailbox according to the configured
  interval.
- `mailai learn-now`: trigger the learning pipeline immediately.
- `mailai diag --redact`: emit a redacted diagnostics report.

## Global runtime configuration

MailAI centralises all runtime tunables inside a single `config.cfg` file. The
loader accepts either YAML or JSON and validates the payload against the
[`RuntimeConfig`](mailai/src/mailai/config/schema.py) schema. Typical settings
include IMAP defaults (control namespace, quarantine folder, configuration
subjects), size limits for the control mails, filesystem paths for state and
models, local LLM parameters, and optional feedback mailboxes.

When running inside Docker place `config.cfg` under `/etc/mailai/`. Native
deployments search the current working directory first and fall back to
`/etc/mailai/config.cfg` and `/var/lib/mailai/config.cfg`. A reference document
is available under [`examples/config.cfg`](../examples/config.cfg).

### LLM warm-up and healthcheck timeouts

The `llm` section of `config.cfg` controls how the embedded llama.cpp runtime is
initialised. In addition to the model location and threading parameters, three
timeouts guard the warm-up and healthcheck sequence:

- `load_timeout_s` (default **120s**) – caps how long the GGUF model may take to
  load during start-up before MailAI aborts the attempt.
- `warmup_completion_timeout_s` (default **10s**) – limits the inference window
  for the warm-up completion that verifies the model can answer a trivial
  prompt.
- `healthcheck_timeout_s` (default **5s**) – bounds the execution time for the
  `mailai-health-llm` probe that revalidates the warm-up sentinel.

Increase these values when running larger models or slower storage where the
initial load might exceed the defaults. The sample configuration documents
recommended overrides for Raspberry Pi deployments.

## IMAP YAML Configuration

MailAI stores its configuration and diagnostics inside the IMAP account under a
dedicated control namespace. By default the agent reuses the standard `Drafts`
mailbox that contains two human-readable messages:

- **`MailAI: rules.yaml`** – the authoritative configuration described by the
  [`RulesV2`](mailai/src/mailai/config/schema.py) schema. Every rule must include
  a human-facing `description`, a justification in `why`, and a `source`
  indicating whether the rule was `deterministic` or emitted by the
  `learner`. The agent automatically restores a minimal rule-set if the file is
  missing or becomes corrupted and keeps a mirror copy as
  `MailAI: rules.bak.yaml`.
- **`MailAI: status.yaml`** – the latest diagnostic snapshot following the
  [`StatusV2`](mailai/src/mailai/config/schema.py) schema. It records aggregate
  metrics, privacy checks, and a `proposals` section where learner-generated
  rules are rendered as YAML diffs with an explanation.

Both messages inherit their soft and hard size limits from `config.cfg`. MailAI
attempts to keep the payload below the soft ceiling by truncating verbose
sections such as `notes` and `proposals` while preserving the most relevant
entries. Example documents are provided under
[`examples/rules.yaml`](../examples/rules.yaml) and
[`examples/status.yaml`](../examples/status.yaml). Account bootstrap data for
`accountctl` is illustrated in [`examples/accounts.yaml`](../examples/accounts.yaml).

## Testing

```bash
pip install .[test]
pytest
```

## License

MailAI is released under the MIT License. See [LICENSE](LICENSE).

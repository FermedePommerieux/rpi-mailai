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

## Testing

```bash
pip install .[test]
pytest
```

## License

MailAI is released under the MIT License. See [LICENSE](LICENSE).

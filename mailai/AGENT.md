# MailAI Agent Notes

This codebase is designed around a strict privacy model for offline email
processing. When modifying modules keep the following in mind:

- **Never** introduce logging or persistence of raw email content. Logs must
  remain structured JSON with redacted payloads only.
- Encryption helpers in `core/privacy.py` must default to ChaCha20-Poly1305
  using libsodium bindings. Do not downgrade the cipher without an explicit
  discussion.
- Ensure the rule engine stays idempotent. All IMAP actions must apply the
  `X-MailAI` header guard before performing network operations.
- Unit tests must cover new failure modes. Prefer deterministic fixtures over
  randomised data to ease debugging on Raspberry Pi hardware.
- CLI changes should remain ergonomic and scriptable; avoid interactive prompts
  unless specifically required by a feature flag.

Threat modelling quick-reference:

1. Assume the IMAP server is honest-but-curious. Every temporary buffer should
   be kept in-memory and wiped after use when possible.
2. The feature store is encrypted at rest; validate key rotation paths when
   editing those modules.
3. Differential privacy layers must never leak counts for fewer than five
   samples even after noise injection.

## Operational flows

- `mailai once`: run a single inference pass against the selected account. The
  agent mounts the `Drafts` control mailbox, restores missing configuration if
  required, and emits a fresh `status.yaml` snapshot in dry-run mode by default.
- `mailai watch`: identical to `once` but loops according to the configured
  `schedule.inference_interval_s`. Auto-repair logic for `rules.yaml` is invoked
  on every cycle before processing user messages.
- `mailai learn-now`: execute the training pipeline immediately. The learner may
  populate the `proposals` section of `status.yaml` with candidate rules which
  are disabled by default (`enabled: false`, `source: learner`).

## Gesture interpretation

- **move** gestures are treated as explicit, high-confidence labels. When the
  learner sees repeated manual moves, it synthesises candidate rules with
  `why` explaining the automation intent.
- **delete** gestures feed into heuristics in `core/delete_semantics.py`. Signals
  such as `spam_score_high`, `promotion_sender`, `conversation_closed`, and
  `invite_expired` are mapped to interpretable reasons so the user understands
  why MailAI suggests archival or deletion.

## Rule proposals

Learner-generated rules are serialised into the `proposals` array inside
`status.yaml` as YAML diffs alongside a human-readable justification. Only the
top-N proposals are retained when the status document would exceed the 64 KB
soft limit, and the originals remain disabled until the operator promotes them
into `rules.yaml`.

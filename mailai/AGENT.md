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

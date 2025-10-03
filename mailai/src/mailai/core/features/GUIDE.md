# Core Features Guide

## What
MailAI's `core.features` package transforms raw RFC822 messages into
privacy-preserving feature records. It covers hashed structural features as well
as the optional intent enrichment layer that annotates messages with bounded
scores and closed-vocabulary identifiers.

## Why
Feature extraction sits between sensitive email content and the learning/routing
pipelines. The code must prevent plaintext persistence while still surfacing the
signals needed for classification, quarantine, and audit trails. Centralising
this logic keeps privacy guards auditable and consistent across the system.

## How
- Parse MIME messages through hardened utilities and hash textual content with a
  pepper/salt pair so no raw tokens are stored.
- Compute aggregate statistics (uppercase ratio, exclamation density, keyword
  hits, URL domains) that feed deterministic heuristics.
- Optionally invoke the locally hosted llama.cpp model with a JSON-only prompt,
  validate the response against closed vocabularies, and merge it into the
  feature record.
- Expose conversion helpers that serialise only IDs, flags, and bounded scores.

## Interfaces & Boundaries
- `extract_features` and `hash_text_window` provide the historical hashed
  feature set.
- `build_mail_feature_record` orchestrates hashing plus intent enrichment and
  returns a `MailFeatureRecord`.
- Schema classes (`IntentFeatureSettings`, `IntentLLMSettings`,
  `IntentThresholds`, `IntentFeatures`, `MailFeatureRecord`) define the runtime
  configuration and persisted structure.
- `infer_intent_and_tone` in `intent_extract.py` combines heuristics with the
  local LLM through the `StructuredLLM` protocol.
- Privacy helpers in `mailai.utils.privacy` are mandatory checkpoints before
  writing enriched metadata to disk.

## Invariants
- No plaintext bodies or subjects leave the module; only hashed artefacts and
  aggregated counters are exposed.
- Intent enrichment values must originate from the closed vocabularies declared
  in `schema.py`; violations raise `PrivacyViolation`.
- Bounded scores remain within the 0..3 range.
- Quarantine decisions rely solely on the configured thresholds without storing
  free-form rationale text.

## Known Pitfalls
- Forgetting to wipe or delete intermediate strings can retain plaintext longer
  than expected; always release body variables after deriving aggregates.
- When adding new suspicion flags or intent labels, update both the vocabulary
  tuples in `schema.py` and the validation tests.
- LLM prompts must never include raw message fragments; stick to aggregated
  metrics when constructing prompts.

## Audit Checklist
- [ ] All outputs are hashed IDs, closed-vocabulary labels, or bounded scores.
- [ ] New vocabulary entries are documented and covered by unit tests.
- [ ] Privacy guards (`assert_closed_vocab`, `assert_bounded_scores`) wrap every
      LLM response before persistence.
- [ ] Quarantine thresholds align with `config.schema.IntentFeaturesConfig`.
- [ ] Tests cover heuristic triggers and ensure plaintext is absent from stored
      records.

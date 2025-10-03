"""mailai.core.features.schema

What:
  Define structured data containers and closed vocabularies used by the intent
  enrichment pipeline when annotating email-derived feature records.

Why:
  The enrichment stage must guarantee that only bounded numeric scores or
  identifiers from a documented whitelist are persisted. Centralising the data
  model keeps privacy guards consistent between heuristics, LLM outputs, and the
  downstream feature store.

How:
  - Declare immutable dataclasses describing runtime configuration knobs,
    intermediate parsing artefacts, and the final persisted record.
  - Maintain closed vocabularies for intents, speech acts, persuasion tactics,
    and suspicion flags; helper methods expose these for validation routines.
  - Provide lightweight conversion helpers so callers can serialise the record
    without leaking raw email text.

Interfaces:
  - :class:`IntentFeatureSettings`, :class:`IntentThresholds`, and
    :class:`IntentLLMSettings` consumed by the extractor.
  - :class:`ParsedMailMeta`, :class:`TextStats`, and :class:`UrlInfo` describing
    intermediate metrics.
  - :class:`IntentFeatures` and :class:`MailFeatureRecord` persisted by the
    feature store.

Invariants & Safety:
  - All vocabularies are explicit tuples; attempting to persist unknown values
    must be rejected by privacy guards.
  - Numeric scores are restricted to the 0..3 range to prevent unbounded scale
    leakage.
  - Dataclasses avoid storing raw body text; only hashed artefacts and
    aggregated counters survive beyond the extraction pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple


INTENT_VOCAB: Tuple[str, ...] = (
    "unknown",
    "personal",
    "transactional",
    "operational_alert",
    "marketing",
    "fraud_solicitation",
)
SPEECH_ACT_VOCAB: Tuple[str, ...] = (
    "inform",
    "request",
    "demand",
    "warn",
    "question",
)
PERSUASION_VOCAB: Tuple[str, ...] = (
    "none",
    "urgency",
    "scarcity",
    "authority",
    "fear",
    "social_proof",
)
SUSPICION_FLAG_VOCAB: Tuple[str, ...] = (
    "link_mismatch",
    "attachment_push",
    "urgent_language",
    "caps_excess",
    "llm_reject",
)


class StructuredLLM(Protocol):
    """Protocol describing the narrow JSON-only completion interface.

    What:
      Capture the callable signature used by the intent extractor when
      requesting structured JSON output from the embedded LLM.

    Why:
      The extractor must remain agnostic to the underlying llama.cpp bindings
      while still enforcing timeout and sampling controls passed via keyword
      arguments.

    How:
      Declare a ``structured_completion`` method accepting the prompt and the
      numeric knobs required by the configuration. Implementations may wrap the
      real llama bindings or provide deterministic fallbacks during tests.

    Args:
      prompt: Sanitised prompt describing aggregated email features.
      max_tokens: Hard upper bound for the completion length.
      temperature: Sampling temperature controlling randomness.
      timeout_s: Execution timeout in seconds for the completion.

    Returns:
      Raw JSON string produced by the backend.
    """

    def structured_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> str:
        """Return a JSON completion enforcing keyword-only runtime parameters."""


@dataclass(frozen=True)
class IntentThresholds:
    """Numeric gates that control downstream responses to enrichment scores.

    What:
      Encapsulate the quarantine and warning thresholds applied to the bounded
      scores emitted by the enrichment stage.

    Why:
      Centralising thresholds keeps the feature extractor, storage layer, and
      tests aligned on the same response criteria while avoiding magic numbers
      scattered across modules.

    How:
      Provide an immutable dataclass storing integer cut-offs. Callers may reuse
      the defaults or supply custom limits when instantiating the extractor.

    Attributes:
      scam_singularity_quarantine: Score above which messages are quarantined.
      urgency_warn: Score above which an urgency warning is emitted.
    """

    scam_singularity_quarantine: int = 2
    urgency_warn: int = 3


@dataclass(frozen=True)
class IntentLLMSettings:
    """Runtime knobs passed to the structured LLM completion helper.

    What:
      Store the bounded token budget, deterministic temperature, and timeout
      enforced when requesting JSON responses.

    Why:
      Privacy guarantees require strict limits on completion size and runtime;
      keeping these parameters explicit ensures the extractor cannot silently
      drift away from audited limits.

    How:
      The dataclass keeps values immutable and hashable, simplifying caching or
      reuse across extractions.

    Attributes:
      max_tokens: Maximum tokens to request from the backend.
      temperature: Sampling temperature (0.0 for deterministic output).
      timeout_s: Timeout in seconds for the completion call.
    """

    max_tokens: int = 64
    temperature: float = 0.0
    timeout_s: float = 3.0


@dataclass(frozen=True)
class IntentFeatureSettings:
    """Toggle and configuration container for intent enrichment.

    What:
      Represent the operator-configured enablement flag together with the LLM
      runtime parameters and score thresholds.

    Why:
      Callers need to carry a single object encapsulating the enrichment policy
      when invoking the extractor, avoiding brittle positional arguments.

    How:
      Provide boolean enablement plus nested dataclasses for LLM and threshold
      configuration.

    Attributes:
      enabled: Whether enrichment should be performed.
      llm: Structured LLM runtime parameters.
      thresholds: Numeric cut-offs influencing quarantine decisions.
    """

    enabled: bool = False
    llm: IntentLLMSettings = field(default_factory=IntentLLMSettings)
    thresholds: IntentThresholds = field(default_factory=IntentThresholds)


@dataclass(frozen=True)
class ParsedMailMeta:
    """Sanitised metadata derived from an email message.

    What:
      Capture domain-level sender information and structural indicators extracted
      from headers without retaining raw subject text.

    Why:
      Heuristics rely on reply depth, attachment indicators, and sender domains
      to detect urgency or pressure tactics; this dataclass stores only the
      aggregated values required for those checks.

    How:
      Fields include sender domain, reply/relance counters, attachment flags, and
      an optional callable that proxies structured LLM completions.

    Attributes:
      from_domain: Domain extracted from the ``From`` header.
      reply_depth: Count of ``Re:`` prefixes observed in the subject.
      relance_count: Count of follow-up indicators (e.g. ``Fwd:``, ``FW:``).
      has_attachments: Whether the message declared attachments.
      attachment_pressure: Heuristic flag derived from body text references.
      llm: Optional structured LLM interface for downstream completions.
      llm_settings: Optional runtime knobs passed when invoking the LLM.
    """

    from_domain: str
    reply_depth: int
    relance_count: int
    has_attachments: bool
    attachment_pressure: bool
    llm: Optional[StructuredLLM] = None
    llm_settings: Optional[IntentLLMSettings] = None


@dataclass(frozen=True)
class TextStats:
    """Aggregated counters derived from the normalised text body.

    What:
      Store bounded ratios and keyword counts used to assess urgency,
      insistence, or commercial pressure without preserving the raw body.

    Why:
      Ratio-based heuristics drive the enrichment logic; encapsulating them keeps
      the extractor interface explicit and ensures the data can be wiped after
      use.

    How:
      Record total character count, uppercase ratio, exclamation mark density,
      call-to-action keyword hits, and attachment mention counts.

    Attributes:
      length: Total characters considered when computing ratios.
      caps_ratio: Uppercase-to-alphabetic ratio.
      exclamation_density: Exclamation marks per character.
      call_to_action_score: Keyword hit count for pressure tactics.
      attachment_mentions: Keyword count referencing attachments.
    """

    length: int
    caps_ratio: float
    exclamation_density: float
    call_to_action_score: int
    attachment_mentions: int


@dataclass(frozen=True)
class UrlInfo:
    """Summaries of URLs observed in the message body.

    What:
      Provide lower-cased domain lists for URLs discovered in anchors or plain
      text.

    Why:
      Comparing sender and URL domains is a strong phishing indicator without
      requiring raw URLs to persist.

    How:
      Store deduplicated tuples of target and (optional) display domains.

    Attributes:
      target_domains: Domains extracted from hyperlink targets.
      display_domains: Domains surfaced to the user (e.g. anchor text).
    """

    target_domains: Tuple[str, ...] = ()
    display_domains: Tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentFeatures:
    """Immutable representation of the enrichment outcome.

    What:
      Persist closed-vocabulary identifiers and bounded scores describing the
      inferred intent, tone, and risk posture of the message.

    Why:
      Downstream consumers (rule synthesis, routing) require deterministic field
      names and values to remain privacy-compliant.

    How:
      Store the chosen intent label, associated speech acts, persuasion tactics,
      suspicion flags, and bounded numeric scores.

    Attributes:
      intent: Primary intent identifier.
      speech_acts: Tuple of speech-act identifiers.
      persuasion: Tuple of persuasion tactic identifiers.
      urgency_score: Urgency intensity (0..3).
      insistence_score: Persistence/relance intensity (0..3).
      commercial_pressure: Commercial push intensity (0..3).
      suspicion_flags: Tuple of suspicion flag identifiers.
      scam_singularity: Aggregated scam risk score (0..3).
    """

    intent: str
    speech_acts: Tuple[str, ...]
    persuasion: Tuple[str, ...]
    urgency_score: int
    insistence_score: int
    commercial_pressure: int
    suspicion_flags: Tuple[str, ...]
    scam_singularity: int

    def as_dict(self) -> Dict[str, object]:
        """Return a serialisable dictionary representation of the features."""

        return {
            "intent": self.intent,
            "speech_acts": self.speech_acts,
            "persuasion": self.persuasion,
            "urgency_score": self.urgency_score,
            "insistence_score": self.insistence_score,
            "commercial_pressure": self.commercial_pressure,
            "suspicion_flags": self.suspicion_flags,
            "scam_singularity": self.scam_singularity,
        }


@dataclass(frozen=True)
class MailFeatureRecord:
    """Container bundling hashed features with enrichment metadata.

    What:
      Capture the privacy-preserving feature dictionary, optional intent
      enrichment, and derived policy actions such as quarantine decisions.

    Why:
      Centralising the persisted artefacts makes it straightforward to assert
      privacy invariants before writing to disk or training data structures.

    How:
      Store the hashed feature mapping, the optional :class:`IntentFeatures`, and
      a boolean indicating whether the message should be quarantined according to
      configured thresholds.

    Attributes:
      hashed_features: Mapping of hashed/aggregate features.
      intent_features: Optional enrichment metadata.
      should_quarantine: Whether the message should be quarantined.
    """

    hashed_features: Mapping[str, object]
    intent_features: Optional[IntentFeatures] = None
    should_quarantine: bool = False

    def as_dict(self) -> Dict[str, object]:
        """Serialise the record into a JSON-ready mapping."""

        payload: Dict[str, object] = {"hashed_features": dict(self.hashed_features)}
        if self.intent_features is not None:
            payload["intent_features"] = self.intent_features.as_dict()
        payload["should_quarantine"] = self.should_quarantine
        return payload


def iter_closed_vocabularies() -> Mapping[str, Sequence[str]]:
    """Expose the declared vocabularies for validation routines."""

    return {
        "intent": INTENT_VOCAB,
        "speech_acts": SPEECH_ACT_VOCAB,
        "persuasion": PERSUASION_VOCAB,
        "suspicion_flags": SUSPICION_FLAG_VOCAB,
    }

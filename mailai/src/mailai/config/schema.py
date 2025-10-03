"""Strict configuration schemas for MailAI runtime documents.

What:
  Define the strongly typed models representing ``config.yaml``, ``rules.yaml``,
  and ``status.yaml`` along with helper validators.

Why:
  The application ingests operator-maintained YAML via IMAP; rigorous schemas are
  required to enforce privacy limits, scheduling constraints, and deterministic
  behaviour across deployments.

How:
  Use Pydantic models with ``extra="forbid"`` policies, custom validators, and
  helper functions to normalise user input before rule execution. Convenience
  constructors such as :meth:`RulesV2.minimal` provide safe defaults for recovery
  workflows.

Interfaces:
  - Schema classes (e.g. :class:`RuntimeConfig`, :class:`RulesV2`,
    :class:`StatusV2`) consumed by loaders and runtime components.
  - Utility validators and helper functions that enforce structural invariants.

Invariants:
  - Every model forbids unexpected keys to highlight typos early.
  - Validators raise :class:`ValidationError` to keep error reporting uniform.

Safety/Performance:
  - Minimal constructors avoid expensive runtime introspection and provide
    deterministic bootstrap documents for offline recovery on constrained
    hardware.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError as _PydanticValidationError
from pydantic import field_validator, model_validator


class ValidationError(ValueError):
    """Schema violation raised during configuration validation.

    What:
      Represent user-facing validation issues encountered while parsing
      configuration payloads.

    Why:
      Normalising on a single exception simplifies error handling and preserves
      backwards compatibility with earlier releases.

    How:
      Extend :class:`ValueError` so the message is surfaced directly to CLI tools
      and log outputs.
    """


class SizeLimits(BaseModel):
    """Envelope limits governing YAML attachments in IMAP messages.

    What:
      Capture the soft and hard size boundaries enforced when downloading rules
      or status payloads.

    Why:
      Prevents oversized IMAP messages from exhausting memory on Raspberry Pi
      hardware and ensures truncation behaviour is auditable.

    How:
      Pydantic validators ensure the hard limit exceeds the soft limit, with both
      expressed as integer byte counts.

    Attributes:
      soft_limit: Threshold where warnings are emitted but processing continues.
      hard_limit: Maximum accepted size before the payload is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    soft_limit: int = Field(ge=0)
    hard_limit: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_limits(cls, model: "SizeLimits") -> "SizeLimits":
        """Ensure configured soft/hard limits maintain a safe ordering.

        What:
          Reject configurations where the hard limit is lower than the soft
          limit, as that would create unreachable warning thresholds.

        Why:
          Operators rely on the soft limit for advisory truncation; if the hard
          limit is lower, uploads could fail unexpectedly without prior warning.

        How:
          Compare the two integer fields and raise :class:`ValidationError` when
          the hard limit undercuts the soft limit.

        Args:
          model: The :class:`SizeLimits` instance to validate.

        Returns:
          The validated :class:`SizeLimits` instance for chaining.
        """

        if model.hard_limit < model.soft_limit:
            raise ValidationError("hard_limit must be greater than or equal to soft_limit")
        return model


class RulesMailConfig(BaseModel):
    """Location and constraints for the rules configuration email.

    What:
      Describe the IMAP folder, subject, and size limits containing
      ``rules.yaml``.

    Why:
      The loader needs this metadata to locate the authoritative configuration
      message and enforce confidentiality limits.

    How:
      Store human-readable strings for folder/subject along with a
      :class:`SizeLimits` instance used during sync.

    Attributes:
      subject: Subject line identifying the rules message.
      folder: IMAP mailbox path holding the message.
      limits: Size policy applied when fetching the payload.
    """

    model_config = ConfigDict(extra="forbid")

    subject: str
    folder: str
    limits: SizeLimits


class StatusMailConfig(BaseModel):
    """Location and constraints for the status telemetry email.

    What:
      Capture the IMAP folder/subject storing ``status.yaml`` and its size
      limits.

    Why:
      Enables the runtime to synchronise operational telemetry without leaking
      sensitive information or missing rotation events.

    How:
      Provide optional folder overrides, falling back to the rules folder when
      unspecified, and reuse :class:`SizeLimits` for payload policies.

    Attributes:
      subject: Subject line for the status message.
      folder: Optional mailbox path overriding the rules folder.
      limits: Size thresholds enforced during sync.
    """

    model_config = ConfigDict(extra="forbid")

    subject: str
    folder: Optional[str] = None
    limits: SizeLimits


class MailSettings(BaseModel):
    """Aggregate configuration for rules and status IMAP resources.

    What:
      Group the rules and status mail configuration into a single document
      section.

    Why:
      Keeps mail-related settings cohesive, simplifying validation and auditing
      boundaries between IMAP-specific logic and other subsystems.

    How:
      Compose :class:`RulesMailConfig` and :class:`StatusMailConfig` into a
      Pydantic model with ``extra="forbid"`` to highlight typos.

    Attributes:
      rules: Configuration for the rules source message.
      status: Configuration for the status telemetry message.
    """

    model_config = ConfigDict(extra="forbid")

    rules: RulesMailConfig
    status: StatusMailConfig


class ImapSettings(BaseModel):
    """IMAP namespace defaults applied by the runtime.

    What:
      Define the folders used for message processing and quarantine actions.

    Why:
      Consistent IMAP paths are necessary to avoid sequence-number operations and
      guarantee UID-first workflows.

    How:
      Specify the default mailbox, control namespace used for metadata, and the
      quarantine subfolder for potentially dangerous messages.

    Attributes:
      default_mailbox: Primary mailbox to scan (defaults to ``INBOX``).
      control_namespace: Root folder where control mails (rules/status) reside.
      quarantine_subfolder: Destination for messages requiring manual review.
    """

    model_config = ConfigDict(extra="forbid")

    default_mailbox: str = "INBOX"
    control_namespace: str
    quarantine_subfolder: str


class PathsConfig(BaseModel):
    """Filesystem directories required by the runtime.

    What:
      Declare the paths housing runtime state, configuration, and models.

    Why:
      Explicit directory definitions avoid accidental writes outside controlled
      locations, which is critical on devices with constrained storage.

    How:
      Provide strings pointing to directories resolved by higher-level loaders.

    Attributes:
      state_dir: Location for mutable runtime state.
      config_dir: Directory containing synchronised configuration mails.
      models_dir: Storage area for local LLM models.
    """

    model_config = ConfigDict(extra="forbid")

    state_dir: str
    config_dir: str
    models_dir: str


class FeedbackConfig(BaseModel):
    """Settings controlling optional user feedback ingestion.

    What:
      Configure whether the runtime collects manual feedback emails and how to
      identify them.

    Why:
      Feedback is optional and may be disabled in privacy-sensitive deployments;
      explicit settings make that choice auditable.

    How:
      Toggle the feature with a boolean flag and record mailbox/subject metadata
      when enabled.

    Attributes:
      enabled: Whether feedback ingestion is active.
      mailbox: Optional IMAP folder to monitor for feedback.
      subject_prefix: Optional subject prefix used to classify feedback.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mailbox: Optional[str] = None
    subject_prefix: Optional[str] = None


class RuntimeLLMConfig(BaseModel):
    """Local LLM execution parameters.

    What:
      Describe the model path, threading, context window, and timeout behaviour
      used to host ``llama-cpp-python``.

    Why:
      Raspberry Pi deployments need carefully tuned limits to balance startup
      latency, inference responsiveness, and watchdog health checks.

    How:
      Provide numeric settings validated by Pydantic, including explicit timeout
      fields to accommodate cold starts and subsequent warm-ups.

    Attributes:
      model_path: Filesystem location of the GGUF model.
      threads: Number of CPU threads allocated to inference.
      ctx_size: Token context window for completions.
      sentinel_path: Path to the sentinel prompt used during health checks.
      max_age: Maximum acceptable age of the sentinel completion cache.
      load_timeout_s: Startup timeout when loading the model from disk.
      warmup_completion_timeout_s: Timeout for the warm-up completion request.
      healthcheck_timeout_s: Timeout for lightweight readiness probes.
    """

    model_config = ConfigDict(extra="forbid")

    model_path: str
    threads: int = Field(gt=0)
    ctx_size: int = Field(gt=0)
    sentinel_path: str
    max_age: int = Field(gt=0)
    load_timeout_s: int = Field(gt=0, default=120)
    warmup_completion_timeout_s: int = Field(gt=0, default=10)
    healthcheck_timeout_s: int = Field(gt=0, default=5)


class IntentFeaturesLLMConfig(BaseModel):
    """Structured LLM runtime parameters for intent enrichment."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(gt=0, le=256, default=64)
    temperature: float = Field(ge=0.0, le=1.0, default=0.0)
    timeout_s: float = Field(gt=0.0, default=3.0)


class IntentFeaturesThresholds(BaseModel):
    """Thresholds controlling downstream reactions to enrichment scores."""

    model_config = ConfigDict(extra="forbid")

    scam_singularity_quarantine: int = Field(ge=0, le=3, default=2)
    urgency_warn: int = Field(ge=0, le=3, default=3)


class IntentFeaturesConfig(BaseModel):
    """Top-level toggle and settings for intent enrichment."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    llm: IntentFeaturesLLMConfig = Field(default_factory=IntentFeaturesLLMConfig)
    thresholds: IntentFeaturesThresholds = Field(default_factory=IntentFeaturesThresholds)


class RuntimeConfig(BaseModel):
    """Top-level runtime configuration derived from ``config.yaml``.

    What:
      Aggregate IMAP, filesystem, LLM, and feedback settings consumed by the
      application at startup.

    Why:
      Keeping a single authoritative model ensures all subsystems share a common
      view of operator intent.

    How:
      Compose subordinate models with strict validation and provide default
      values where applicable (e.g. ``version`` and ``feedback``).

    Attributes:
      version: Configuration schema version indicator.
      paths: Filesystem paths required by the runtime.
      imap: IMAP namespace defaults.
      mail: Rules/status mailbox configuration.
      llm: Embedded LLM parameters.
      feedback: Optional feedback ingestion settings.
      intent_features: Intent enrichment toggle and runtime parameters.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    paths: PathsConfig
    imap: ImapSettings
    mail: MailSettings
    llm: RuntimeLLMConfig
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    intent_features: IntentFeaturesConfig = Field(default_factory=IntentFeaturesConfig)


def _ensure_dict(value: Any, name: str) -> Dict[str, Any]:
    """Assert that ``value`` is a dictionary or raise :class:`ValidationError`.

    What:
      Provide defensive validation for nested structures within the rules
      document.

    Why:
      The YAML parser may produce scalars or lists where mappings are expected;
      normalising these early produces clearer error messages for operators.

    How:
      Use :func:`isinstance` checks and raise :class:`ValidationError` with field
      context when the type does not match.

    Args:
      value: The object to validate.
      name: Human-readable field identifier used in the error message.

    Returns:
      The same value cast as ``Dict[str, Any]`` when validation succeeds.
    """

    if not isinstance(value, dict):
        raise ValidationError(f"{name} expected mapping")
    return value


def _ensure_list(value: Any, name: str) -> List[Any]:
    """Assert that ``value`` is a list or raise :class:`ValidationError`.

    What:
      Validate list-typed fields embedded within the rules schema.

    Why:
      Prevents subtle bugs where YAML scalars or mappings appear due to operator
      mistakes, ensuring deterministic parsing.

    How:
      Perform an ``isinstance`` check and raise :class:`ValidationError` with the
      offending field name when the expectation is not met.

    Args:
      value: The object to check.
      name: Field identifier used in error reporting.

    Returns:
      The list when validation succeeds.
    """

    if not isinstance(value, list):
        raise ValidationError(f"{name} expected list")
    return value


def _normalise_condition(value: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single rule condition and copy its payload.

    What:
      Ensure that condition dictionaries contain exactly one recognised key and
      return a shallow copy for downstream processing.

    Why:
      Guards against ambiguous conditions and enforces the whitelist of
      supported predicates, which is critical for privacy reviews.

    How:
      Check the length of the mapping, verify the key is allowed, and duplicate
      nested dictionaries to avoid retaining mutable references to user input.

    Args:
      value: Mapping representing a rule condition.

    Returns:
      A normalised mapping with the same key and a copied payload.

    Raises:
      ValidationError: If the condition is malformed or uses unsupported fields.
    """

    if len(value) != 1:
        raise ValidationError("condition must have exactly one entry")
    field, payload = next(iter(value.items()))
    allowed = {
        "header",
        "category_pred",
        "subject",
        "body",
        "from",
        "to",
        "cc",
        "bcc",
        "mailbox",
        "has_attachment",
        "attachment_name",
        "range",
    }
    if field not in allowed:
        raise ValidationError(f"unsupported condition field '{field}'")
    if isinstance(payload, dict):
        return {field: dict(payload)}
    return {field: payload}


class Metadata(BaseModel):
    """Descriptive metadata attached to the rules document.

    What:
      Record human-readable information about the policy, including ownership and
      last update timestamp.

    Why:
      Helps auditors and operators track provenance when reviewing automation
      changes.

    How:
      Store immutable strings validated by Pydantic.

    Attributes:
      description: Short summary of the ruleset.
      owner: Identifier of the maintainer or team.
      updated_at: ISO 8601 timestamp of the last update.
    """

    model_config = ConfigDict(extra="forbid")

    description: str
    owner: str
    updated_at: str


class Schedule(BaseModel):
    """Timing configuration for learning and inference tasks.

    What:
      Capture the cron schedule for retraining and the interval for inference
      passes.

    Why:
      Ensures the learning pipeline runs at predictable intervals appropriate for
      the deployment.

    How:
      Validate cron strings and durations as simple fields consumed by the cron
      helper utilities.

    Attributes:
      learn_cron: Cron expression controlling retraining cadence.
      inference_interval_s: Seconds between inference cycles.
    """

    model_config = ConfigDict(extra="forbid")

    learn_cron: str
    inference_interval_s: int


class EncryptionConfig(BaseModel):
    """Encryption toggles and key management configuration.

    What:
      Specify whether at-rest encryption is enabled and where to obtain the key.

    Why:
      Protects persisted feature stores and backups by referencing the correct
      secret material.

    How:
      Provide a boolean flag and a key path that runtime components resolve at
      startup.

    Attributes:
      enabled: Whether encryption should be applied.
      key_path: Filesystem path to the encryption key material.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    key_path: str


class PrivacyConfig(BaseModel):
    """Privacy safeguards for persisted analytics.

    What:
      Define storage locations and cryptographic materials required to preserve
      confidentiality when storing features or hashes.

    Why:
      MailAI handles sensitive content and must prevent plaintext leaks while
      enabling aggregate analytics.

    How:
      Combine file paths for encrypted stores with pepper/salt locations used for
      irreversible hashing, plus thresholds for acceptable plaintext windows.

    Attributes:
      feature_store_path: Path to the encrypted feature database.
      encryption: Nested :class:`EncryptionConfig` describing encryption state.
      hashing_pepper_path: Location of the pepper secret.
      hashing_salt_path: Location of the salt secret.
      max_plaintext_window_chars: Maximum plaintext retention allowed.
    """

    model_config = ConfigDict(extra="forbid")

    feature_store_path: str
    encryption: EncryptionConfig
    hashing_pepper_path: str
    hashing_salt_path: str
    max_plaintext_window_chars: int


class LLMConfig(BaseModel):
    """Settings for remote or local LLM providers used during learning.

    What:
      Describe the provider endpoint, model identifier, and sampling parameters
      used for proposal generation.

    Why:
      The learner may interact with external services; explicit configuration
      ensures reproducible behaviour and cost controls.

    How:
      Store provider strings alongside numeric hyperparameters validated by
      Pydantic.

    Attributes:
      provider: Name of the LLM provider (e.g. ``local``).
      model: Identifier for the model or preset.
      max_tokens: Maximum tokens to request per completion.
      temperature: Sampling temperature applied during generation.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    max_tokens: int
    temperature: float


class EmbeddingConfig(BaseModel):
    """Optional embedding backend parameters.

    What:
      Configure whether embedding generation is enabled and how to reach the
      backend.

    Why:
      Embeddings are optional due to resource constraints; explicit toggles
      prevent accidental activation without full configuration.

    How:
      Validate that backend and dimensionality are present when enabled, using a
      post-validation hook.

    Attributes:
      enabled: Whether embedding generation runs.
      backend: Identifier or URL for the embedding service.
      dim: Expected embedding dimensionality.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: Optional[str] = None
    dim: Optional[int] = None

    @model_validator(mode="after")
    def _validate_backend(cls, model: "EmbeddingConfig") -> "EmbeddingConfig":
        """Check embedding enablement requirements are satisfied.

        What:
          Prevent partially configured embedding backends from passing schema
          validation when ``enabled`` is true.

        Why:
          Downstream components assume both ``backend`` and ``dim`` are present
          before attempting to generate embeddings; missing values would surface
          as runtime failures.

        How:
          When ``enabled`` evaluates to ``True``, assert that ``backend`` and
          ``dim`` are populated, otherwise raise :class:`ValidationError`.

        Args:
          model: Candidate embedding configuration.

        Returns:
          The validated :class:`EmbeddingConfig` instance.
        """

        if model.enabled:
            if not model.backend:
                raise ValidationError("embeddings.backend required when enabled")
            if model.dim is None:
                raise ValidationError("embeddings.dim required when enabled")
        return model


class RuleSynthesisConfig(BaseModel):
    """Controls for the automated rule synthesis pipeline.

    What:
      Configure whether the learner should propose new rules and how many to
      surface per iteration.

    Why:
      Helps balance automation agility with operator oversight, ensuring the
      learner does not overwhelm reviewers.

    How:
      Provide flags for enablement and confirmation requirements validated via
      Pydantic defaults.

    Attributes:
      enabled: Whether the learner may generate proposals.
      max_rules_per_pass: Upper bound on proposals per cycle.
      require_user_confirmation: Whether proposals must be manually approved.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_rules_per_pass: int = 3
    require_user_confirmation: bool = True


class DeleteSemanticsConfig(BaseModel):
    """Configuration for delete semantics inference.

    What:
      Describe whether the learner should infer user intent around deletions and
      which signals inform that inference.

    Why:
      Deletion behaviour is privacy sensitive; explicit settings make automated
      suggestions auditable.

    How:
      Store a boolean toggle and a list of signals consumed by the learner.

    Attributes:
      infer_meaning: Whether delete semantics inference is enabled.
      signals: Telemetry keys considered during inference.
    """

    model_config = ConfigDict(extra="forbid")

    infer_meaning: bool = True
    signals: List[str] = Field(default_factory=list)


class LearningConfig(BaseModel):
    """High-level parameters controlling the learning subsystem.

    What:
      Capture enablement flags, window sizes, and nested configuration for LLMs,
      embeddings, rule synthesis, and delete semantics.

    Why:
      Centralises the knobs operators use to tune the learner while ensuring
      dependencies are validated as a cohesive unit.

    How:
      Compose several subordinate models with strict validation and default
      factories.

    Attributes:
      enabled: Whether the learning pipeline runs.
      window_days: Historical window size used for training.
      min_samples_per_class: Minimum labelled samples per category.
      llm: :class:`LLMConfig` describing external inference provider settings.
      embeddings: :class:`EmbeddingConfig` controlling embedding generation.
      rule_synthesis: :class:`RuleSynthesisConfig` for proposal controls.
      delete_semantics: :class:`DeleteSemanticsConfig` describing delete logic.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    window_days: int
    min_samples_per_class: int
    llm: LLMConfig
    embeddings: EmbeddingConfig
    rule_synthesis: RuleSynthesisConfig
    delete_semantics: DeleteSemanticsConfig


class DefaultsConfig(BaseModel):
    """Global behaviour defaults applied by the rule engine.

    What:
      Specify how the engine treats case sensitivity, stop-on-match semantics,
      and dry-run mode when rules omit explicit overrides.

    Why:
      Makes implicit behaviour visible to operators and keeps execution
      predictable across rule updates.

    How:
      Store booleans with sensible defaults validated by Pydantic.

    Attributes:
      case_sensitive: Whether textual matching is case sensitive by default.
      stop_on_first_match: Whether the engine halts after the first rule match.
      dry_run: Whether actions are simulated without affecting mailboxes.
    """

    model_config = ConfigDict(extra="forbid")

    case_sensitive: bool = False
    stop_on_first_match: bool = False
    dry_run: bool = True


class RuleMatch(BaseModel):
    """Container for the match clauses of a rule.

    What:
      Hold the ``any`` / ``all`` / ``none`` condition lists after validation.

    Why:
      Encapsulating the structure clarifies how rules are evaluated and ensures
      each clause is validated consistently.

    How:
      Use post-validation to ensure at least one clause is present and normalise
      nested conditions via helper functions.

    Attributes:
      any: List of conditions where any may match.
      all: List of conditions that must all match.
      none: List of conditions that must not match.
    """

    model_config = ConfigDict(extra="forbid")

    any: List[Dict[str, Any]] = Field(default_factory=list)
    all: List[Dict[str, Any]] = Field(default_factory=list)
    none: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalise(cls, model: "RuleMatch") -> "RuleMatch":
        """Normalise rule clauses and enforce presence of at least one condition.

        What:
          Reject empty rule matches and convert clause dictionaries into the
          canonical format consumed by the engine.

        Why:
          Empty match blocks would create rules that always fire; normalisation
          ensures comparisons behave consistently regardless of YAML formatting.

        How:
          Check that at least one of ``any``, ``all``, or ``none`` contains
          entries, then run each mapping through :func:`_ensure_dict` and
          :func:`_normalise_condition`.

        Args:
          model: Rule match structure produced by initial parsing.

        Returns:
          The validated :class:`RuleMatch` instance.
        """

        if not model.any and not model.all and not model.none:
            raise ValidationError("match must define at least one clause")
        model.any = [_normalise_condition(_ensure_dict(item, "match.any")) for item in model.any]
        model.all = [_normalise_condition(_ensure_dict(item, "match.all")) for item in model.all]
        model.none = [_normalise_condition(_ensure_dict(item, "match.none")) for item in model.none]
        return model


class Rule(BaseModel):
    """Single automation rule definition.

    What:
      Store metadata, matching conditions, and actions describing one automation
      rule.

    Why:
      Rules are the core policy unit; capturing their structure in a Pydantic
      model allows consistent validation and downstream execution.

    How:
      Enforce allowed action formats via validators and require descriptive
      fields for auditing.

    Attributes:
      id: Stable identifier for the rule.
      description: Human-readable summary of the behaviour.
      why: Rationale explaining the intent behind the rule.
      source: Origin of the rule (deterministic or learner-generated).
      enabled: Whether the rule is active.
      priority: Ordering used during evaluation.
      match: :class:`RuleMatch` describing matching conditions.
      actions: List of normalised actions to apply.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    why: str
    source: Literal["deterministic", "learner"]
    enabled: bool = True
    priority: int
    match: RuleMatch
    actions: List[Dict[str, Any]]

    @field_validator("actions", mode="before")
    @classmethod
    def _validate_actions(cls, value: Any) -> List[Dict[str, Any]]:
        """Normalise action mappings and ensure one operation per entry.

        What:
          Convert raw YAML structures into dictionaries containing exactly one
          action verb.

        Why:
          Multi-operation entries would be ambiguous for the engine; forcing one
          operation per dict keeps ordering explicit and simplifies execution.

        How:
          Coerce ``value`` into a list, validate each element is a mapping with a
          single key, and clone the mapping into a plain ``dict`` for storage.

        Args:
          value: Raw action payload parsed from YAML.

        Returns:
          List of single-operation dictionaries ready for execution.
        """

        items = _ensure_list(value, "rule.actions")
        result: List[Dict[str, Any]] = []
        for item in items:
            mapping = _ensure_dict(item, "rule.action")
            if len(mapping) != 1:
                raise ValidationError("action must contain exactly one operation")
            result.append(dict(mapping))
        return result


class RulesV2(BaseModel):
    """Top-level schema describing ``rules.yaml``.

    What:
      Aggregate metadata, scheduling, privacy safeguards, defaults, and the list
      of automation rules.

    Why:
      Provides a single validated object representing the full automation policy
      loaded from IMAP.

    How:
      Compose subordinate models with strict validation and expose helper
      constructors for deterministic bootstrap documents.

    Attributes:
      version: Schema version indicator (fixed to ``2``).
      meta: :class:`Metadata` describing the policy.
      schedule: :class:`Schedule` controlling execution cadence.
      privacy: :class:`PrivacyConfig` detailing safeguards.
      learning: :class:`LearningConfig` governing learner behaviour.
      defaults: :class:`DefaultsConfig` specifying engine defaults.
      rules: List of :class:`Rule` entries executed in priority order.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[2]
    meta: Metadata
    schedule: Schedule
    privacy: PrivacyConfig
    learning: LearningConfig
    defaults: DefaultsConfig
    rules: List[Rule]

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "RulesV2":  # type: ignore[override]
        """Validate payloads and surface schema errors as :class:`ValidationError`.

        What:
          Convert arbitrary dictionaries into :class:`RulesV2` models while
          preserving human-readable error messages.

        Why:
          Downstream callers expect ``ValidationError`` instances rather than raw
          Pydantic exceptions to maintain compatibility with earlier releases.

        How:
          Delegate to the parent implementation and wrap
          :class:`pydantic.ValidationError` into the repository-specific error.

        Args:
          data: Payload to validate.

        Returns:
          A validated :class:`RulesV2` instance.
        """

        try:
            return super().model_validate(data)
        except _PydanticValidationError as exc:  # pragma: no cover - exercised indirectly
            raise ValidationError(str(exc)) from exc

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:  # type: ignore[override]
        """Return a serialisable representation respecting repository defaults.

        What:
          Provide a consistent dictionary suitable for YAML dumping regardless of
          the requested ``mode``.

        Why:
          The loader expects deterministic JSON-compatible payloads; overriding
          ensures no extra metadata leaks into the serialisation.

        How:
          Delegate to :meth:`BaseModel.model_dump` ignoring the ``mode`` argument
          while satisfying the type checker with the override.

        Args:
          mode: Ignored but kept for signature compatibility.

        Returns:
          Dictionary representation of the model.
        """

        return super().model_dump()

    @classmethod
    def minimal(cls) -> "RulesV2":
        """Return a deterministic baseline policy suitable for bootstrapping.

        What:
          Construct a permissive ruleset that leaves messages untouched while
          exercising the engine pipeline.

        Why:
          Recovery flows and new deployments need a safe default to ensure
          services remain operational even without custom automation.

        How:
          Build a catch-all rule targeting ``INBOX`` and combine it with sensible
          defaults for learning, privacy, and scheduling. The configuration keeps
          encryption enabled and disables optional subsystems that require extra
          provisioning (such as embeddings).

        Returns:
          A fully validated :class:`RulesV2` instance.
        """

        baseline_match = RuleMatch(any=[{"mailbox": {"equals": "INBOX"}}], all=[], none=[])
        baseline_rule = Rule(
            id="baseline",
            description="Baseline catch-all to leave messages untouched",
            why="Ensures the engine records activity even when no automation is configured",
            source="deterministic",
            enabled=True,
            priority=100,
            match=baseline_match,
            actions=[{"stop_processing": True}],
        )
        return cls(
            version=2,
            meta=Metadata(
                description="Default MailAI policy",
                owner="mailai",
                updated_at="1970-01-01T00:00:00Z",
            ),
            schedule=Schedule(learn_cron="0 3 * * *", inference_interval_s=900),
            privacy=PrivacyConfig(
                feature_store_path="/var/lib/mailai/features.db",
                encryption=EncryptionConfig(enabled=True, key_path="/run/secrets/sqlcipher_key"),
                hashing_pepper_path="/run/secrets/account_pepper",
                hashing_salt_path="/run/secrets/global_hash_salt",
                max_plaintext_window_chars=2048,
            ),
            learning=LearningConfig(
                enabled=True,
                window_days=14,
                min_samples_per_class=5,
                llm=LLMConfig(provider="local", model="mailai-tiny", max_tokens=512, temperature=0.0),
                embeddings=EmbeddingConfig(enabled=False, backend=None, dim=None),
                rule_synthesis=RuleSynthesisConfig(enabled=True, max_rules_per_pass=3, require_user_confirmation=True),
                delete_semantics=DeleteSemanticsConfig(
                    infer_meaning=True,
                    signals=["calendar_invite_past", "thread_is_resolved"],
                ),
            ),
            defaults=DefaultsConfig(case_sensitive=False, stop_on_first_match=False, dry_run=True),
            rules=[baseline_rule],
        )


class StatusSummary(BaseModel):
    """Aggregate counters describing the latest engine run.

    What:
      Record message counts, action totals, and error tallies for reporting.

    Why:
      Operators rely on this summary to spot regressions or spikes without
      parsing the full event log.

    How:
      Store integer counters validated by Pydantic.

    Attributes:
      scanned_messages: Number of messages scanned in the run.
      matched_messages: Number of messages matched by any rule.
      actions_applied: Total actions executed.
      errors: Count of errors encountered.
      warnings: Count of warnings emitted.
    """

    model_config = ConfigDict(extra="forbid")

    scanned_messages: int
    matched_messages: int
    actions_applied: int
    errors: int
    warnings: int


class RuleMetrics(BaseModel):
    """Per-rule execution counters.

    What:
      Track how often a rule matched, how many actions it executed, and errors it
      produced.

    Why:
      Facilitates tuning and auditing individual rules for accuracy and safety.

    How:
      Maintain integer counters validated by Pydantic.

    Attributes:
      matches: Number of times the rule matched.
      actions: Number of actions emitted by the rule.
      errors: Number of errors attributed to the rule.
    """

    model_config = ConfigDict(extra="forbid")

    matches: int
    actions: int
    errors: int


class LearningMetrics(BaseModel):
    """Telemetry describing the learner's recent activity.

    What:
      Capture timing information, sample counts, class distribution, and rule
      proposal statistics.

    Why:
      Operators need insight into whether learning is running, converging, or
      proposing actionable changes.

    How:
      Store optional timestamps and counters validated by Pydantic with default
      factories.

    Attributes:
      last_train_started_at: ISO timestamp of the last training start.
      last_train_finished_at: ISO timestamp of the last training end.
      samples_used: Number of samples used during training.
      classes: Labels observed during training.
      macro_f1: Optional macro F1 score summarising model quality.
      proposed_rules: Number of rules proposed in the last cycle.
      delete_semantics: Mapping of delete semantics signals and counts.
    """

    model_config = ConfigDict(extra="forbid")

    last_train_started_at: Optional[str] = None
    last_train_finished_at: Optional[str] = None
    samples_used: int = 0
    classes: List[str] = Field(default_factory=list)
    macro_f1: Optional[float] = None
    proposed_rules: int = 0
    delete_semantics: Dict[str, int] = Field(default_factory=dict)


class PrivacyStatus(BaseModel):
    """Operational indicators for privacy protections.

    What:
      Track whether encryption is active, if plaintext leaks were detected, and
      whether pepper rotation is due.

    Why:
      Provides at-a-glance assurance that privacy safeguards remain effective.

    How:
      Store booleans and counters validated by Pydantic.

    Attributes:
      feature_store_encrypted: Whether the feature store is encrypted.
      plaintext_leaks_detected: Count of detected plaintext leak events.
      pepper_rotation_due: Whether pepper rotation is required.
    """

    model_config = ConfigDict(extra="forbid")

    feature_store_encrypted: bool = True
    plaintext_leaks_detected: int = 0
    pepper_rotation_due: bool = False


class Proposal(BaseModel):
    """Learner-generated rule proposal metadata.

    What:
      Describe a proposed rule diff, including identifier and rationale.

    Why:
      Allows operators to review and approve or reject learner suggestions.

    How:
      Store identifiers and textual explanations validated by Pydantic.

    Attributes:
      rule_id: Identifier for the proposed rule.
      diff: Unified diff or textual change summary.
      why: Explanation describing the proposal rationale.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    diff: str
    why: str


class ConfigReference(BaseModel):
    """Pointer to the IMAP message holding the active configuration.

    What:
      Record IMAP UID, message-id, timestamp, and checksum referencing the active
      rules document.

    Why:
      Enables reconciliation between local state and the authoritative email copy
      during audits.

    How:
      Store identifiers validated by Pydantic.

    Attributes:
      uid: IMAP UID of the configuration message.
      message_id: RFC822 Message-ID if available.
      internaldate: IMAP internal date string.
      checksum: SHA-256 checksum of the message body.
    """

    model_config = ConfigDict(extra="forbid")

    uid: int
    message_id: Optional[str]
    internaldate: str
    checksum: str


class StatusEvent(BaseModel):
    """Timeline event documenting configuration changes.

    What:
      Describe when the configuration was updated, invalid, or restored.

    Why:
      Maintains an audit trail showing how the runtime responded to operator
      actions.

    How:
      Store timestamped entries with a fixed set of event types and free-form
      details.

    Attributes:
      ts: ISO timestamp when the event occurred.
      type: Event type describing the action.
      details: Additional human-readable context.
    """

    model_config = ConfigDict(extra="forbid")

    ts: str
    type: Literal["config_updated", "config_invalid", "config_restored"]
    details: str


class StatusV2(BaseModel):
    """Top-level schema representing ``status.yaml``.

    What:
      Capture runtime mode flags, per-rule metrics, learning telemetry, privacy
      status, and references to the active configuration message.

    Why:
      Provides a single document summarising the agent's health for diagnostics
      and auditing.

    How:
      Compose subordinate metric classes with strict validation, provide minimal
      constructors, and override validation to translate Pydantic errors into the
      repository-specific :class:`ValidationError`.

    Attributes:
      run_id: Identifier of the runtime invocation.
      config_checksum: Checksum of the active rules document.
      mailbox: Mailbox currently being processed.
      last_run_started_at: Timestamp when the run began.
      last_run_finished_at: Optional timestamp for run completion.
      mode: Mapping describing runtime mode toggles.
      summary: :class:`StatusSummary` counters.
      by_rule: Mapping of rule IDs to :class:`RuleMetrics`.
      learning: :class:`LearningMetrics` telemetry.
      privacy: :class:`PrivacyStatus` overview.
      notes: Operator notes captured during the run.
      proposals: List of :class:`Proposal` entries awaiting review.
      config_ref: Optional :class:`ConfigReference` pointing to the source email.
      events: Chronological list of :class:`StatusEvent` entries.
      restored_rules_from_backup: Whether the run used the encrypted rules backup.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    config_checksum: str
    mailbox: str
    last_run_started_at: str
    last_run_finished_at: Optional[str] = None
    mode: Dict[str, bool] = Field(default_factory=dict)
    summary: StatusSummary
    by_rule: Dict[str, RuleMetrics] = Field(default_factory=dict)
    learning: LearningMetrics
    privacy: PrivacyStatus
    notes: List[str] = Field(default_factory=list)
    proposals: List[Proposal] = Field(default_factory=list)
    config_ref: Optional[ConfigReference] = None
    events: List[StatusEvent] = Field(default_factory=list)
    restored_rules_from_backup: bool = False

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "StatusV2":  # type: ignore[override]
        """Validate status payloads and rethrow schema issues uniformly.

        What:
          Convert dictionaries into :class:`StatusV2` models while maintaining
          consistent error types.

        Why:
          Aligns validation error handling with :class:`RulesV2` so loaders can
          treat both documents identically.

        How:
          Delegate to Pydantic's validator and rewrap its exceptions as
          :class:`ValidationError` instances.

        Args:
          data: Candidate payload for validation.

        Returns:
          A validated :class:`StatusV2` model.
        """

        try:
            return super().model_validate(data)
        except _PydanticValidationError as exc:  # pragma: no cover - exercised indirectly
            raise ValidationError(str(exc)) from exc

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:  # type: ignore[override]
        """Return a canonical dictionary representation of the status payload.

        What:
          Provide deterministic serialisation suitable for YAML emission.

        Why:
          Ensures downstream tools receive consistent structures regardless of
          Pydantic defaults.

        How:
          Delegate to :meth:`BaseModel.model_dump` while ignoring ``mode`` for
          compatibility.

        Args:
          mode: Ignored but preserved for signature compatibility.

        Returns:
          Dictionary representation of the status snapshot.
        """

        return super().model_dump()

    @classmethod
    def minimal(cls) -> "StatusV2":
        """Return a minimal status document used when no telemetry is available.

        What:
          Provide a deterministic baseline snapshot with zeroed counters and empty
          collections.

        Why:
          Recovery flows and initial deployments require a valid status document
          even before the first run completes.

        How:
          Instantiate the subordinate models with default values and return the
          aggregate :class:`StatusV2` instance.

        Returns:
          A :class:`StatusV2` object populated with safe defaults.
        """

        return cls(
            run_id="bootstrap",
            config_checksum="",
            mailbox="",
            last_run_started_at="",
            last_run_finished_at=None,
            mode={},
            summary=StatusSummary(
                scanned_messages=0,
                matched_messages=0,
                actions_applied=0,
                errors=0,
                warnings=0,
            ),
            by_rule={},
            learning=LearningMetrics(),
            privacy=PrivacyStatus(),
            notes=[],
            proposals=[],
            config_ref=None,
            events=[],
            restored_rules_from_backup=False,
        )


# TODO: Other modules require the same treatment (Quoi/Pourquoi/Comment docstrings + module header).

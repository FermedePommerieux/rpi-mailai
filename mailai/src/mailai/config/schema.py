"""Pydantic models describing MailAI configuration documents."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError as _PydanticValidationError
from pydantic import field_validator, model_validator


class ValidationError(ValueError):
    """Raised when configuration data does not satisfy the schema."""


class SizeLimits(BaseModel):
    """Size thresholds enforced when syncing IMAP resources."""

    model_config = ConfigDict(extra="forbid")

    soft_limit: int = Field(ge=0)
    hard_limit: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_limits(cls, model: "SizeLimits") -> "SizeLimits":
        if model.hard_limit < model.soft_limit:
            raise ValidationError("hard_limit must be greater than or equal to soft_limit")
        return model


class RulesMailConfig(BaseModel):
    """Configuration describing how the rules mail is stored."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    folder: str
    limits: SizeLimits


class StatusMailConfig(BaseModel):
    """Configuration describing how the status mail is stored."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    folder: Optional[str] = None
    limits: SizeLimits


class MailSettings(BaseModel):
    """Top-level mail resource configuration."""

    model_config = ConfigDict(extra="forbid")

    rules: RulesMailConfig
    status: StatusMailConfig


class ImapSettings(BaseModel):
    """Server level IMAP defaults used by the runtime."""

    model_config = ConfigDict(extra="forbid")

    default_mailbox: str = "INBOX"
    control_namespace: str
    quarantine_subfolder: str


class PathsConfig(BaseModel):
    """Filesystem layout used by the runtime."""

    model_config = ConfigDict(extra="forbid")

    state_dir: str
    config_dir: str
    models_dir: str


class FeedbackConfig(BaseModel):
    """User feedback controls."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mailbox: Optional[str] = None
    subject_prefix: Optional[str] = None


class RuntimeLLMConfig(BaseModel):
    """Runtime parameters for the embedded LLM."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    threads: int = Field(gt=0)
    ctx_size: int = Field(gt=0)
    sentinel_path: str
    max_age: int = Field(gt=0)


class RuntimeConfig(BaseModel):
    """Root configuration loaded from ``config.cfg``."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    paths: PathsConfig
    imap: ImapSettings
    mail: MailSettings
    llm: RuntimeLLMConfig
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)


def _ensure_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} expected mapping")
    return value


def _ensure_list(value: Any, name: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} expected list")
    return value


def _normalise_condition(value: Dict[str, Any]) -> Dict[str, Any]:
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
    """Metadata describing the rules document."""

    model_config = ConfigDict(extra="forbid")

    description: str
    owner: str
    updated_at: str


class Schedule(BaseModel):
    """Learning and inference schedule configuration."""

    model_config = ConfigDict(extra="forbid")

    learn_cron: str
    inference_interval_s: int


class EncryptionConfig(BaseModel):
    """Privacy encryption configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    key_path: str


class PrivacyConfig(BaseModel):
    """Settings governing privacy guarantees."""

    model_config = ConfigDict(extra="forbid")

    feature_store_path: str
    encryption: EncryptionConfig
    hashing_pepper_path: str
    hashing_salt_path: str
    max_plaintext_window_chars: int


class LLMConfig(BaseModel):
    """Local LLM serving configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    max_tokens: int
    temperature: float


class EmbeddingConfig(BaseModel):
    """Embedding backend configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: Optional[str] = None
    dim: Optional[int] = None

    @model_validator(mode="after")
    def _validate_backend(cls, model: "EmbeddingConfig") -> "EmbeddingConfig":
        if model.enabled:
            if not model.backend:
                raise ValidationError("embeddings.backend required when enabled")
            if model.dim is None:
                raise ValidationError("embeddings.dim required when enabled")
        return model


class RuleSynthesisConfig(BaseModel):
    """Learner configuration for synthetic rule proposals."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_rules_per_pass: int = 3
    require_user_confirmation: bool = True


class DeleteSemanticsConfig(BaseModel):
    """Settings for delete semantics inference."""

    model_config = ConfigDict(extra="forbid")

    infer_meaning: bool = True
    signals: List[str] = Field(default_factory=list)


class LearningConfig(BaseModel):
    """High level learning pipeline settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    window_days: int
    min_samples_per_class: int
    llm: LLMConfig
    embeddings: EmbeddingConfig
    rule_synthesis: RuleSynthesisConfig
    delete_semantics: DeleteSemanticsConfig


class DefaultsConfig(BaseModel):
    """Global defaults affecting rule execution."""

    model_config = ConfigDict(extra="forbid")

    case_sensitive: bool = False
    stop_on_first_match: bool = False
    dry_run: bool = True


class RuleMatch(BaseModel):
    """Container for rule match clauses."""

    model_config = ConfigDict(extra="forbid")

    any: List[Dict[str, Any]] = Field(default_factory=list)
    all: List[Dict[str, Any]] = Field(default_factory=list)
    none: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalise(cls, model: "RuleMatch") -> "RuleMatch":
        if not model.any and not model.all and not model.none:
            raise ValidationError("match must define at least one clause")
        model.any = [_normalise_condition(_ensure_dict(item, "match.any")) for item in model.any]
        model.all = [_normalise_condition(_ensure_dict(item, "match.all")) for item in model.all]
        model.none = [_normalise_condition(_ensure_dict(item, "match.none")) for item in model.none]
        return model


class Rule(BaseModel):
    """Representation of a single automation rule."""

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
        items = _ensure_list(value, "rule.actions")
        result: List[Dict[str, Any]] = []
        for item in items:
            mapping = _ensure_dict(item, "rule.action")
            if len(mapping) != 1:
                raise ValidationError("action must contain exactly one operation")
            result.append(dict(mapping))
        return result


class RulesV2(BaseModel):
    """Top-level configuration for rule execution."""

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
        try:
            return super().model_validate(data)
        except _PydanticValidationError as exc:  # pragma: no cover - exercised indirectly
            raise ValidationError(str(exc)) from exc

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:  # type: ignore[override]
        return super().model_dump()

    @classmethod
    def minimal(cls) -> "RulesV2":
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
    """Summary metrics for the latest run."""

    model_config = ConfigDict(extra="forbid")

    scanned_messages: int
    matched_messages: int
    actions_applied: int
    errors: int
    warnings: int


class RuleMetrics(BaseModel):
    """Per-rule execution metrics."""

    model_config = ConfigDict(extra="forbid")

    matches: int
    actions: int
    errors: int


class LearningMetrics(BaseModel):
    """Metrics emitted by the learning pipeline."""

    model_config = ConfigDict(extra="forbid")

    last_train_started_at: Optional[str] = None
    last_train_finished_at: Optional[str] = None
    samples_used: int = 0
    classes: List[str] = Field(default_factory=list)
    macro_f1: Optional[float] = None
    proposed_rules: int = 0
    delete_semantics: Dict[str, int] = Field(default_factory=dict)


class PrivacyStatus(BaseModel):
    """Operational state of privacy safeguards."""

    model_config = ConfigDict(extra="forbid")

    feature_store_encrypted: bool = True
    plaintext_leaks_detected: int = 0
    pepper_rotation_due: bool = False


class Proposal(BaseModel):
    """Learner generated rule proposal."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    diff: str
    why: str


class ConfigReference(BaseModel):
    """Reference pointing to the active configuration message."""

    model_config = ConfigDict(extra="forbid")

    uid: int
    message_id: Optional[str]
    internaldate: str
    checksum: str


class StatusEvent(BaseModel):
    """Timeline event describing configuration activity."""

    model_config = ConfigDict(extra="forbid")

    ts: str
    type: Literal["config_updated", "config_invalid", "config_restored"]
    details: str


class StatusV2(BaseModel):
    """Status snapshot emitted by MailAI."""

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
        try:
            return super().model_validate(data)
        except _PydanticValidationError as exc:  # pragma: no cover - exercised indirectly
            raise ValidationError(str(exc)) from exc

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:  # type: ignore[override]
        return super().model_dump()

    @classmethod
    def minimal(cls) -> "StatusV2":
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

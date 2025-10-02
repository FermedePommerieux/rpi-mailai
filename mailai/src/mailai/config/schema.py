"""Lightweight schema validation for MailAI configuration documents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ValidationError(ValueError):
    """Raised when configuration data does not satisfy the schema."""


def _ensure_keys(name: str, data: Dict[str, Any], allowed: List[str]) -> None:
    extra = set(data) - set(allowed)
    if extra:
        raise ValidationError(f"{name}: unexpected keys {sorted(extra)}")


def _require(data: Dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValidationError(f"Missing required key '{key}'")
    return data[key]


def _ensure_type(name: str, value: Any, expected_type: type) -> Any:
    if not isinstance(value, expected_type):
        raise ValidationError(f"{name} expected {expected_type.__name__}")
    return value


def _ensure_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} expected bool")
    return value


def _ensure_list(name: str, value: Any) -> List[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} expected list")
    return value


def _ensure_dict(name: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} expected mapping")
    return value


def _ensure_enum(name: str, value: Any, *, allowed: List[str]) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} expected string from {allowed}")
    if value not in allowed:
        raise ValidationError(f"{name} must be one of {allowed}")
    return value


def _ensure_number(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} expected number")
    return float(value)


def _condition_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    if len(data) != 1:
        raise ValidationError("condition must have exactly one entry")
    field, payload = next(iter(data.items()))
    if field not in {
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
    }:
        raise ValidationError(f"unsupported condition field '{field}'")
    if isinstance(payload, dict):
        return {field: dict(payload)}
    return {field: payload}


def _parse_actions(actions: List[Any]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for item in actions:
        mapping = _ensure_dict("action", item)
        if len(mapping) != 1:
            raise ValidationError("action must contain exactly one operation")
        parsed.append(dict(mapping))
    return parsed


def _parse_conditions(name: str, values: Optional[List[Any]]) -> List[Dict[str, Any]]:
    if values is None:
        return []
    parsed: List[Dict[str, Any]] = []
    for item in values:
        condition = _ensure_dict("condition", item)
        parsed.append(_condition_from_dict(condition))
    return parsed


@dataclass
class Metadata:
    description: str
    owner: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metadata":
        _ensure_keys("meta", data, ["description", "owner", "updated_at"])
        return cls(
            description=_ensure_type("meta.description", _require(data, "description"), str),
            owner=_ensure_type("meta.owner", _require(data, "owner"), str),
            updated_at=_ensure_type("meta.updated_at", _require(data, "updated_at"), str),
        )


@dataclass
class Schedule:
    learn_cron: str
    inference_interval_s: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schedule":
        _ensure_keys("schedule", data, ["learn_cron", "inference_interval_s"])
        return cls(
            learn_cron=_ensure_type("schedule.learn_cron", _require(data, "learn_cron"), str),
            inference_interval_s=int(
                _ensure_type("schedule.inference_interval_s", _require(data, "inference_interval_s"), int)
            ),
        )


@dataclass
class EncryptionConfig:
    enabled: bool
    key_path: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptionConfig":
        _ensure_keys("encryption", data, ["enabled", "key_path"])
        return cls(
            enabled=_ensure_bool("encryption.enabled", data.get("enabled", True)),
            key_path=_ensure_type("encryption.key_path", _require(data, "key_path"), str),
        )


@dataclass
class PrivacyConfig:
    feature_store_path: str
    encryption: EncryptionConfig
    hashing_pepper_path: str
    hashing_salt_path: str
    max_plaintext_window_chars: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrivacyConfig":
        _ensure_keys(
            "privacy",
            data,
            [
                "feature_store_path",
                "encryption",
                "hashing_pepper_path",
                "hashing_salt_path",
                "max_plaintext_window_chars",
            ],
        )
        return cls(
            feature_store_path=_ensure_type(
                "privacy.feature_store_path", _require(data, "feature_store_path"), str
            ),
            encryption=EncryptionConfig.from_dict(_ensure_dict("privacy.encryption", _require(data, "encryption"))),
            hashing_pepper_path=_ensure_type(
                "privacy.hashing_pepper_path", _require(data, "hashing_pepper_path"), str
            ),
            hashing_salt_path=_ensure_type(
                "privacy.hashing_salt_path", _require(data, "hashing_salt_path"), str
            ),
            max_plaintext_window_chars=int(
                _ensure_type(
                    "privacy.max_plaintext_window_chars",
                    _require(data, "max_plaintext_window_chars"),
                    int,
                )
            ),
        )


@dataclass
class LLMConfig:
    provider: str
    model: str
    max_tokens: int
    temperature: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        _ensure_keys("llm", data, ["provider", "model", "max_tokens", "temperature"])
        return cls(
            provider=_ensure_type("llm.provider", _require(data, "provider"), str),
            model=_ensure_type("llm.model", _require(data, "model"), str),
            max_tokens=int(_ensure_number("llm.max_tokens", _require(data, "max_tokens"))),
            temperature=float(_ensure_number("llm.temperature", _require(data, "temperature"))),
        )


@dataclass
class EmbeddingConfig:
    enabled: bool
    backend: Optional[str]
    dim: Optional[int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingConfig":
        _ensure_keys("embeddings", data, ["enabled", "backend", "dim"])
        enabled = _ensure_bool("embeddings.enabled", data.get("enabled", False))
        backend = data.get("backend") if enabled else None
        if enabled and not isinstance(backend, str):
            raise ValidationError("embeddings.backend required when enabled")
        dim = data.get("dim") if enabled else None
        if enabled and not isinstance(dim, int):
            raise ValidationError("embeddings.dim required when enabled")
        return cls(enabled=enabled, backend=backend, dim=dim)


@dataclass
class RuleSynthesisConfig:
    enabled: bool
    max_rules_per_pass: int
    require_user_confirmation: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleSynthesisConfig":
        _ensure_keys("rule_synthesis", data, ["enabled", "max_rules_per_pass", "require_user_confirmation"])
        return cls(
            enabled=_ensure_bool("rule_synthesis.enabled", data.get("enabled", True)),
            max_rules_per_pass=int(
                _ensure_number("rule_synthesis.max_rules_per_pass", data.get("max_rules_per_pass", 3))
            ),
            require_user_confirmation=_ensure_bool(
                "rule_synthesis.require_user_confirmation", data.get("require_user_confirmation", True)
            ),
        )


@dataclass
class DeleteSemanticsConfig:
    infer_meaning: bool
    signals: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeleteSemanticsConfig":
        _ensure_keys("delete_semantics", data, ["infer_meaning", "signals"])
        signals = _ensure_list("delete_semantics.signals", data.get("signals", []))
        for item in signals:
            _ensure_type("delete_semantics.signal", item, str)
        return cls(
            infer_meaning=_ensure_bool("delete_semantics.infer_meaning", data.get("infer_meaning", True)),
            signals=signals,
        )


@dataclass
class LearningConfig:
    enabled: bool
    window_days: int
    min_samples_per_class: int
    llm: LLMConfig
    embeddings: EmbeddingConfig
    rule_synthesis: RuleSynthesisConfig
    delete_semantics: DeleteSemanticsConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningConfig":
        _ensure_keys(
            "learning",
            data,
            [
                "enabled",
                "window_days",
                "min_samples_per_class",
                "llm",
                "embeddings",
                "rule_synthesis",
                "delete_semantics",
            ],
        )
        return cls(
            enabled=_ensure_bool("learning.enabled", data.get("enabled", True)),
            window_days=int(_ensure_number("learning.window_days", _require(data, "window_days"))),
            min_samples_per_class=int(
                _ensure_number("learning.min_samples_per_class", _require(data, "min_samples_per_class"))
            ),
            llm=LLMConfig.from_dict(_ensure_dict("learning.llm", _require(data, "llm"))),
            embeddings=EmbeddingConfig.from_dict(_ensure_dict("learning.embeddings", _require(data, "embeddings"))),
            rule_synthesis=RuleSynthesisConfig.from_dict(
                _ensure_dict("learning.rule_synthesis", _require(data, "rule_synthesis"))
            ),
            delete_semantics=DeleteSemanticsConfig.from_dict(
                _ensure_dict("learning.delete_semantics", _require(data, "delete_semantics"))
            ),
        )


@dataclass
class DefaultsConfig:
    case_sensitive: bool
    stop_on_first_match: bool
    dry_run: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DefaultsConfig":
        _ensure_keys("defaults", data, ["case_sensitive", "stop_on_first_match", "dry_run"])
        return cls(
            case_sensitive=_ensure_bool("defaults.case_sensitive", data.get("case_sensitive", False)),
            stop_on_first_match=_ensure_bool("defaults.stop_on_first_match", data.get("stop_on_first_match", False)),
            dry_run=_ensure_bool("defaults.dry_run", data.get("dry_run", True)),
        )


@dataclass
class RuleMatch:
    any: List[Dict[str, Any]] = field(default_factory=list)
    all: List[Dict[str, Any]] = field(default_factory=list)
    none: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleMatch":
        _ensure_keys("match", data, ["any", "all", "none"])
        if not any(data.get(key) for key in ("any", "all", "none")):
            raise ValidationError("match must define at least one clause")
        return cls(
            any=_parse_conditions("match.any", data.get("any")),
            all=_parse_conditions("match.all", data.get("all")),
            none=_parse_conditions("match.none", data.get("none")),
        )


@dataclass
class Rule:
    id: str
    description: str
    why: str
    source: str
    enabled: bool
    priority: int
    match: RuleMatch
    actions: List[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        _ensure_keys(
            "rule",
            data,
            ["id", "description", "why", "source", "enabled", "priority", "match", "actions"],
        )
        return cls(
            id=_ensure_type("rule.id", _require(data, "id"), str),
            description=_ensure_type("rule.description", _require(data, "description"), str),
            why=_ensure_type("rule.why", _require(data, "why"), str),
            source=_ensure_enum("rule.source", _require(data, "source"), allowed=["deterministic", "learner"]),
            enabled=_ensure_bool("rule.enabled", data.get("enabled", True)),
            priority=int(_ensure_number("rule.priority", _require(data, "priority"))),
            match=RuleMatch.from_dict(_ensure_dict("rule.match", _require(data, "match"))),
            actions=_parse_actions(_ensure_list("rule.actions", _require(data, "actions"))),
        )


def _default_metadata() -> Metadata:
    return Metadata(
        description="Default MailAI policy",
        owner="mailai",
        updated_at="1970-01-01T00:00:00Z",
    )


def _default_schedule() -> Schedule:
    return Schedule(learn_cron="0 3 * * *", inference_interval_s=900)


def _default_privacy() -> PrivacyConfig:
    return PrivacyConfig(
        feature_store_path="/var/lib/mailai/features.db",
        encryption=EncryptionConfig(enabled=True, key_path="/run/secrets/sqlcipher_key"),
        hashing_pepper_path="/run/secrets/account_pepper",
        hashing_salt_path="/run/secrets/global_hash_salt",
        max_plaintext_window_chars=2048,
    )


def _default_learning() -> LearningConfig:
    return LearningConfig(
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
    )


def _default_defaults() -> DefaultsConfig:
    return DefaultsConfig(case_sensitive=False, stop_on_first_match=False, dry_run=True)


def _default_rules() -> List[Rule]:
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
    return [baseline_rule]


@dataclass
class RulesV2:
    version: int
    meta: Metadata
    schedule: Schedule
    privacy: PrivacyConfig
    learning: LearningConfig
    defaults: DefaultsConfig
    rules: List[Rule]

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "RulesV2":
        _ensure_keys(
            "rules.yaml",
            data,
            ["version", "meta", "schedule", "privacy", "learning", "defaults", "rules"],
        )
        version = int(_ensure_number("version", _require(data, "version")))
        if version != 2:
            raise ValidationError("version must be 2")
        rules_data = _ensure_list("rules", data.get("rules", []))
        rules = [Rule.from_dict(_ensure_dict("rule", item)) for item in rules_data]
        return cls(
            version=version,
            meta=Metadata.from_dict(_ensure_dict("meta", _require(data, "meta"))),
            schedule=Schedule.from_dict(_ensure_dict("schedule", _require(data, "schedule"))),
            privacy=PrivacyConfig.from_dict(_ensure_dict("privacy", _require(data, "privacy"))),
            learning=LearningConfig.from_dict(_ensure_dict("learning", _require(data, "learning"))),
            defaults=DefaultsConfig.from_dict(_ensure_dict("defaults", _require(data, "defaults"))),
            rules=rules,
        )

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:
        return {
            "version": self.version,
            "meta": self.meta.__dict__,
            "schedule": self.schedule.__dict__,
            "privacy": {
                "feature_store_path": self.privacy.feature_store_path,
                "encryption": self.privacy.encryption.__dict__,
                "hashing_pepper_path": self.privacy.hashing_pepper_path,
                "hashing_salt_path": self.privacy.hashing_salt_path,
                "max_plaintext_window_chars": self.privacy.max_plaintext_window_chars,
            },
            "learning": {
                "enabled": self.learning.enabled,
                "window_days": self.learning.window_days,
                "min_samples_per_class": self.learning.min_samples_per_class,
                "llm": self.learning.llm.__dict__,
                "embeddings": self.learning.embeddings.__dict__,
                "rule_synthesis": self.learning.rule_synthesis.__dict__,
                "delete_semantics": {
                    "infer_meaning": self.learning.delete_semantics.infer_meaning,
                    "signals": list(self.learning.delete_semantics.signals),
                },
            },
            "defaults": self.defaults.__dict__,
            "rules": [
                {
                    "id": rule.id,
                    "description": rule.description,
                    "source": rule.source,
                    "enabled": rule.enabled,
                    "priority": rule.priority,
                    "match": {
                        key: getattr(rule.match, key)
                        for key in ["any", "all", "none"]
                        if getattr(rule.match, key)
                    },
                    "actions": rule.actions,
                    "why": rule.why,
                }
                for rule in self.rules
            ],
        }

    @classmethod
    def minimal(cls) -> "RulesV2":
        return cls(
            version=2,
            meta=_default_metadata(),
            schedule=_default_schedule(),
            privacy=_default_privacy(),
            learning=_default_learning(),
            defaults=_default_defaults(),
            rules=_default_rules(),
        )


@dataclass
class StatusSummary:
    scanned_messages: int
    matched_messages: int
    actions_applied: int
    errors: int
    warnings: int


@dataclass
class RuleMetrics:
    matches: int
    actions: int
    errors: int


@dataclass
class Proposal:
    rule_id: str
    diff: str
    why: str


@dataclass
class LearningMetrics:
    last_train_started_at: Optional[str]
    last_train_finished_at: Optional[str]
    samples_used: int
    classes: List[str]
    macro_f1: Optional[float]
    proposed_rules: int
    delete_semantics: Dict[str, int]


@dataclass
class PrivacyStatus:
    feature_store_encrypted: bool
    plaintext_leaks_detected: int
    pepper_rotation_due: bool


@dataclass
class StatusV2:
    run_id: str
    config_checksum: str
    mailbox: str
    last_run_started_at: str
    last_run_finished_at: Optional[str]
    mode: Dict[str, bool]
    summary: StatusSummary
    by_rule: Dict[str, RuleMetrics]
    learning: LearningMetrics
    privacy: PrivacyStatus
    notes: List[str]
    proposals: List[Proposal]

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "StatusV2":
        _ensure_keys(
            "status.yaml",
            data,
            [
                "run_id",
                "config_checksum",
                "mailbox",
                "last_run_started_at",
                "last_run_finished_at",
                "mode",
                "summary",
                "by_rule",
                "learning",
                "privacy",
                "notes",
                "proposals",
            ],
        )
        summary_dict = _ensure_dict("summary", _require(data, "summary"))
        summary = StatusSummary(
            scanned_messages=int(_ensure_number("summary.scanned_messages", _require(summary_dict, "scanned_messages"))),
            matched_messages=int(_ensure_number("summary.matched_messages", _require(summary_dict, "matched_messages"))),
            actions_applied=int(_ensure_number("summary.actions_applied", _require(summary_dict, "actions_applied"))),
            errors=int(_ensure_number("summary.errors", _require(summary_dict, "errors"))),
            warnings=int(_ensure_number("summary.warnings", _require(summary_dict, "warnings"))),
        )
        by_rule_dict = _ensure_dict("by_rule", data.get("by_rule", {}))
        rule_metrics = {
            key: RuleMetrics(
                matches=int(_ensure_number(f"by_rule.{key}.matches", _require(value, "matches"))),
                actions=int(_ensure_number(f"by_rule.{key}.actions", _require(value, "actions"))),
                errors=int(_ensure_number(f"by_rule.{key}.errors", _require(value, "errors"))),
            )
            for key, value in by_rule_dict.items()
        }
        learning_dict = _ensure_dict("learning", _require(data, "learning"))
        learning = LearningMetrics(
            last_train_started_at=learning_dict.get("last_train_started_at"),
            last_train_finished_at=learning_dict.get("last_train_finished_at"),
            samples_used=int(_ensure_number("learning.samples_used", learning_dict.get("samples_used", 0))),
            classes=[_ensure_type("learning.classes", item, str) for item in learning_dict.get("classes", [])],
            macro_f1=learning_dict.get("macro_f1"),
            proposed_rules=int(_ensure_number("learning.proposed_rules", learning_dict.get("proposed_rules", 0))),
            delete_semantics={
                key: int(_ensure_number(f"learning.delete_semantics.{key}", value))
                for key, value in learning_dict.get("delete_semantics", {}).items()
            },
        )
        privacy_dict = _ensure_dict("privacy", _require(data, "privacy"))
        privacy = PrivacyStatus(
            feature_store_encrypted=_ensure_bool(
                "privacy.feature_store_encrypted", privacy_dict.get("feature_store_encrypted", True)
            ),
            plaintext_leaks_detected=int(
                _ensure_number("privacy.plaintext_leaks_detected", privacy_dict.get("plaintext_leaks_detected", 0))
            ),
            pepper_rotation_due=_ensure_bool("privacy.pepper_rotation_due", privacy_dict.get("pepper_rotation_due", False)),
        )
        proposals: List[Proposal] = []
        for item in data.get("proposals", []):
            mapping = _ensure_dict("proposals", item)
            _ensure_keys("proposal", mapping, ["rule_id", "diff", "why"])
            proposals.append(
                Proposal(
                    rule_id=_ensure_type("proposal.rule_id", _require(mapping, "rule_id"), str),
                    diff=_ensure_type("proposal.diff", _require(mapping, "diff"), str),
                    why=_ensure_type("proposal.why", _require(mapping, "why"), str),
                )
            )
        return cls(
            run_id=_ensure_type("run_id", _require(data, "run_id"), str),
            config_checksum=_ensure_type("config_checksum", _require(data, "config_checksum"), str),
            mailbox=_ensure_type("mailbox", _require(data, "mailbox"), str),
            last_run_started_at=_ensure_type("last_run_started_at", _require(data, "last_run_started_at"), str),
            last_run_finished_at=data.get("last_run_finished_at"),
            mode={str(k): bool(v) for k, v in _ensure_dict("mode", data.get("mode", {})).items()},
            summary=summary,
            by_rule=rule_metrics,
            learning=learning,
            privacy=privacy,
            notes=[_ensure_type("notes", item, str) for item in data.get("notes", [])],
            proposals=proposals,
        )

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_checksum": self.config_checksum,
            "mailbox": self.mailbox,
            "last_run_started_at": self.last_run_started_at,
            "last_run_finished_at": self.last_run_finished_at,
            "mode": dict(self.mode),
            "summary": self.summary.__dict__,
            "by_rule": {key: value.__dict__ for key, value in self.by_rule.items()},
            "learning": {
                "last_train_started_at": self.learning.last_train_started_at,
                "last_train_finished_at": self.learning.last_train_finished_at,
                "samples_used": self.learning.samples_used,
                "classes": list(self.learning.classes),
                "macro_f1": self.learning.macro_f1,
                "proposed_rules": self.learning.proposed_rules,
                "delete_semantics": dict(self.learning.delete_semantics),
            },
            "privacy": {
                "feature_store_encrypted": self.privacy.feature_store_encrypted,
                "plaintext_leaks_detected": self.privacy.plaintext_leaks_detected,
                "pepper_rotation_due": self.privacy.pepper_rotation_due,
            },
            "notes": list(self.notes),
            "proposals": [
                {"rule_id": item.rule_id, "diff": item.diff, "why": item.why}
                for item in self.proposals
            ],
        }

    @classmethod
    def minimal(cls) -> "StatusV2":
        summary = StatusSummary(
            scanned_messages=0,
            matched_messages=0,
            actions_applied=0,
            errors=0,
            warnings=0,
        )
        learning = LearningMetrics(
            last_train_started_at=None,
            last_train_finished_at=None,
            samples_used=0,
            classes=[],
            macro_f1=None,
            proposed_rules=0,
            delete_semantics={},
        )
        privacy = PrivacyStatus(
            feature_store_encrypted=True,
            plaintext_leaks_detected=0,
            pepper_rotation_due=False,
        )
        return cls(
            run_id="bootstrap",
            config_checksum="",
            mailbox="",
            last_run_started_at="",
            last_run_finished_at=None,
            mode={},
            summary=summary,
            by_rule={},
            learning=learning,
            privacy=privacy,
            notes=[],
            proposals=[],
        )

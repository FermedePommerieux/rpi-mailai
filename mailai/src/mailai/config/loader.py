"""Utilities for loading MailAI configuration documents."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from pydantic import ValidationError as _PydanticValidationError

from . import yamlshim
from .schema import (
    RulesV2,
    RuntimeConfig,
    StatusV2,
    ValidationError,
)


class ConfigLoadError(Exception):
    """Raised when configuration parsing or validation fails."""


class RuntimeConfigError(ConfigLoadError):
    """Raised when the runtime configuration cannot be loaded."""


class YamlValidationError(ConfigLoadError):
    """Backward compatible alias for :class:`ConfigLoadError`."""


@dataclass
class LoadedDocument:
    """Wrapper containing both the parsed model and raw YAML text."""

    model: Any
    raw: str
    checksum: str


_CONFIG_ENV = "MAILAI_CONFIG_PATH"
_DEFAULT_LOCATIONS: Tuple[Path, ...] = (
    Path("config.cfg"),
    Path("/etc/mailai/config.cfg"),
    Path("/var/lib/mailai/config.cfg"),
)
_RUNTIME_CACHE: Optional[Tuple[Path, RuntimeConfig]] = None


def _candidate_paths(path: Optional[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    if path is not None:
        candidate = path.expanduser()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate
    env_path = os.environ.get(_CONFIG_ENV)
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate
    for default in _DEFAULT_LOCATIONS:
        candidate = default.expanduser()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def _parse_config_payload(text: str, source: Path) -> dict[str, Any]:
    stripped = text.lstrip()
    if source.suffix.lower() == ".json" or stripped.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeConfigError(f"Invalid JSON in {source}: {exc}") from exc
    else:
        try:
            payload = yamlshim.load(text.encode("utf-8")) or {}
        except Exception as exc:
            raise RuntimeConfigError(f"Invalid YAML in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeConfigError("config.cfg must contain a mapping at the top-level")
    return payload


def _load_runtime_from_path(path: Path) -> RuntimeConfig:
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise RuntimeConfigError(f"Configuration file missing: {path}") from exc
    except OSError as exc:  # pragma: no cover - filesystem surface
        raise RuntimeConfigError(f"Unable to read configuration file {path}: {exc}") from exc
    payload = _parse_config_payload(text, path)
    try:
        return RuntimeConfig.model_validate(payload)
    except _PydanticValidationError as exc:
        raise RuntimeConfigError(f"Invalid config.cfg: {exc}") from exc


def load_runtime_config(
    path: Optional[Path | str] = None,
    *,
    reload: bool = False,
) -> RuntimeConfig:
    """Load the runtime configuration from ``config.cfg``."""

    global _RUNTIME_CACHE

    requested_path = Path(path).expanduser() if isinstance(path, (str, Path)) else None
    if not reload and _RUNTIME_CACHE is not None:
        cached_path, cached_config = _RUNTIME_CACHE
        if requested_path is None or cached_path == requested_path:
            return cached_config

    errors: list[str] = []
    for candidate in _candidate_paths(requested_path):
        if not candidate.exists():
            errors.append(str(candidate))
            continue
        config = _load_runtime_from_path(candidate)
        _RUNTIME_CACHE = (candidate, config)
        return config

    searched = ", ".join(errors) if errors else "<none>"
    raise RuntimeConfigError(f"Unable to locate config.cfg (searched: {searched})")


def get_runtime_config() -> RuntimeConfig:
    """Return the cached runtime configuration, loading it if required."""

    return load_runtime_config()


def reset_runtime_config() -> None:
    """Clear the runtime configuration cache (useful in tests)."""

    global _RUNTIME_CACHE
    _RUNTIME_CACHE = None


def parse_and_validate(yaml_text: str) -> RulesV2:
    """Parse and validate a rules YAML document returning the pydantic model."""

    try:
        payload = yamlshim.load(yaml_text.encode("utf-8")) or {}
    except Exception as exc:  # pragma: no cover - passthrough for context
        raise ConfigLoadError(f"Invalid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigLoadError("rules.yaml must contain a mapping at the top-level")
    try:
        return RulesV2.model_validate(payload)
    except ValidationError as exc:
        raise ConfigLoadError(str(exc)) from exc


def _checksum(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def load_rules(source: bytes) -> LoadedDocument:
    """Load a rules document from bytes and validate it."""

    text = source.decode("utf-8")
    model = parse_and_validate(text)
    return LoadedDocument(model=model, raw=text, checksum=_checksum(text))


def load_status(source: bytes) -> LoadedDocument:
    """Load and validate a status document."""

    try:
        payload = yamlshim.load(source) or {}
    except Exception as exc:  # pragma: no cover - passthrough for context
        raise ConfigLoadError(f"Invalid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigLoadError("status.yaml must contain a mapping at the top-level")
    try:
        model = StatusV2.model_validate(payload)
    except ValidationError as exc:
        raise ConfigLoadError(str(exc)) from exc
    text = yamlshim.dump(model.model_dump(mode="json"))
    return LoadedDocument(model=model, raw=text, checksum=_checksum(text))


def dump_rules(model: RulesV2) -> bytes:
    """Serialise a rules model back into YAML bytes."""

    return yamlshim.dump(model.model_dump(mode="json")).encode("utf-8")


def dump_status(model: StatusV2) -> bytes:
    """Serialise a status model back into YAML bytes."""

    return yamlshim.dump(model.model_dump(mode="json")).encode("utf-8")

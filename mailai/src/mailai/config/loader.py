"""Utilities for loading and validating YAML configuration files."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .schema import RulesV2, StatusV2, ValidationError
from . import yamlshim


@dataclass
class LoadedDocument:
    """Wrapper containing both the parsed model and raw YAML bytes."""

    model: Any
    raw: bytes
    checksum: str


class YamlValidationError(ValueError):
    """Raised when YAML fails validation against the Pydantic schema."""


def _load_yaml_document(source: bytes) -> Any:
    try:
        return yamlshim.load(source) or {}
    except Exception as exc:  # pragma: no cover - passthrough for context
        raise YamlValidationError(f"Invalid YAML: {exc}") from exc


def _checksum(data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def load_rules(source: bytes) -> LoadedDocument:
    """Load and validate a rules document."""

    payload = _load_yaml_document(source)
    try:
        model = RulesV2.model_validate(payload)
    except ValidationError as exc:
        raise YamlValidationError(str(exc)) from exc
    return LoadedDocument(model=model, raw=source, checksum=_checksum(source))


def load_status(source: bytes) -> LoadedDocument:
    """Load and validate a status document."""

    payload = _load_yaml_document(source)
    try:
        model = StatusV2.model_validate(payload)
    except ValidationError as exc:
        raise YamlValidationError(str(exc)) from exc
    return LoadedDocument(model=model, raw=source, checksum=_checksum(source))


def dump_rules(model: RulesV2) -> bytes:
    """Serialise a rules model back into YAML bytes."""

    return yamlshim.dump(model.model_dump(mode="json")).encode("utf-8")


def dump_status(model: StatusV2) -> bytes:
    """Serialise a status model back into YAML bytes."""

    return yamlshim.dump(model.model_dump(mode="json")).encode("utf-8")

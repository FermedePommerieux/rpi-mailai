"""Strict loaders and serializers for MailAI configuration documents.

What:
  Provide helpers to locate, parse, validate, and serialise the runtime
  configuration files (``config.yaml``, ``rules.yaml``, ``status.yaml``).

Why:
  Configuration lives outside the application bundle and can be malformed or
  tampered with. Centralising the parsing logic enforces consistent validation
  and hashing so that downstream services can trust the resulting models.

How:
  Resolve candidate file locations based on explicit parameters, environment
  variables, and defaults. Parse YAML payloads through the local YAML shim,
  validate them using Pydantic models, and expose deterministic hashing to track
  changes across IMAP synchronisation cycles.

Interfaces:
  - :func:`load_runtime_config` / :func:`get_runtime_config` /
    :func:`reset_runtime_config`: Manage ``config.yaml`` discovery and caching.
  - :func:`load_rules` / :func:`load_status`: Parse YAML payloads embedded in
    IMAP messages into typed models.
  - :func:`dump_rules` / :func:`dump_status`: Serialise typed models back into
    YAML bytes for persistence.
  - :class:`LoadedDocument`: Bundle parsed models with their raw text and
    checksums for auditability.

Invariants:
  - All external payloads must pass strict Pydantic validation before they are
    returned to callers.
  - The runtime configuration cache respects explicit reload requests and the
    precedence order of candidate paths.

Safety/Performance:
  - Hash computations use SHA-256 to uniquely identify IMAP payload revisions
    without storing plaintext beyond the minimal processing window.
  - File operations avoid silent failures by converting OS errors into typed
    exceptions that include path context.
"""

from __future__ import annotations

import hashlib
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
    """Base error for configuration parsing or validation failures.

    What:
      Represent fatal issues encountered while reading or validating
      configuration documents.

    Why:
      Grouping failures under a single type allows callers to handle user input
      mistakes separately from infrastructure errors such as IMAP connectivity
      problems.

    How:
      Derive from :class:`Exception` and rely on callers to include contextual
      messages describing the failing document.
    """


class RuntimeConfigError(ConfigLoadError):
    """Error raised when ``config.yaml`` cannot be loaded or validated.

    What:
      Signal issues related specifically to runtime configuration discovery or
      schema validation.

    Why:
      Differentiating runtime configuration failures helps CLI tools display
      targeted remediation steps without conflating them with rules/status
      parsing issues.

    How:
      Subclass :class:`ConfigLoadError` so upstream handlers can catch the broad
      category or the specialised variant as needed.
    """


class YamlValidationError(ConfigLoadError):
    """Backward-compatible alias for :class:`ConfigLoadError`.

    What:
      Preserve the historical exception name used by earlier releases.

    Why:
      Existing automation and tests expect this symbol; keeping it avoids a
      breaking change while forwarding to the new hierarchy.

    How:
      Inherit directly from :class:`ConfigLoadError` without adding behaviour.
    """


@dataclass
class LoadedDocument:
    """Bundle parsed configuration models with their raw representation.

    What:
      Provide a structured return type that includes the Pydantic model, the raw
      textual payload, and a checksum capturing its contents.

    Why:
      Downstream components (such as IMAP sync and auditing) need both the typed
      data and the exact bytes to detect tampering or reconstruct emails.

    How:
      Store the fields as dataclass attributes so callers can access them
      directly without extra wrappers.

    Attributes:
      model: The validated Pydantic model.
      raw: The canonical text representation of the payload.
      checksum: SHA-256 checksum prefixed with ``sha256:`` for log correlation.
    """

    model: Any
    raw: str
    checksum: str


_CONFIG_ENV = "MAILAI_CONFIG_PATH"
_DEFAULT_LOCATIONS: Tuple[Path, ...] = (
    Path("config.yaml"),
    Path("/etc/mailai/config.yaml"),
    Path("/var/lib/mailai/config.yaml"),
)
_RUNTIME_CACHE: Optional[Tuple[Path, RuntimeConfig]] = None


def _candidate_paths(path: Optional[Path]) -> Iterable[Path]:
    """Yield configuration file locations in priority order.

    What:
      Produce the ordered list of paths that should be inspected for
      ``config.yaml``.

    Why:
      The runtime allows operators to override the configuration path through a
      function argument, environment variable, or well-known defaults; this
      helper captures that precedence chain.

    How:
      Accumulate deduplicated :class:`~pathlib.Path` objects by checking the
      explicit argument, the ``MAILAI_CONFIG_PATH`` environment variable, and the
      default locations. Paths are expanded to handle ``~`` resolutions.

    Args:
      path: Explicit path requested by the caller, or ``None`` to rely on
        environment/defaults.

    Yields:
      Candidate paths ordered from most specific to least specific.
    """

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
        # NOTE: Deduplicate while preserving the user-visible precedence order.
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def _parse_config_payload(text: str, source: Path) -> dict[str, Any]:
    """Parse ``config.yaml`` text into a dictionary payload.

    What:
      Convert raw configuration text to a Python mapping ready for validation.

    Why:
      ``config.yaml`` is the canonical runtime document. Centralising YAML
      parsing enforces consistent error messages and prevents callers from
      bypassing schema checks with alternative formats.

    How:
      Delegate to the YAML shim for decoding, wrap any parsing issues in
      :class:`RuntimeConfigError` that includes file context, and verify the
      resulting object is a mapping before returning it.

    Args:
      text: Raw configuration contents.
      source: Path to the file being parsed (used for diagnostics).

    Returns:
      A dictionary representing the configuration payload.

    Raises:
      RuntimeConfigError: If the file cannot be parsed or does not contain a
      mapping.
    """

    try:
        payload = yamlshim.load(text.encode("utf-8")) or {}
    except Exception as exc:
        raise RuntimeConfigError(f"Invalid YAML in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeConfigError("config.yaml must contain a mapping at the top-level")
    return payload


def _load_runtime_from_path(path: Path) -> RuntimeConfig:
    """Load and validate ``config.yaml`` from a specific path.

    What:
      Read the file at ``path`` and convert it into a validated
      :class:`RuntimeConfig` model.

    Why:
      Splitting the functionality keeps :func:`load_runtime_config` focused on
      path discovery while this helper handles IO and schema validation.

    How:
      Read the file contents, parse them via :func:`_parse_config_payload`, and
      validate using :meth:`RuntimeConfig.model_validate`. Wrap filesystem or
      validation failures in :class:`RuntimeConfigError` with descriptive
      messages.

    Args:
      path: Filesystem location of the runtime configuration.

    Returns:
      The validated :class:`RuntimeConfig` model.

    Raises:
      RuntimeConfigError: If the file cannot be read or fails validation.
    """

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
        raise RuntimeConfigError(f"Invalid config.yaml: {exc}") from exc


def load_runtime_config(
    path: Optional[Path | str] = None,
    *,
    reload: bool = False,
) -> RuntimeConfig:
    """Resolve, parse, and cache the runtime configuration.

    What:
      Locate ``config.yaml`` using the configured precedence chain, parse it, and
      return a validated :class:`RuntimeConfig` instance.

    Why:
      Many components require runtime settings; caching avoids repeated disk IO
      while ``reload`` enables deterministic refreshes during tests or config
      changes.

    How:
      Convert string paths to :class:`~pathlib.Path`, consult the global cache
      unless ``reload`` is requested, iterate through candidate paths until a
      readable file is found, and store the successful result for future calls.

    Args:
      path: Optional explicit location of ``config.yaml``.
      reload: When ``True`` forces a fresh load bypassing the cache.

    Returns:
      The validated runtime configuration.

    Raises:
      RuntimeConfigError: If no suitable configuration file can be located or
      validated.
    """

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
    raise RuntimeConfigError(f"Unable to locate config.yaml (searched: {searched})")


def get_runtime_config() -> RuntimeConfig:
    """Return the cached runtime configuration, loading it on demand.

    What:
      Provide a convenience accessor for callers that do not care about cache
      invalidation semantics.

    Why:
      Simplifies call sites by delegating to :func:`load_runtime_config` with the
      default caching behaviour.

    How:
      Call :func:`load_runtime_config` without arguments and return its result.

    Returns:
      The validated runtime configuration.
    """

    return load_runtime_config()


def reset_runtime_config() -> None:
    """Clear the runtime configuration cache.

    What:
      Reset the memoised tuple storing the last loaded configuration.

    Why:
      Test suites and long-running processes need deterministic ways to force a
      reload when configuration files change.

    How:
      Assign ``None`` to the module-level ``_RUNTIME_CACHE`` variable.
    """

    global _RUNTIME_CACHE
    _RUNTIME_CACHE = None


def parse_and_validate(yaml_text: str) -> RulesV2:
    """Convert rules YAML into a validated :class:`RulesV2` model.

    What:
      Decode YAML text, ensure it produces a mapping, and validate it using the
      strict schema while supporting the streamlined "rules-only" storage
      format.

    Why:
      The rules document originates from user-maintained email bodies; the engine
      must guard against malformed or malicious payloads before applying actions
      to messages. New deployments only persist the ``rules`` list to avoid
      confusing operators with internal defaults, so the loader needs to hydrate
      the remaining fields deterministically.

    How:
      Delegate to the YAML shim for parsing, assert the result is a dictionary,
      and branch based on the keys present. When only ``rules`` (and optionally
      ``version``) are provided, merge them into the baseline
      :meth:`RulesV2.minimal` template before validation. For legacy payloads that
      still include the full schema, fall back to the strict validation path to
      remain backwards compatible.

    Args:
      yaml_text: Raw ``rules.yaml`` contents in UTF-8 text form.

    Returns:
      A validated :class:`RulesV2` model.

    Raises:
      ConfigLoadError: If parsing fails or the document violates the schema.
    """

    try:
        payload = yamlshim.load(yaml_text.encode("utf-8")) or {}
    except Exception as exc:  # pragma: no cover - passthrough for context
        raise ConfigLoadError(f"Invalid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigLoadError("rules.yaml must contain a mapping at the top-level")
    # The streamlined format only exposes the ``rules`` list to operators.
    allowed_keys = {"rules", "version"}
    payload_keys = set(payload)
    if payload_keys.issubset(allowed_keys):
        if "rules" not in payload:
            raise ConfigLoadError("rules.yaml must define a 'rules' list")
        version = payload.get("version")
        if version is not None and version != 2:
            raise ConfigLoadError("rules.yaml version must be 2")
        baseline = RulesV2.minimal().model_dump(mode="json")
        baseline["rules"] = payload["rules"]
        try:
            return RulesV2.model_validate(baseline)
        except ValidationError as exc:
            raise ConfigLoadError(str(exc)) from exc
    try:
        return RulesV2.model_validate(payload)
    except ValidationError as exc:
        raise ConfigLoadError(str(exc)) from exc


def _checksum(text: str) -> str:
    """Return a stable SHA-256 checksum prefixing the digest with ``sha256:``.

    What:
      Provide a deterministic identifier for the supplied text.

    Why:
      The IMAP synchronisation pipeline compares checksums to detect content
      changes without storing plaintext snapshots.

    How:
      Hash the UTF-8 encoding of ``text`` using :func:`hashlib.sha256` and format
      the digest according to repository conventions.

    Args:
      text: The string to hash.

    Returns:
      A checksum string of the form ``sha256:<hex>``.
    """

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def load_rules(source: bytes) -> LoadedDocument:
    """Parse and validate a ``rules.yaml`` payload provided as bytes.

    What:
      Convert the given bytes to text, validate via :func:`parse_and_validate`,
      and package the result alongside the original text and checksum.

    Why:
      IMAP downloads deliver raw bytes; higher-level services need a consistent
      entry point to obtain validated models and metadata.

    How:
      Decode using UTF-8, validate, compute the checksum, and return a
      :class:`LoadedDocument` dataclass instance.

    Args:
      source: Raw bytes of the YAML document.

    Returns:
      A :class:`LoadedDocument` containing the model, canonical text, and
      checksum.
    """

    text = source.decode("utf-8")
    model = parse_and_validate(text)
    return LoadedDocument(model=model, raw=text, checksum=_checksum(text))


def load_status(source: bytes) -> LoadedDocument:
    """Parse and validate a ``status.yaml`` payload provided as bytes.

    What:
      Transform the status document into a :class:`StatusV2` model while
      preserving a canonical text representation for auditing.

    Why:
      Status emails track runtime activity; ensuring they conform to the schema
      prevents dashboards from operating on inconsistent data.

    How:
      Parse YAML via the shim, validate through :class:`StatusV2`, serialise the
      validated model back to canonical YAML, and return it wrapped in
      :class:`LoadedDocument` with its checksum.

    Args:
      source: Raw bytes containing the YAML document.

    Returns:
      A :class:`LoadedDocument` representing the validated status payload.

    Raises:
      ConfigLoadError: If parsing or validation fails.
    """

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
    """Serialise a :class:`RulesV2` model into canonical YAML bytes.

    What:
      Convert the validated rules model into a YAML payload suitable for
      persistence while emitting only the public ``rules`` list.

    Why:
      Operators should not see or edit internal defaults inside the IMAP control
      message. Persisting just the rule definitions keeps the document concise
      and prevents accidental edits that could desynchronise runtime settings.

    How:
      Dump the model using ``mode="json"``, extract the ``rules`` collection, and
      feed the reduced mapping to the YAML shim before encoding to UTF-8 bytes.

    Args:
      model: Validated rules configuration.

    Returns:
      Canonical YAML bytes representing only the rule list.
    """

    payload = model.model_dump(mode="json")
    rules_only = {"rules": payload.get("rules", [])}
    return yamlshim.dump(rules_only).encode("utf-8")


def dump_status(model: StatusV2) -> bytes:
    """Serialise a :class:`StatusV2` model into canonical YAML bytes.

    What:
      Produce a YAML representation of the status snapshot suitable for storage
      or transmission.

    Why:
      Maintaining canonical formatting ensures status comparisons and backups are
      deterministic.

    How:
      Mirror :func:`dump_rules` by dumping the Pydantic model with
      ``mode="json"`` before YAML serialisation and UTF-8 encoding.

    Args:
      model: Validated status snapshot.

    Returns:
      Canonical YAML bytes representing the status payload.
    """

    return yamlshim.dump(model.model_dump(mode="json")).encode("utf-8")


# TODO: Other modules require the same treatment (Quoi/Pourquoi/Comment docstrings + module header).

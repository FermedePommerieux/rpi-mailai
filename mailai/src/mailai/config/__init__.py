"""MailAI configuration orchestration package.

What:
  Provide a cohesive import surface for configuration loading, validation, and
  persistence helpers used by the runtime and diagnostic tooling.

Why:
  Centralising the exports shields callers from the internal layout while
  enforcing that only audited primitives are reachable. This guards against
  bypassing validation layers or interacting with partially documented
  utilities.

How:
  Re-export the specific loader helpers and pydantic schema classes that form
  the supported API surface. Keeping ``__all__`` explicit avoids accidental
  leakage of low-level modules and documents the dependency flow for new
  contributors.

Interfaces:
  - load_rules / dump_rules: Transform validated rules YAML to and from the
    in-memory models.
  - load_status / dump_status: Manage the ``status.yaml`` runtime document.
  - get_runtime_config / load_runtime_config / reset_runtime_config: Resolve
    ``config.cfg`` and expose a cached runtime configuration object.
  - RulesV2 / RuntimeConfig / StatusV2 / ValidationError: Pydantic models and
    error type used by the higher-level services.

Invariants:
  - Only the documented helpers are reachable through the package namespace.
  - Callers must go through the schema types to ensure strict validation before
    touching user-provided YAML/JSON payloads.

Safety/Performance:
  - Explicit exports minimise import cost on constrained Raspberry Pi systems
    and prevent tooling from instantiating heavyweight helpers unnecessarily.
"""

from .loader import (
    dump_rules,
    dump_status,
    get_runtime_config,
    load_runtime_config,
    load_rules,
    load_status,
    reset_runtime_config,
)
from .schema import RulesV2, RuntimeConfig, StatusV2, ValidationError

__all__ = [
    "load_rules",
    "load_status",
    "dump_rules",
    "dump_status",
    "get_runtime_config",
    "load_runtime_config",
    "reset_runtime_config",
    "RulesV2",
    "RuntimeConfig",
    "StatusV2",
    "ValidationError",
]

# TODO: Other modules require the same treatment (Quoi/Pourquoi/Comment docstrings + module header).

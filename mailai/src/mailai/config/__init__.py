"""Configuration schemas and loaders for MailAI."""

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

"""Configuration schemas and loaders for MailAI."""

from .loader import load_rules, load_status, dump_rules, dump_status
from .schema import RulesV2, StatusV2, ValidationError

__all__ = [
    "load_rules",
    "load_status",
    "dump_rules",
    "dump_status",
    "RulesV2",
    "StatusV2",
    "ValidationError",
]

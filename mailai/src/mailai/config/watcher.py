"""Helpers for detecting configuration changes."""
from __future__ import annotations

from typing import Optional

from .schema import ConfigReference
from ..imap.rules_mail import RulesMailRef


def has_changed(prev: Optional[ConfigReference], new: Optional[RulesMailRef]) -> bool:
    """Return ``True`` when the configuration reference has changed."""

    if new is None:
        return prev is not None
    if prev is None:
        return True
    if prev.uid != new.uid:
        return True
    return prev.checksum != new.checksum


def change_reason(prev: Optional[ConfigReference], new: Optional[RulesMailRef]) -> str:
    """Return a human readable reason for the change detection."""

    if prev is None and new is None:
        return "missing"
    if prev is None:
        return "bootstrap"
    if new is None:
        return "missing"
    if prev.uid != new.uid:
        return "uid change"
    if prev.checksum != new.checksum:
        return "checksum change"
    return "unchanged"

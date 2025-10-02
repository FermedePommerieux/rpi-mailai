"""Health checks for MailAI services."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class HealthReport:
    """Simple health report."""

    status: str
    details: Dict[str, str]


def basic_health() -> HealthReport:
    """Return a nominal health report used for diagnostics."""

    return HealthReport(status="ok", details={"engine": "ready", "learner": "idle"})

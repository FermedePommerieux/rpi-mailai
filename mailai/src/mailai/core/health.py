"""mailai.core.health

What:
  Provide lightweight health reporting structures used by diagnostics and
  monitoring endpoints to express the readiness of MailAI components.

Why:
  The platform must answer health probes quickly, even when heavier subsystems
  (LLM warmup, IMAP connectivity) operate elsewhere. A dedicated module keeps
  status reporting predictable and decoupled from business logic.

How:
  - Define a simple :class:`HealthReport` dataclass that can serialise into JSON
    or structured logs.
  - Offer helpers such as :func:`basic_health` returning canned responses for
    command-line diagnostics or unit tests.

Interfaces:
  - :class:`HealthReport` representing the health payload.
  - :func:`basic_health` generating a nominal report.

Invariants & Safety:
  - Status payloads remain small and purely informational to ensure health
    checks cannot leak sensitive message data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class HealthReport:
    """Typed payload describing component health.

    What:
      Encapsulate a status string and a key/value map with component details so
      callers can return structured health responses.

    Why:
      Centralises the schema for probe responses, preventing divergent ad-hoc
      dictionaries across CLI commands or HTTP endpoints.

    How:
      Represent the payload as a dataclass to benefit from type hints and
      predictable serialisation via ``dataclasses.asdict``.

    Attributes:
      status: High-level health indicator (e.g., ``ok`` or ``degraded``).
      details: Fine-grained component statuses suitable for logging.
    """

    status: str
    details: Dict[str, str]


def basic_health() -> HealthReport:
    """Return a best-effort optimistic health snapshot.

    What:
      Produce a :class:`HealthReport` that signals the engine and learner are
      available, intended for use when deeper subsystem checks are unnecessary.

    Why:
      Command-line utilities and smoke tests need a quick sanity indicator even
      before the full runtime (LLM, IMAP) comes online.

    How:
      Instantiate :class:`HealthReport` with ``ok`` status and static component
      descriptors that communicate readiness without querying external services.

    Returns:
      Health report declaring nominal readiness of core modules.
    """

    return HealthReport(status="ok", details={"engine": "ready", "learner": "idle"})


# TODO: Other modules require the same treatment (What/Why/How docstrings + module header).

"""MailAI logging helpers with deterministic JSON emission and redaction safeguards.

What:
  Offer a tiny facade over Python streams so every MailAI component can emit
  JSON log lines with consistent fields and automatic removal of sensitive
  payloads.

Why:
  Operational forensics on Raspberry Pi deployments rely on grepping logs. A
  structured layout keeps parsing trivial while preventing accidental leakage of
  message bodies or other user content when debugging rule execution.

How:
  Provide a :class:`JsonLogger` dataclass that accepts a target stream and
  enforces uppercase severity levels. ``extra`` dictionaries are deep-copied and
  scrubbed via a recursive redaction helper before being serialised with
  ``json.dump``.

Interfaces:
  :class:`JsonLogger`, :func:`get_logger`.

Invariants & Safety:
  - The emitted log payload always includes an ISO8601 timestamp, severity, and
    component name so downstream tooling can index entries reliably.
  - Known sensitive keys (``subject``, ``body``, ``preview``, ``snippet``) are
    replaced with ``[redacted]`` even inside nested dictionaries.
  - Streams are flushed after every write to avoid losing diagnostics on abrupt
    power loss.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


REDACTED = "[redacted]"


@dataclass
class JsonLogger:
    """Structured JSON logger with automatic redaction.

    What:
      Encapsulates the logic required to emit single-line JSON log entries that
      include timestamps, severity, a component tag, and optional supplemental
      fields.

    Why:
      Centralising structured logging avoids duplicating the redaction logic and
      guarantees a uniform schema for observability pipelines and test
      assertions.

    How:
      Stores the destination stream and component label, then exposes helper
      methods (:meth:`log`, :meth:`info`, :meth:`warning`, :meth:`error`) that
      merge a canonical payload with redacted extras before serialising the
      result using :mod:`json`.
    """

    stream: Any = field(default_factory=lambda: sys.stdout)
    component: str = "mailai"

    def log(self, level: str, message: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """Emit a structured JSON log entry.

        What:
          Serialises ``message`` and ``extra`` metadata to the configured stream
          using the MailAI log schema (``ts``, ``lvl``, ``msg``, ``component``).

        Why:
          All log entries should adhere to a predictable contract so monitoring
          dashboards and tests can parse without ad-hoc heuristics.

        How:
          Builds a dictionary with the core fields, merges a redacted copy of
          ``extra`` (if provided), writes a JSON payload, and immediately flushes
          the stream to minimise data loss on abrupt termination.

        Args:
          level: Human-readable severity (e.g., ``"info"`` or ``"error"``).
          message: Core log message.
          extra: Optional context dictionary that will be redacted recursively.
        """

        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "lvl": level.upper(),
            "msg": message,
            "component": self.component,
        }
        if extra:
            payload.update(self._redact(extra))
        json.dump(payload, self.stream, separators=(",", ":"))
        self.stream.write("\n")
        self.stream.flush()

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an informational message with structured context.

        What:
          Convenience wrapper for :meth:`log` that sets the severity to ``INFO``.

        Why:
          Keeps call sites terse while ensuring all metadata passes through the
          centralised redaction logic.

        How:
          Forwards ``message`` and keyword arguments to :meth:`log`, storing the
          kwargs as ``extra`` context.

        Args:
          message: Human-readable description of the event.
          **kwargs: Structured fields to attach to the log payload.
        """

        self.log("INFO", message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message while enforcing redaction.

        What:
          Emits a ``WARN`` level entry using the structured payload pipeline.

        Why:
          Ensures cautionary events (e.g., retries) are captured consistently so
          operators can monitor them without parsing ad-hoc strings.

        How:
          Delegates to :meth:`log` with ``level`` set to ``"WARN"`` and passes
          along keyword arguments as extra context.

        Args:
          message: Description of the warning condition.
          **kwargs: Structured metadata describing the context.
        """

        self.log("WARN", message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error entry suitable for alerting.

        What:
          Emits an ``ERROR`` level JSON payload to capture critical failures.

        Why:
          Fatal or retryable errors must produce consistent diagnostic artefacts
          for automated escalation and regression tests.

        How:
          Calls :meth:`log` with ``level`` forced to ``"ERROR"`` and propagates
          structured keyword arguments for redacted serialisation.

        Args:
          message: Summary of the failure condition.
          **kwargs: Additional fields for troubleshooting.
        """

        self.log("ERROR", message, extra=kwargs)

    @staticmethod
    def _redact(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive keys from a payload recursively.

        What:
          Produces a copy of ``data`` where predefined fields are replaced with a
          sentinel ``[redacted]`` string.

        Why:
          MailAI processes personally identifiable content. Redacting common
          fields prevents accidental disclosure when logs are collected by shared
          observability infrastructure.

        How:
          Walks the dictionary, applying the sentinel to known keys and recursing
          into nested dictionaries to ensure deep redaction while preserving
          structure for downstream parsing.

        Args:
          data: Arbitrary metadata to sanitise.

        Returns:
          A copy of ``data`` with sensitive values masked.
        """

        sensitive_keys = {"subject", "body", "preview", "snippet"}
        result: Dict[str, Any] = {}
        for key, value in data.items():
            if key in sensitive_keys:
                result[key] = REDACTED
            elif isinstance(value, dict):
                result[key] = JsonLogger._redact(value)
            else:
                result[key] = value
        return result


def get_logger(component: str) -> JsonLogger:
    """Construct a :class:`JsonLogger` for the requested component.

    What:
      Returns a ready-to-use :class:`JsonLogger` bound to ``component``.

    Why:
      Call sites should avoid instantiating :class:`JsonLogger` directly so the
      import surface remains stable and shared invariants (redaction keys,
      default stream) can evolve centrally.

    How:
      Delegates to :class:`JsonLogger` with the supplied component label and the
      default ``stdout`` stream.

    Args:
      component: Logical subsystem name to include in log payloads.

    Returns:
      Configured :class:`JsonLogger` instance.
    """

    return JsonLogger(component=component)


# TODO: Other modules in this repository still require the same What/Why/How documentation.

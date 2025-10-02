"""Structured logging utilities with automatic redaction."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


REDACTED = "[redacted]"


@dataclass
class JsonLogger:
    """A minimal JSON logger that enforces redaction policies."""

    stream: Any = field(default_factory=lambda: sys.stdout)
    component: str = "mailai"

    def log(self, level: str, message: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """Emit a structured log line."""

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
        self.log("INFO", message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.log("WARN", message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.log("ERROR", message, extra=kwargs)

    @staticmethod
    def _redact(data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact known sensitive fields from the log payload."""

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
    """Factory returning a configured logger instance."""

    return JsonLogger(component=component)

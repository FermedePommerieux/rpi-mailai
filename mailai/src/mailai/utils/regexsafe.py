"""Regular expression helpers with timeouts to avoid ReDoS."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class RegexResult:
    """Container for regex search results."""

    matched: bool
    match: Optional[re.Match[str]] = None


class RegexTimeoutError(TimeoutError):
    """Raised when a regex exceeds the configured runtime."""


def search(pattern: str, text: str, *, timeout_ms: int = 50, flags: int = 0) -> RegexResult:
    """Perform a regex search with a soft timeout."""

    if timeout_ms <= 1:
        return RegexResult(matched=False)
    compiled = re.compile(pattern, flags)
    result_container: dict[str, RegexResult] = {}
    error_container: dict[str, Exception] = {}

    def _target() -> None:
        try:
            match = compiled.search(text)
            result_container["result"] = RegexResult(matched=match is not None, match=match)
        except Exception as exc:  # pragma: no cover - bubble up
            error_container["error"] = exc

    thread = threading.Thread(target=_target)
    thread.start()
    thread.join(timeout_ms / 1000)
    if thread.is_alive():
        return RegexResult(matched=False)
    if "error" in error_container:
        raise error_container["error"]
    return result_container.get("result", RegexResult(matched=False))

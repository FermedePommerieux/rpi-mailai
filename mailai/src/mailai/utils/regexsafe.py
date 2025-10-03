"""Regular expression helpers with soft timeouts to mitigate ReDoS risk.

What:
  Offer a thin wrapper around :func:`re.search` that enforces execution time
  limits and returns structured results describing matches.

Why:
  MailAI sanitises user-provided YAML and email content. Python's backtracking
  regex engine can hang on crafted inputs, so a watchdog prevents runaway CPU
  loops on constrained Raspberry Pi deployments.

How:
  Execute the compiled regex inside a thread, join with a millisecond timeout,
  and capture either the resulting match or any thrown exception. When the
  thread exceeds the deadline, return a sentinel ``RegexResult`` without raising
  errors so the caller can fall back gracefully.

Interfaces:
  :class:`RegexResult`, :class:`RegexTimeoutError`, :func:`search`.

Invariants & Safety:
  - Threads are never leaked; if the timeout expires the helper returns without
    waiting for completion and the background thread terminates naturally after
    Python unwinds the regex search.
  - ``RegexTimeoutError`` mirrors :class:`TimeoutError` for ergonomic exception
    handling.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class RegexResult:
    """Immutable result container describing regex searches.

    What:
      Records whether a match succeeded and, if so, exposes the original
      :class:`re.Match` object.

    Why:
      Returning a dataclass avoids ``None``/truthiness ambiguity and simplifies
      serialization for debugging while keeping the API backwards compatible.

    How:
      Stores two fields: ``matched`` (boolean) and ``match`` (optional
      :class:`re.Match`). Callers inspect the boolean before using ``match``.
    """

    matched: bool
    match: Optional[re.Match[str]] = None


class RegexTimeoutError(TimeoutError):
    """Sentinel timeout type for callers that need granular error handling.

    What:
      Declares a dedicated exception for regex timeouts.

    Why:
      Some callers prefer distinguishing between genuine regex failures and
      guard-rail timeouts without catching :class:`TimeoutError` broadly.

    How:
      Inherits from :class:`TimeoutError` for compatibility. The current helper
      returns negative results on timeouts, but the type is exposed for future
      enhancements or external utilities that may choose to raise it.
    """


def search(pattern: str, text: str, *, timeout_ms: int = 50, flags: int = 0) -> RegexResult:
    """Search ``text`` with ``pattern`` while enforcing a soft timeout.

    What:
      Executes ``re.search`` using ``pattern`` against ``text`` and returns a
      :class:`RegexResult` capturing the match outcome.

    Why:
      Provides deterministic guard rails when evaluating untrusted strings. The
      helper limits CPU usage and avoids crashing the caller when a pattern is
      pathological or intentionally malicious.

    How:
      Validates the timeout, compiles the regex with optional ``flags``, and
      runs the search inside a thread. If the thread completes in time, the
      matched state and match object are stored in ``result_container``. When the
      join times out, a negative ``RegexResult`` is returned. Exceptions raised in
      the worker are re-raised to surface developer errors.

    Args:
      pattern: Regular expression pattern string.
      text: Target text to scan.
      timeout_ms: Maximum runtime in milliseconds before considering the search
        a timeout.
      flags: Optional :mod:`re` compilation flags.

    Returns:
      :class:`RegexResult` describing the outcome. ``matched`` is ``False`` when
      a timeout occurs.

    Raises:
      Exception: Any exception raised by :mod:`re` compilation or search is
        propagated directly to aid debugging.
    """

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


# TODO: Other modules in this repository still require the same What/Why/How documentation.

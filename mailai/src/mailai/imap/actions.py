"""Idempotent IMAP action helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SupportsActions(Protocol):
    """Minimal protocol that IMAP clients must satisfy."""

    def move(self, uid: int, destination: str) -> None: ...

    def copy(self, uid: int, destination: str) -> None: ...

    def add_label(self, uid: int, label: str) -> None: ...

    def mark_read(self, uid: int, read: bool) -> None: ...

    def add_flag(self, uid: int, flag: str) -> None: ...

    def set_header(self, uid: int, name: str, value: str) -> None: ...


@dataclass
class ActionRequest:
    """Action request produced by the rule engine."""

    uid: int
    name: str
    value: object | None = None


class UnsupportedActionError(ValueError):
    """Raised when the engine attempts to dispatch an unknown action."""


def execute(action: ActionRequest, *, client: SupportsActions) -> None:
    """Dispatch a single action to the IMAP client."""

    if action.name == "move_to":
        client.move(action.uid, str(action.value))
        return
    if action.name == "copy_to":
        client.copy(action.uid, str(action.value))
        return
    if action.name == "add_label":
        client.add_label(action.uid, str(action.value))
        return
    if action.name == "mark_read":
        client.mark_read(action.uid, bool(action.value))
        return
    if action.name == "add_flag":
        client.add_flag(action.uid, str(action.value))
        return
    if action.name == "set_header":
        name, value = action.value  # type: ignore[misc]
        client.set_header(action.uid, str(name), str(value))
        return
    if action.name == "stop_processing":
        return
    if action.name == "delete":
        if action.value:
            client.add_flag(action.uid, "\\Deleted")
        return
    if action.name == "forward_to":  # pragma: no cover - network side effect
        raise UnsupportedActionError(
            "Forwarding must be managed by a whitelisted SMTP relay upstream"
        )
    raise UnsupportedActionError(f"Unsupported action {action.name}")

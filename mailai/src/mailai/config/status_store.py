"""Persistence helpers for `status.yaml`."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .loader import dump_status, load_status
from .schema import ConfigReference, StatusEvent, StatusV2
from ..imap.rules_mail import RulesMailRef


class StatusStore:
    """Persist and mutate the status snapshot stored on disk."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self.save(StatusV2.minimal())

    def load(self) -> StatusV2:
        try:
            document = load_status(self._path.read_bytes())
        except FileNotFoundError:
            status = StatusV2.minimal()
            self.save(status)
            return status
        return document.model

    def save(self, status: StatusV2) -> None:
        self._path.write_bytes(dump_status(status))

    def current_config_ref(self) -> Optional[ConfigReference]:
        return self.load().config_ref

    def update_config_ref(self, ref: RulesMailRef, *, reset_errors: bool = True) -> None:
        status = self.load()
        status.config_ref = ConfigReference(
            uid=ref.uid,
            message_id=ref.message_id,
            internaldate=ref.internaldate.isoformat(),
            checksum=ref.checksum,
        )
        status.config_checksum = ref.checksum
        if reset_errors:
            status.summary.errors = 0
        status.restored_rules_from_backup = False
        self.save(status)

    def mark_restored(self, from_backup: bool) -> None:
        status = self.load()
        status.restored_rules_from_backup = from_backup
        self.save(status)

    def mark_invalid(self) -> None:
        status = self.load()
        status.summary.errors = max(status.summary.errors, 1)
        self.save(status)

    def append_event(self, event_type: str, details: str) -> None:
        status = self.load()
        status.events.append(
            StatusEvent(
                ts=datetime.now(timezone.utc).isoformat(),
                type=event_type,  # type: ignore[arg-type]
                details=details,
            )
        )
        self.save(status)

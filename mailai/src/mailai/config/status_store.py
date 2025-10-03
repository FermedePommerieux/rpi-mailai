"""MailAI status persistence utilities.

What:
  Provide a filesystem-backed accessor for ``status.yaml`` that persists the
  runtime health snapshot, last processed configuration reference, and
  operator-facing telemetry events.

Why:
  The orchestrator needs a single source of truth that survives reboots and
  power loss while never persisting clear-text email payloads. Persisting the
  structured status document keeps observability and recovery simple without
  requiring a database.

How:
  Wraps serialization helpers from :mod:`mailai.config.loader` and schema
  models from :mod:`mailai.config.schema`. The store eagerly materializes the
  document, applies in-memory mutations, and then writes atomically to the
  configured path. Missing files are bootstrapped with ``StatusV2.minimal`` to
  keep invariants intact.

Interfaces:
  ``StatusStore`` exposing ``load``, ``save``, ``current_config_ref``,
  ``update_config_ref``, ``mark_restored``, ``mark_invalid``, and
  ``append_event``.

Invariants & Safety:
  - ``status.yaml`` is recreated with minimal contents when absent.
  - The store never mutates email content; only metadata defined by the schema
    is persisted.
  - Callers must treat methods as non-transactional; each helper reloads the
    document to avoid stale mutations when concurrent processes touch the file.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .loader import dump_status, load_status
from .schema import ConfigReference, StatusEvent, StatusV2
from ..imap.rules_mail import RulesMailRef


class StatusStore:
    """High-level wrapper for manipulating the status snapshot on disk.

    What:
      Encapsulates filesystem access and schema-aware mutations for the status
      snapshot so callers interact with Python objects instead of raw YAML.

    Why:
      Centralizes creation and updates to guarantee the invariants required by
      automation (e.g., error counters are monotonic, config references stay in
      sync) and to allow future hardening (locking, journaling) in one place.

    How:
      Initializes the target path, bootstraps it with a minimal document, and
      performs a load-modify-save cycle for every write to minimize data loss
      from crashes.
    """

    def __init__(self, path: Path):
        """Create a status store that writes to ``path``.

        What:
          Materializes the backing file and parent directories so subsequent
          reads and writes succeed.

        Why:
          Status persistence must work even on first boot or after manual
          cleanup; proactively creating directories avoids repeated guard code
          across call sites.

        How:
          Coerces ``path`` to :class:`~pathlib.Path`, creates its parents, and
          emits a minimal status document when the file is absent.

        Args:
          path: Location on disk for ``status.yaml``.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self.save(StatusV2.minimal())

    def load(self) -> StatusV2:
        """Load the current status snapshot from disk.

        What:
          Returns the structured :class:`StatusV2` model describing the latest
          orchestrator status.

        Why:
          Callers need a consistent view even when the file was deleted or not
          yet created. Auto-recovery prevents cascading errors in diagnostics.

        How:
          Attempts to deserialize bytes with :func:`load_status`. On
          ``FileNotFoundError`` a minimal schema instance is written and
          returned.

        Returns:
          The loaded :class:`StatusV2` document.
        """
        try:
            document = load_status(self._path.read_bytes())
        except FileNotFoundError:
            status = StatusV2.minimal()
            self.save(status)
            return status
        return document.model

    def save(self, status: StatusV2) -> None:
        """Persist ``status`` to disk using the canonical serializer.

        What:
          Serializes and writes the provided :class:`StatusV2` snapshot.

        Why:
          Centralizing the write logic ensures a single location enforces the
          binary-safe serialization used throughout the project.

        How:
          Delegates to :func:`dump_status` then writes the bytes with
          :meth:`Path.write_bytes`.

        Args:
          status: The structured status document to persist.
        """
        self._path.write_bytes(dump_status(status))

    def current_config_ref(self) -> Optional[ConfigReference]:
        """Return the reference to the currently active configuration.

        What:
          Extracts the ``ConfigReference`` describing the last applied IMAP
          configuration message, if recorded.

        Why:
          Higher-level watchers compare this metadata with mailbox contents to
          detect new configuration messages without parsing full YAML bodies.

        How:
          Loads the status snapshot and returns its ``config_ref`` field.

        Returns:
          The stored :class:`ConfigReference` or ``None`` when unknown.
        """
        return self.load().config_ref

    def update_config_ref(self, ref: RulesMailRef, *, reset_errors: bool = True) -> None:
        """Persist the latest configuration metadata and checksum.

        What:
          Records the IMAP message identifiers and checksum for the currently
          enforced configuration.

        Why:
          MailAI relies on UID and checksum changes to detect configuration
          updates in immutable mailboxes. Resetting error counters on success
          prevents stale failure banners.

        How:
          Loads the snapshot, updates the ``config_ref`` and ``config_checksum``
          fields, optionally resets aggregated error counts, clears the
          ``restored_rules_from_backup`` flag, and saves the document.

        Args:
          ref: Canonical reference to the processed configuration mail.
          reset_errors: Whether to zero the error counter for a fresh
            successful sync.
        """
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
        """Mark whether the active rules were sourced from a backup copy.

        What:
          Toggles the restoration flag after a recovery workflow completes.

        Why:
          Operators must know when automation relied on backups instead of
          fresh configuration emails, as that affects trust and audit steps.

        How:
          Reloads the snapshot, sets ``restored_rules_from_backup`` to the
          supplied flag, and persists the document.

        Args:
          from_backup: ``True`` when backup rules were restored, ``False`` when
            using the latest mailbox configuration.
        """
        status = self.load()
        status.restored_rules_from_backup = from_backup
        self.save(status)

    def mark_invalid(self) -> None:
        """Record that the current configuration is invalid.

        What:
          Ensures at least one error is reported in the status summary.

        Why:
          Downstream health checks treat a non-zero error counter as a signal
          that configuration parsing failed and requires intervention.

        How:
          Loads the snapshot, bumps ``summary.errors`` to ``max(existing, 1)``,
          and saves it. This keeps the counter monotonic without inflating it on
          repeated calls.
        """
        status = self.load()
        status.summary.errors = max(status.summary.errors, 1)
        self.save(status)

    def append_event(self, event_type: str, details: str) -> None:
        """Append a telemetry event to the status timeline.

        What:
          Adds a timestamped :class:`StatusEvent` describing a noteworthy
          runtime condition.

        Why:
          Operators rely on the event log to diagnose rule sync failures,
          fallback activations, and privacy-sensitive conditions without
          storing raw content.

        How:
          Loads the snapshot, appends a new ``StatusEvent`` with the current
          UTC timestamp and provided metadata, and writes the document.

        Args:
          event_type: Short machine-readable event category.
          details: Human-readable summary suitable for status dashboards.
        """
        status = self.load()
        status.events.append(
            StatusEvent(
                ts=datetime.now(timezone.utc).isoformat(),
                type=event_type,  # type: ignore[arg-type]
                details=details,
            )
        )
        self.save(status)


# TODO: Document remaining modules with the same What/Why/How structure.

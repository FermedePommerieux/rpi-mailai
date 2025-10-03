"""Encrypted backup store for `rules.yaml`."""
from __future__ import annotations

from pathlib import Path

from ..core.privacy import decrypt, encrypt, KEY_SIZE
from ..config import yamlshim
from ..config.schema import RulesV2


class BackupError(RuntimeError):
    """Raised when backup encryption or decryption fails."""


class EncryptedRulesBackup:
    """Persist the last known good rules document encrypted at rest."""

    def __init__(self, path: Path, key: bytes):
        if len(key) != KEY_SIZE:
            raise ValueError("Backup key must be 32 bytes")
        self._path = Path(path)
        self._key = key
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, yaml_text: str) -> None:
        """Persist a validated rules document."""

        payload = yaml_text.encode("utf-8")
        blob = encrypt(payload, self._key)
        self._path.write_bytes(blob)

    def load_or_minimal(self) -> str:
        """Return the stored configuration or a minimal template when unavailable."""

        try:
            blob = self._path.read_bytes()
        except FileNotFoundError:
            return _minimal_yaml()
        try:
            data = decrypt(blob, self._key)
        except Exception as exc:
            raise BackupError("Failed to decrypt rules backup") from exc
        return data.decode("utf-8")

    def last_known_good(self) -> str:
        """Return the stored configuration or fall back to the minimal template."""

        try:
            blob = self._path.read_bytes()
        except FileNotFoundError:
            return _minimal_yaml()
        try:
            data = decrypt(blob, self._key)
        except Exception:
            return _minimal_yaml()
        return data.decode("utf-8")

    def has_backup(self) -> bool:
        return self._path.exists()


def _minimal_yaml() -> str:
    minimal = RulesV2.minimal()
    return yamlshim.dump(minimal.model_dump(mode="json"))

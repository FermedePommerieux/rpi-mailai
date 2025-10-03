"""SQLCipher helpers for encrypted SQLite storage.

What:
  Provide a guarded import of :mod:`pysqlcipher3` plus a convenience function for
  opening encrypted SQLite databases with optional PRAGMA configuration.

Why:
  MailAI stores backups and cached status on Raspberry Pi hardware where SQLCipher
  may be absent. By centralising availability checks we can fail fast with a
  descriptive exception and keep configuration plumbing simple.

How:
  Attempt to import :mod:`pysqlcipher3`. If unavailable, expose
  :class:`SqlCipherUnavailable`. When present, :func:`open_encrypted_database`
  calls ``sqlcipher.connect`` and applies caller-supplied PRAGMAs to configure
  page size, kdf iterations, etc.

Interfaces:
  :class:`SqlCipherUnavailable`, :func:`open_encrypted_database`.

Invariants & Safety:
  - Connections are only returned once the ``PRAGMA key`` command has executed
    successfully.
  - Additional PRAGMAs are executed verbatim; callers must supply trusted values
    sourced from configuration defaults.
"""
from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from pysqlcipher3 import dbapi2 as _sqlcipher

    SqlCipherConnection = _sqlcipher.Connection
else:  # pragma: no cover - runtime fallback
    SqlCipherConnection = object


try:  # pragma: no cover - optional dependency
    from pysqlcipher3 import dbapi2 as sqlcipher
except ImportError:  # pragma: no cover
    sqlcipher = None  # type: ignore[assignment]


class SqlCipherUnavailable(RuntimeError):
    """Exception signalling missing SQLCipher support.

    What:
      Communicates that :mod:`pysqlcipher3` could not be imported on the current
      system.

    Why:
      Callers need a dedicated error to surface actionable remediation steps to
      operators (install the dependency or disable encrypted storage).

    How:
      Subclasses :class:`RuntimeError` without additional state. Raised by
      :func:`open_encrypted_database` when SQLCipher is absent.
    """


def open_encrypted_database(
    path: str,
    *,
    key: str,
    pragmas: Optional[Dict[str, str]] = None,
) -> SqlCipherConnection:
    """Open an encrypted SQLite database guarded by SQLCipher.

    What:
      Establishes a connection to ``path`` using SQLCipher, applies the provided
      ``key`` as the encryption secret, and executes optional PRAGMA statements.

    Why:
      Encapsulating the boilerplate ensures all components initialise SQLCipher
      consistently (key derivation, cipher settings) and emit a clear error when
      the dependency is unavailable.

    How:
      Validates the presence of ``sqlcipher``. Raises
      :class:`SqlCipherUnavailable` if the module import failed. Otherwise opens
      the database, executes ``PRAGMA key = ?`` with the supplied key, and
      iterates through the ``pragmas`` mapping executing each ``PRAGMA`` command
      verbatim.

    Args:
      path: Filesystem path to the encrypted database file.
      key: Secret string used to derive the SQLCipher encryption key.
      pragmas: Optional mapping of PRAGMA directives to configure the connection
        (e.g., ``{"cipher_page_size": "4096"}``).

    Returns:
      Active SQLCipher connection object ready for database operations.

    Raises:
      SqlCipherUnavailable: If ``pysqlcipher3`` is not installed on the host.
    """

    if sqlcipher is None:
        raise SqlCipherUnavailable("SQLCipher driver pysqlcipher3 is required for encrypted stores")
    connection = sqlcipher.connect(path)
    connection.execute("PRAGMA key = ?", (key,))
    for pragma, value in (pragmas or {}).items():
        connection.execute(f"PRAGMA {pragma} = {value}")
    return connection


# TODO: Other modules in this repository still require the same What/Why/How documentation.

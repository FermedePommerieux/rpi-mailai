"""Helpers for working with SQLCipher encrypted SQLite databases."""
from __future__ import annotations

from typing import Dict, Optional

try:  # pragma: no cover - optional dependency
    from pysqlcipher3 import dbapi2 as sqlcipher
except ImportError:  # pragma: no cover
    sqlcipher = None  # type: ignore[assignment]


class SqlCipherUnavailable(RuntimeError):
    """Raised when SQLCipher support is not available on the host."""


def open_encrypted_database(
    path: str,
    *,
    key: str,
    pragmas: Optional[Dict[str, str]] = None,
):
    """Open an encrypted SQLite database using SQLCipher.

    Parameters
    ----------
    path:
        Filesystem path to the database.
    key:
        Secret used to derive the encryption key. The caller is responsible for
        supplying a suitably random string.
    pragmas:
        Additional PRAGMA statements to execute after the database is opened.

    Returns
    -------
    Connection
        A SQLCipher database connection ready for use.
    """

    if sqlcipher is None:
        raise SqlCipherUnavailable("SQLCipher driver pysqlcipher3 is required for encrypted stores")
    connection = sqlcipher.connect(path)
    connection.execute("PRAGMA key = ?", (key,))
    for pragma, value in (pragmas or {}).items():
        connection.execute(f"PRAGMA {pragma} = {value}")
    return connection

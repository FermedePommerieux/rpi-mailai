"""Expose the public utility surface for MailAI.

What:
  Re-export logging, identifier, and SQLCipher helpers that other packages may
  import without knowing the underlying module layout.

Why:
  Centralising exports provides a stable facade so downstream code can perform
  ``from mailai import utils`` imports without depending on internal filenames.

How:
  Imports the canonical utility functions/classes and populates ``__all__`` for
  lint friendliness.

Interfaces:
  ``get_logger``, ``new_run_id``, ``checksum``, ``SqlCipherUnavailable``, and
  ``open_encrypted_database``.

Invariants & Safety:
  - The module only re-exports side-effect-free callables to keep import order
    predictable.
  - Logging defaults emit redacted JSON lines; consumers should avoid bypassing
    these helpers.
"""

from .logging import get_logger
from .ids import new_run_id, checksum
from .sqlcipher import SqlCipherUnavailable, open_encrypted_database

__all__ = [
    "get_logger",
    "new_run_id",
    "checksum",
    "SqlCipherUnavailable",
    "open_encrypted_database",
]


# TODO: Other modules in this repository still require the same What/Why/How documentation.

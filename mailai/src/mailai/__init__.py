"""
Module: mailai.__init__

What:
  Aggregate package exports for the MailAI offline email triage agent and expose
  the primary namespace segments (configuration, core logic, IMAP handling, and
  utilities).

Why:
  Centralising the exports keeps downstream tooling and entry points stable
  while the internal layout evolves. Importers rely on these names to build CLI
  commands, load configuration schemas, and assemble the processing pipeline
  without touching private modules.

How:
  Provide an explicit ``__all__`` declaration that enumerates the public
  subpackages. This guards against accidental re-export of helper modules and
  clarifies the intended extension points for developers adding new features or
  audits.

Interfaces:
  - config: Configuration schema loaders and validators.
  - core: Rule engine, learner, and privacy-preserving analytics.
  - imap: Offline-safe IMAP client helpers for UID-first processing.
  - utils: Shared helpers for logging, MIME handling, and secure primitives.

Invariants:
  - Only vetted subpackages are exposed to callers; new exports must undergo
    the same documentation and privacy review as the rest of the project.
  - The package never re-exports modules that could leak raw email content or
    bypass encryption safeguards.

Safety/Performance:
  - Keeping the surface minimal reduces import overhead on constrained
    Raspberry Pi hardware and prevents misuse of experimental features.
"""

__all__ = [
    "config",
    "core",
    "imap",
    "utils",
]

# TODO: Other modules require the same treatment (Quoi/Pourquoi/Comment docstrings + module header).

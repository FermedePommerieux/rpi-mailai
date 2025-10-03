"""Facade for the IMAP integration layer.

What:
  Surface the strongly-typed :class:`~mailai.imap.client.ImapConfig` data class
  and :class:`~mailai.imap.client.MailAIImapClient` context manager used across
  the codebase for all IMAP interactions.

Why:
  Keeping the import surface minimal prevents call sites from depending on
  internal helper modules, making it easier to evolve the IMAP stack without
  sweeping refactors.

How:
  Re-exports the canonical client and configuration classes and leaves the
  detailed operations to submodules such as ``imap.actions`` and
  ``imap.rules_mail``.

Interfaces:
  ``ImapConfig`` and ``MailAIImapClient``.

Invariants & Safety:
  - Consumers are expected to operate in UID-first mode to avoid race
    conditions.
  - All IMAP operations should go through :class:`MailAIImapClient` to inherit
    rate limiting and mailbox guardrails.
"""

from .client import ImapConfig, MailAIImapClient

__all__ = ["ImapConfig", "MailAIImapClient"]


# TODO: Other modules in this repository still require the same What/Why/How documentation.

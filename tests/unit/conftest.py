"""Pytest fixtures for unit tests requiring IMAP fakes.

What:
  Prepare the unit test environment by ensuring ``tests/unit`` is importable and
  by exposing an ``imap_client`` fixture backed by :class:`FakeImapBackend`.

Why:
  Unit tests interact with the IMAP client abstraction extensively. Providing a
  consistent fake prevents tests from depending on network resources and keeps
  message flows deterministic.

How:
  Append the unit directory to ``sys.path`` for local imports, instantiate
  :class:`MailAIImapClient` with a fake backend, and yield both the high-level
  client and backend for assertions.

Interfaces:
  :func:`imap_client` (pytest fixture).

Invariants & Safety:
  - The fake backend avoids network access and ensures actions like ``move`` and
    ``copy`` manipulate in-memory dictionaries only.
  - Each test receives a fresh backend instance to eliminate state leakage.
"""

import sys
from pathlib import Path

import pytest

from mailai.imap.client import ImapConfig, MailAIImapClient

UNIT_DIR = Path(__file__).resolve().parent
if str(UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(UNIT_DIR))

from fakes import FakeImapBackend


@pytest.fixture
def imap_client(monkeypatch: pytest.MonkeyPatch):
    """Yield a MailAI IMAP client backed by the in-memory fake backend.

    What:
      Returns a tuple containing the context-managed :class:`MailAIImapClient`
      and the shared :class:`FakeImapBackend` instance.

    Why:
      Tests assert on backend state (mailboxes, UIDs) while invoking high-level
      client methods. Exposing both objects keeps assertions explicit.

    How:
      Monkeypatches ``mailai.imap.client.IMAPClient`` to return the fake backend,
      builds an :class:`ImapConfig` with dummy credentials, and yields the client
      within its context manager so the login/logout flow mirrors production.

    Args:
      monkeypatch: Pytest helper used to replace the IMAP client constructor.

    Yields:
      Tuple ``(MailAIImapClient, FakeImapBackend)`` for use in tests.
    """

    backend = FakeImapBackend()
    monkeypatch.setattr("mailai.imap.client.IMAPClient", lambda host, port, ssl: backend)
    config = ImapConfig(host="localhost", username="user", password="pass")
    with MailAIImapClient(config) as client:
        yield client, backend


# TODO: Other modules in this repository still require the same What/Why/How documentation.

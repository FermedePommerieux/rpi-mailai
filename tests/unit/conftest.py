import sys
from pathlib import Path

import pytest

from mailai.imap.client import ImapConfig, MailAIImapClient

UNIT_DIR = Path(__file__).resolve().parent
if str(UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(UNIT_DIR))

from fakes import FakeImapBackend


@pytest.fixture
def imap_client(monkeypatch):
    backend = FakeImapBackend()
    monkeypatch.setattr("mailai.imap.client.IMAPClient", lambda host, port, ssl: backend)
    config = ImapConfig(host="localhost", username="user", password="pass")
    with MailAIImapClient(config) as client:
        yield client, backend

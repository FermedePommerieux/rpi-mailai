import pytest

from mailai.utils.sqlcipher import SqlCipherUnavailable, open_encrypted_database


def test_open_encrypted_database_executes_pragmas(monkeypatch):
    executed = []

    class FakeConnection:
        def execute(self, sql, params=None):
            executed.append((sql, params))
            return None

    class FakeDriver:
        def connect(self, path):
            executed.append(("connect", path))
            return FakeConnection()

    fake_driver = FakeDriver()
    module = __import__("mailai.utils.sqlcipher", fromlist=["sqlcipher"])
    monkeypatch.setattr(module, "sqlcipher", fake_driver, raising=False)

    conn = open_encrypted_database("/tmp/test.db", key="secret", pragmas={"cipher_memory_security": "ON"})
    assert executed[0] == ("connect", "/tmp/test.db")
    assert ("PRAGMA key = ?", ("secret",)) in executed
    assert ("PRAGMA cipher_memory_security = ON", None) in executed
    assert isinstance(conn, FakeConnection)


def test_open_encrypted_database_requires_driver(monkeypatch):
    module = __import__("mailai.utils.sqlcipher", fromlist=["sqlcipher"])
    monkeypatch.setattr(module, "sqlcipher", None, raising=False)
    with pytest.raises(SqlCipherUnavailable):
        open_encrypted_database("/tmp/test.db", key="secret")

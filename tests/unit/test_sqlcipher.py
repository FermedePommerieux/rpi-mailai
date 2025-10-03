"""
Module: tests/unit/test_sqlcipher.py

What:
    Validate the SQLCipher convenience wrapper that opens encrypted SQLite
    databases with mandatory pragmas and defensive driver detection.

Why:
    Without these tests the project could silently skip crucial security pragmas
    or continue when the underlying driver is unavailable, risking plaintext
    leaks.

How:
    Simulate driver behaviour with monkeypatching to assert executed SQL and
    ensure the helper raises the appropriate exception when SQLCipher bindings
    are absent.

Interfaces:
    test_open_encrypted_database_executes_pragmas,
    test_open_encrypted_database_requires_driver

Invariants & Safety Rules:
    - ``PRAGMA key`` must be issued on every connection before queries run.
    - Additional pragmas provided by the caller should execute exactly once.
    - Driver absence must raise ``SqlCipherUnavailable``.
"""

import pytest

from mailai.utils.sqlcipher import SqlCipherUnavailable, open_encrypted_database


def test_open_encrypted_database_executes_pragmas(monkeypatch):
    """
    What:
        Ensure the helper configures the SQLCipher connection with key and custom
        pragmas before returning the connection handle.

    Why:
        Encryption pragmas are required to secure the database; missing them would
        leave data readable on disk.

    How:
        Monkeypatch the module-level driver with a fake implementation capturing
        executed SQL, invoke the helper, and assert the expected statements were
        issued.

    Args:
        monkeypatch: Pytest fixture used to inject the fake driver.

    Returns:
        None
    """
    executed = []

    class FakeConnection:
        """Lightweight connection stub capturing executed statements.

        What:
            Record SQL executed by :func:`open_encrypted_database`.

        Why:
            Allows the test to assert PRAGMAs were issued without depending on
            a real SQLCipher library.

        How:
            Append ``(sql, params)`` tuples to the shared ``executed`` list.
        """

        def execute(self, sql, params=None):
            executed.append((sql, params))
            return None

    class FakeDriver:
        """Stub SQLCipher driver returning :class:`FakeConnection` objects.

        What:
            Mimic ``sqlcipher.connect`` while logging connection attempts.

        Why:
            Ensures the helper issues the connect call before applying PRAGMAs.

        How:
            Append a ``("connect", path)`` record and return :class:`FakeConnection`.
        """

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
    """
    What:
        Verify the helper raises ``SqlCipherUnavailable`` when SQLCipher bindings
        are missing.

    Why:
        Attempting to proceed without encryption support would corrupt or expose
        sensitive state; callers must receive a hard failure.

    How:
        Monkeypatch the ``sqlcipher`` attribute to ``None`` and assert the helper
        raises the expected exception upon invocation.

    Args:
        monkeypatch: Fixture enabling dynamic patching of module attributes.

    Returns:
        None

    Raises:
        AssertionError: If ``SqlCipherUnavailable`` is not raised.
    """
    module = __import__("mailai.utils.sqlcipher", fromlist=["sqlcipher"])
    monkeypatch.setattr(module, "sqlcipher", None, raising=False)
    with pytest.raises(SqlCipherUnavailable):
        open_encrypted_database("/tmp/test.db", key="secret")


# TODO: Other modules in this repository still require the same What/Why/How documentation.

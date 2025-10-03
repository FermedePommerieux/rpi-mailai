"""
Module: tests/unit/test_regexsafe.py

What:
    Validate the safe regular expression search helper which protects against
    catastrophic backtracking by enforcing timeouts.

Why:
    Regex denial-of-service can freeze the daemon; tests ensure that matches work
    for benign patterns while pathological inputs respect timeout behaviour.

How:
    Execute the search helper with both a normal pattern and a deliberately
    expensive expression under tight timeout constraints, asserting on match
    outcomes.

Interfaces:
    test_regex_search_matches, test_regex_timeout_returns_false

Invariants & Safety Rules:
    - Successful matches must set ``matched`` to ``True`` for valid patterns.
    - Timeouts must surface as ``matched`` False rather than raising unexpected
      exceptions.
"""

from mailai.utils import regexsafe


def test_regex_search_matches():
    """
    What:
        Confirm the helper returns a successful match on safe input.

    Why:
        Basic functionality must remain intact while the safety wrapper enforces
        timeouts.

    How:
        Call ``regexsafe.search`` with a simple literal and assert the ``matched``
        attribute is ``True``.

    Returns:
        None
    """
    result = regexsafe.search(r"hello", "hello world")
    assert result.matched


def test_regex_timeout_returns_false():
    """
    What:
        Ensure catastrophic patterns respect the timeout and report no match.

    Why:
        Prevents regex DoS by bounding evaluation time and signalling the caller
        without raising.

    How:
        Execute a known backtracking-heavy pattern against a long string with a
        1ms timeout and assert the ``matched`` flag is ``False``.

    Returns:
        None
    """
    pattern = r"(a+)+$"
    result = regexsafe.search(pattern, "a" * 1000, timeout_ms=1)
    assert not result.matched


# TODO: Other modules in this repository still require the same What/Why/How documentation.

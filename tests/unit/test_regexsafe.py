from mailai.utils import regexsafe


def test_regex_search_matches():
    result = regexsafe.search(r"hello", "hello world")
    assert result.matched


def test_regex_timeout_returns_false():
    pattern = r"(a+)+$"
    result = regexsafe.search(pattern, "a" * 1000, timeout_ms=1)
    assert not result.matched

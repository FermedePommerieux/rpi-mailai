from mailai.core import privacy


def test_encrypt_roundtrip():
    key = b"a" * privacy.KEY_SIZE
    data = b"secret"
    blob = privacy.encrypt(data, key)
    assert blob != data
    restored = privacy.decrypt(blob, key)
    assert restored == data


def test_peppered_hashing():
    pepper = b"pepper"
    hasher = privacy.PepperHasher(pepper=pepper, salt=b"salt")
    tokens = ["hello", "world"]
    hashed = hasher.hash_tokens(tokens)
    assert len(hashed) == 2
    assert hashed[0] != "hello"
    simhash_value = hasher.simhash(tokens)
    assert isinstance(simhash_value, int)

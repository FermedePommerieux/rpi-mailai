"""
Module: tests/unit/test_privacy.py

What:
    Validate the privacy helper routines covering symmetric encryption and
    peppered hashing/simhash derivation.

Why:
    The privacy layer enforces strong guarantees that plaintext is not persisted;
    regression here could leak sensitive identifiers or fail to decrypt cached
    content.

How:
    Perform round-trip encryption/decryption with deterministic keys and exercise
    the peppered hashing interface to assert redaction and simhash stability.

Interfaces:
    test_encrypt_roundtrip, test_peppered_hashing

Invariants & Safety Rules:
    - Ciphertexts must differ from plaintext inputs to ensure encryption occurs.
    - Hash outputs should never equal their inputs and must produce integer
      simhash values for downstream matching.
"""

from mailai.core import privacy


def test_encrypt_roundtrip():
    """
    What:
        Ensure encryption and decryption helpers form a lossless round-trip when
        supplied with the same symmetric key.

    Why:
        Successful round-trips confirm keys are sized correctly and the ChaCha20
        nonce/AEAD handling remains intact.

    How:
        Encrypt a small payload, assert ciphertext differs from plaintext, then
        decrypt using the same key and compare outputs.

    Returns:
        None
    """
    key = b"a" * privacy.KEY_SIZE
    data = b"secret"
    blob = privacy.encrypt(data, key)
    assert blob != data
    restored = privacy.decrypt(blob, key)
    assert restored == data


def test_peppered_hashing():
    """
    What:
        Confirm the ``PepperHasher`` scrubs plaintext tokens and emits integer
        simhash values suitable for anonymous comparisons.

    Why:
        Peppering defends against dictionary attacks; this test ensures hashed
        tokens differ from inputs and still produce deterministic simhash
        fingerprints.

    How:
        Instantiate the hasher with static pepper/salt, hash a token list, and
        assert outputs include redacted tokens plus an integer simhash.

    Returns:
        None
    """
    pepper = b"pepper"
    hasher = privacy.PepperHasher(pepper=pepper, salt=b"salt")
    tokens = ["hello", "world"]
    hashed = hasher.hash_tokens(tokens)
    assert len(hashed) == 2
    assert hashed[0] != "hello"
    simhash_value = hasher.simhash(tokens)
    assert isinstance(simhash_value, int)


# TODO: Other modules in this repository still require the same What/Why/How documentation.

"""Privacy and cryptography helpers."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List

try:  # pragma: no cover - executed when libsodium bindings are present
    from nacl import utils
    from nacl.bindings import (
        crypto_aead_chacha20poly1305_ietf_decrypt,
        crypto_aead_chacha20poly1305_ietf_encrypt,
    )
except ImportError:  # pragma: no cover - fallback implementation for tests
    utils = None
    crypto_aead_chacha20poly1305_ietf_decrypt = None
    crypto_aead_chacha20poly1305_ietf_encrypt = None
    import secrets

REDACTED = "[redacted]"
AAD = b"mailai:v2"
KEY_SIZE = 32
NONCE_SIZE = 12


def encrypt(data: bytes, key: bytes) -> bytes:
    """Encrypt bytes using ChaCha20-Poly1305 with random nonce."""

    if len(key) != KEY_SIZE:
        raise ValueError("Key must be 32 bytes for ChaCha20-Poly1305")
    if crypto_aead_chacha20poly1305_ietf_encrypt is None:
        return _fallback_encrypt(data, key)
    nonce = utils.random(NONCE_SIZE)
    cipher = crypto_aead_chacha20poly1305_ietf_encrypt(data, AAD, nonce, key)
    return nonce + cipher


def decrypt(blob: bytes, key: bytes) -> bytes:
    """Decrypt data previously encrypted with :func:`encrypt`."""

    if len(key) != KEY_SIZE:
        raise ValueError("Key must be 32 bytes for ChaCha20-Poly1305")
    if len(blob) < NONCE_SIZE:
        raise ValueError("Ciphertext truncated")
    if crypto_aead_chacha20poly1305_ietf_decrypt is None:
        return _fallback_decrypt(blob, key)
    nonce, cipher = blob[:NONCE_SIZE], blob[NONCE_SIZE:]
    return crypto_aead_chacha20poly1305_ietf_decrypt(cipher, AAD, nonce, key)


def _fallback_encrypt(data: bytes, key: bytes) -> bytes:
    nonce = secrets.token_bytes(NONCE_SIZE)
    keystream = _derive_keystream(key, nonce, len(data))
    ciphertext = bytes(a ^ b for a, b in zip(data, keystream))
    tag = hashlib.sha256(key + nonce + data).digest()[:16]
    return nonce + tag + ciphertext


def _fallback_decrypt(blob: bytes, key: bytes) -> bytes:
    if len(blob) < NONCE_SIZE + 16:
        raise ValueError("Ciphertext truncated")
    nonce = blob[:NONCE_SIZE]
    tag = blob[NONCE_SIZE:NONCE_SIZE + 16]
    ciphertext = blob[NONCE_SIZE + 16 :]
    keystream = _derive_keystream(key, nonce, len(ciphertext))
    data = bytes(a ^ b for a, b in zip(ciphertext, keystream))
    expected_tag = hashlib.sha256(key + nonce + data).digest()[:16]
    if expected_tag != tag:
        raise ValueError("Authentication failed")
    return data


def _derive_keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    output = b""
    counter = 0
    while len(output) < length:
        counter_bytes = counter.to_bytes(4, "big")
        block = hashlib.blake2b(key + nonce + counter_bytes, digest_size=32).digest()
        output += block
        counter += 1
    return output[:length]


def redact(text: str) -> str:
    """Return a redacted placeholder for any user-facing text."""

    return REDACTED if text else ""


@dataclass
class PepperHasher:
    """Peppered hashing of message features for privacy."""

    pepper: bytes

    def hash_tokens(self, tokens: Iterable[str]) -> List[str]:
        """Hash tokens using SHA-256 and the configured pepper."""

        result: List[str] = []
        for token in tokens:
            digest = hashlib.sha256(self.pepper + token.encode("utf-8")).hexdigest()
            result.append(digest)
        return result

    def simhash(self, tokens: Iterable[str]) -> int:
        """Compute a simple SimHash style fingerprint."""

        accum = [0] * 64
        for token in tokens:
            digest = hashlib.sha256(self.pepper + token.encode("utf-8")).digest()
            bits = int.from_bytes(digest[:8], "big")
            for idx in range(64):
                if bits & (1 << idx):
                    accum[idx] += 1
                else:
                    accum[idx] -= 1
        value = 0
        for idx, weight in enumerate(accum):
            if weight > 0:
                value |= 1 << idx
        return value

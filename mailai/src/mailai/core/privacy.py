"""MailAI cryptography and redaction primitives.

What:
  Offer high-level helpers for encrypting configuration snapshots, redacting
  operator-facing strings, and generating privacy-preserving hashes that can be
  shared across processes without leaking plaintext.

Why:
  The platform synchronises state via IMAP and a lightweight SQLite cache; both
  must avoid storing clear-text content to respect tenant privacy. Centralising
  the cryptographic glue in this module guarantees consistent algorithms and
  parameters across backends while keeping a single choke point for upgrades.

How:
  Wraps libsodium's ChaCha20-Poly1305 bindings when available, falling back to a
  deterministic stream cipher shim for testing environments where those bindings
  are absent. Provides peppered hashing helpers used by the learner to anonymise
  message features before persistence. Module-level constants capture the nonce
  and key sizes so they remain identical between encryption and decryption.

Interfaces:
  ``encrypt``, ``decrypt``, ``redact``, and :class:`PepperHasher`.

Invariants & Safety:
  - Keys must always be 32 bytes to satisfy ChaCha20-Poly1305 requirements.
  - Ciphertexts are prefixed with nonces so the decryptor can reconstruct state
    without shared counters.
  - Pepper-based hashing never returns raw input tokens, preventing accidental
    logging of personal data even when learners persist derived features.
"""
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
    """Encrypt arbitrary bytes with authenticated ChaCha20-Poly1305.

    What:
      Serialises ``data`` into a ciphertext prefixed with a random nonce so the
      blob can be stored inside YAML attachments or SQLite pages safely.

    Why:
      Configuration snapshots and learner artefacts may contain sensitive
      metadata. Authenticated encryption both obfuscates the payload and detects
      tampering when state is reloaded from disk.

    How:
      Validates the provided ``key`` length, generates a fresh 96-bit nonce, and
      invokes libsodium's ChaCha20-Poly1305 routine when available. In test
      builds without bindings we fall back to :func:`_fallback_encrypt`, which
      provides deterministic coverage with reduced guarantees.

    Args:
      data: Plaintext payload to protect.
      key: 32-byte symmetric key shared with :func:`decrypt`.

    Returns:
      Ciphertext composed of nonce + encrypted bytes (+ tag when libsodium is
      available).

    Raises:
      ValueError: If ``key`` does not match :data:`KEY_SIZE`.
    """

    if len(key) != KEY_SIZE:
        raise ValueError("Key must be 32 bytes for ChaCha20-Poly1305")
    if crypto_aead_chacha20poly1305_ietf_encrypt is None:
        return _fallback_encrypt(data, key)
    nonce = utils.random(NONCE_SIZE)
    cipher = crypto_aead_chacha20poly1305_ietf_encrypt(data, AAD, nonce, key)
    return nonce + cipher


def decrypt(blob: bytes, key: bytes) -> bytes:
    """Reverse :func:`encrypt`, verifying authenticity.

    What:
      Restores plaintext bytes from an encrypted ``blob`` produced by
      :func:`encrypt`, ensuring the payload has not been tampered with.

    Why:
      Configuration recovery and learner resume flows rely on authenticated
      decryptions to avoid silently accepting corrupted or malicious artefacts
      pulled from IMAP storage.

    How:
      Validates the key length, ensures the blob at least contains a nonce, and
      then routes to the libsodium bindings or fallback implementation depending
      on availability. The libsodium branch automatically checks the Poly1305
      tag, raising when verification fails.

    Args:
      blob: Ciphertext prefixed with nonce.
      key: 32-byte symmetric key shared with :func:`encrypt`.

    Returns:
      The original plaintext payload.

    Raises:
      ValueError: If ``key`` length is invalid, the ciphertext is truncated, or
        authentication fails (in fallback mode).
    """

    if len(key) != KEY_SIZE:
        raise ValueError("Key must be 32 bytes for ChaCha20-Poly1305")
    if len(blob) < NONCE_SIZE:
        raise ValueError("Ciphertext truncated")
    if crypto_aead_chacha20poly1305_ietf_decrypt is None:
        return _fallback_decrypt(blob, key)
    nonce, cipher = blob[:NONCE_SIZE], blob[NONCE_SIZE:]
    return crypto_aead_chacha20poly1305_ietf_decrypt(cipher, AAD, nonce, key)


def _fallback_encrypt(data: bytes, key: bytes) -> bytes:
    """Provide deterministic coverage when libsodium bindings are unavailable.

    What:
      Produces a ciphertext that mimics the shape of ChaCha20-Poly1305 output to
      keep test fixtures stable even without native dependencies.

    Why:
      Continuous integration environments on Arm-based builders occasionally
      omit libsodium. Rather than failing outright, we exercise the surrounding
      persistence code paths with a simpler XOR stream cipher so behaviour stays
      predictable.

    How:
      Generates a random nonce, derives a pseudo keystream via Blake2b hashing,
      XORs it with the plaintext, and appends a truncated SHA-256 tag for basic
      tamper detection.

    Args:
      data: Plaintext payload to protect.
      key: Symmetric key shared with :func:`_fallback_decrypt`.

    Returns:
      Bytes combining nonce, tag, and ciphertext.
    """

    nonce = secrets.token_bytes(NONCE_SIZE)
    keystream = _derive_keystream(key, nonce, len(data))
    ciphertext = bytes(a ^ b for a, b in zip(data, keystream))
    tag = hashlib.sha256(key + nonce + data).digest()[:16]
    return nonce + tag + ciphertext


def _fallback_decrypt(blob: bytes, key: bytes) -> bytes:
    """Reconstruct plaintext using the fallback stream cipher.

    What:
      Recovers the original payload emitted by :func:`_fallback_encrypt` and
      verifies the truncated tag to detect tampering.

    Why:
      Keeps the developer experience deterministic on hosts without libsodium so
      regression tests exercise the same code paths with predictable input.

    How:
      Splits the blob into nonce, tag, and ciphertext segments, regenerates the
      keystream, XORs to restore plaintext, then recomputes the SHA-256-derived
      tag to confirm authenticity.

    Args:
      blob: Ciphertext generated by :func:`_fallback_encrypt`.
      key: Symmetric key matching the encryption phase.

    Returns:
      Decrypted plaintext payload.

    Raises:
      ValueError: If the blob is truncated or the computed tag mismatches.
    """

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
    """Derive a deterministic keystream chunk used by the fallback cipher.

    What:
      Expands the ``key`` and ``nonce`` into a pseudo-random byte stream of the
      requested ``length``.

    Why:
      Ensures that the fallback encrypt/decrypt routines remain invertible and
      stable across interpreter versions without shipping extra dependencies.

    How:
      Iteratively hashes the key, nonce, and a monotonically increasing counter
      using Blake2b, concatenating the resulting blocks until the desired length
      is met.

    Args:
      key: Symmetric key bytes.
      nonce: Random nonce to guarantee uniqueness per encryption.
      length: Number of bytes to emit.

    Returns:
      Deterministic keystream slice of ``length`` bytes.
    """

    output = b""
    counter = 0
    while len(output) < length:
        counter_bytes = counter.to_bytes(4, "big")
        block = hashlib.blake2b(key + nonce + counter_bytes, digest_size=32).digest()
        output += block
        counter += 1
    return output[:length]


def redact(text: str) -> str:
    """Replace user-supplied text with a redacted placeholder.

    What:
      Emits ``"[redacted]"`` when ``text`` is non-empty so telemetry and logs do
      not accidentally record personal data.

    Why:
      Operators require confirmation that sensitive content was intentionally
      hidden. Returning a consistent placeholder avoids leaking the length or
      presence of specific details beyond a yes/no signal.

    How:
      Performs a simple truthy check and maps to the :data:`REDACTED` constant.

    Args:
      text: Source string potentially containing sensitive data.

    Returns:
      ``"[redacted]"`` when ``text`` is truthy, otherwise the empty string.
    """

    return REDACTED if text else ""


@dataclass
class PepperHasher:
    """Pepper-aware hashing utilities used by the learner.

    What:
      Carries a tenant-specific ``pepper`` and short-lived ``salt`` used to hash
      tokenised message features prior to persistence.

    Why:
      Prevents replay attacks and re-identification by ensuring identical
      features hash differently across deployments while still allowing stable
      comparisons within a single run.

    How:
      Concatenates the salt, pepper, and UTF-8 encoded token before hashing with
      SHA-256. Provides helpers for both per-token hashing and coarse-grained
      SimHash fingerprints that power deduplication heuristics.

    Attributes:
      pepper: Secret pepper shared across the deployment.
      salt: Ephemeral salt tied to a specific processing batch.
    """

    pepper: bytes
    salt: bytes

    def hash_tokens(self, tokens: Iterable[str]) -> List[str]:
        """Hash tokens deterministically with the configured pepper.

        What:
          Converts each token into a hexadecimal SHA-256 digest bound to the
          dataclass' ``salt`` and ``pepper``.

        Why:
          Learner checkpoints and telemetry require stable identifiers without
          exposing human-readable content. Hashing allows equality comparisons
          while keeping the raw strings secret.

        How:
          Iterates through ``tokens``, concatenates salt, pepper, and the UTF-8
          encoded token, and appends the digest to a list that mirrors the input
          order.

        Args:
          tokens: Sequence of feature strings to anonymise.

        Returns:
          List of SHA-256 digests as lowercase hexadecimal strings.
        """

        result: List[str] = []
        for token in tokens:
            digest = hashlib.sha256(self.salt + self.pepper + token.encode("utf-8")).hexdigest()
            result.append(digest)
        return result

    def simhash(self, tokens: Iterable[str]) -> int:
        """Compute a privacy-preserving SimHash fingerprint.

        What:
          Produces a 64-bit integer capturing the approximate distribution of
          hashed tokens, enabling fuzzy similarity comparisons.

        Why:
          Allows the learner to detect duplicate conversations or newsletters
          without storing raw subject lines or sender addresses.

        How:
          Reuses the salted SHA-256 digest, interprets the first 64 bits as a
          signed contribution, and accumulates positive/negative votes per bit
          position before collapsing them into a binary fingerprint.

        Args:
          tokens: Iterable of feature strings to fingerprint.

        Returns:
          64-bit SimHash integer derived from the anonymised token stream.
        """

        accum = [0] * 64
        for token in tokens:
            digest = hashlib.sha256(self.salt + self.pepper + token.encode("utf-8")).digest()
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


# TODO: Other modules in this repository still require the same What/Why/How documentation.

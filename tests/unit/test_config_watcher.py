"""
Module: tests/unit/test_config_watcher.py

What:
    Validate change detection utilities that observe IMAP-delivered configuration
    messages to determine when runtime state must be refreshed.

Why:
    Accurate change detection prevents unnecessary reloads while guaranteeing that
    a revised YAML body or new message UID always propagates into the active
    configuration snapshot, supporting safe mailbox polling on constrained
    hardware.

How:
    The tests synthesise configuration references and rule mail descriptors with
    controlled UIDs and checksums, comparing the helper outputs for both UID and
    checksum drift scenarios.

Interfaces:
    test_has_changed_detects_uid_difference, test_has_changed_detects_checksum_difference

Invariants & Safety Rules:
    - UID changes must always invalidate the cached configuration reference.
    - Checksum mismatches signal differing body content regardless of UID reuse.
"""

from datetime import datetime, timezone

from mailai.config.schema import ConfigReference
from mailai.config.watcher import change_reason, has_changed
from mailai.imap.rules_mail import RulesMailRef


def _ref(uid: int, checksum: str) -> ConfigReference:
    """
    What:
        Build a ``ConfigReference`` helper tailored for change-detection tests.

    Why:
        Tests need deterministic references without coupling to production
        factories; this helper keeps focus on UID/checksum behaviour.

    How:
        Populate the dataclass with a synthetic message id, current timestamp, and
        caller-provided checksum so tests can manipulate state precisely.

    Args:
        uid: UID captured during the last known configuration sync.
        checksum: Digest representing the YAML body for that sync point.

    Returns:
        ConfigReference: The assembled reference snapshot used in comparisons.
    """
    return ConfigReference(uid=uid, message_id="m", internaldate=datetime.now(timezone.utc).isoformat(), checksum=checksum)


def _mail(uid: int, checksum: str) -> RulesMailRef:
    """
    What:
        Create a ``RulesMailRef`` structure mimicking the latest configuration
        message retrieved from IMAP.

    Why:
        Direct instantiation keeps the tests focused on change detection logic
        without performing actual IMAP fetches or MIME parsing.

    How:
        Fill the reference with deterministic metadata, current timestamp, UTF-8
        charset, and the provided checksum to emulate a real status email.

    Args:
        uid: UID assigned by the IMAP server to the configuration message.
        checksum: Digest of the message body used to detect textual changes.

    Returns:
        RulesMailRef: The simulated IMAP payload for change evaluation.
    """
    return RulesMailRef(
        uid=uid,
        message_id="m",
        internaldate=datetime.now(timezone.utc),
        charset="utf-8",
        body_text="version: 2\n",
        checksum=checksum,
    )


def test_has_changed_detects_uid_difference():
    """
    What:
        Ensure ``has_changed`` returns ``True`` when the incoming message exposes a
        new UID relative to the cached reference.

    Why:
        UID rotation is how IMAP signals message replacement; failing to detect it
        would leave the runtime stuck with stale configuration.

    How:
        Compare a stored reference and a simulated message with identical checksums
        but differing UIDs, asserting both change detection and reason helpers.

    Returns:
        None
    """
    prev = _ref(1, "sha256:aaa")
    latest = _mail(2, "sha256:aaa")
    assert has_changed(prev, latest) is True
    assert change_reason(prev, latest) == "uid change"


def test_has_changed_detects_checksum_difference():
    """
    What:
        Verify that checksum divergence between cached reference and latest mail
        triggers a change notification.

    Why:
        IMAP servers may reuse UIDs when flagging edits; comparing body digests
        guards against silently accepting modified YAML with the same UID.

    How:
        Construct reference and mail objects with matching UID but differing
        checksums, then assert ``has_changed`` and ``change_reason`` respond with
        the checksum-specific path.

    Returns:
        None
    """
    prev = _ref(1, "sha256:aaa")
    latest = _mail(1, "sha256:bbb")
    assert has_changed(prev, latest) is True
    assert change_reason(prev, latest) == "checksum change"


# TODO: Other modules in this repository still require the same What/Why/How documentation.

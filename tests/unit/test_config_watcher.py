from datetime import datetime, timezone

from mailai.config.schema import ConfigReference
from mailai.config.watcher import change_reason, has_changed
from mailai.imap.rules_mail import RulesMailRef


def _ref(uid: int, checksum: str) -> ConfigReference:
    return ConfigReference(uid=uid, message_id="m", internaldate=datetime.now(timezone.utc).isoformat(), checksum=checksum)


def _mail(uid: int, checksum: str) -> RulesMailRef:
    return RulesMailRef(
        uid=uid,
        message_id="m",
        internaldate=datetime.now(timezone.utc),
        charset="utf-8",
        body_text="version: 2\n",
        checksum=checksum,
    )


def test_has_changed_detects_uid_difference():
    prev = _ref(1, "sha256:aaa")
    latest = _mail(2, "sha256:aaa")
    assert has_changed(prev, latest) is True
    assert change_reason(prev, latest) == "uid change"


def test_has_changed_detects_checksum_difference():
    prev = _ref(1, "sha256:aaa")
    latest = _mail(1, "sha256:bbb")
    assert has_changed(prev, latest) is True
    assert change_reason(prev, latest) == "checksum change"

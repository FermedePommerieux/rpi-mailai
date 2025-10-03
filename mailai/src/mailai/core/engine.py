"""Deterministic rule engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

from ..config.backup import EncryptedRulesBackup
from ..config.loader import ConfigLoadError, get_runtime_config, parse_and_validate
from ..config.schema import Rule, RulesV2
from ..config.status_store import StatusStore
from ..config.watcher import change_reason, has_changed
from ..imap.client import MailAIImapClient
from ..imap.rules_mail import append_minimal_template, find_latest
from ..imap.actions import ActionRequest, SupportsActions, execute
from ..utils.logging import JsonLogger
from ..utils.regexsafe import search


@dataclass
class Message:
    """In-memory representation of an IMAP message for rule evaluation."""

    uid: int
    headers: Dict[str, str]
    body: str
    mailbox: str
    flags: Set[str] = field(default_factory=set)
    categories: Dict[str, float] = field(default_factory=dict)
    applied_rules: Set[str] = field(default_factory=set)
    has_attachment: bool = False
    size: int = 0


@dataclass
class RuleStats:
    matches: int = 0
    actions: int = 0
    errors: int = 0


@dataclass
class EngineStats:
    """Aggregated metrics returned by the engine."""

    scanned_messages: int = 0
    matched_messages: int = 0
    actions_applied: int = 0
    rule_stats: Dict[str, RuleStats] = field(default_factory=dict)


class Engine:
    """Apply configuration rules to messages and dispatch actions."""

    def __init__(
        self,
        rules: RulesV2,
        *,
        client: SupportsActions,
        logger: JsonLogger,
        run_id: str,
    ) -> None:
        self.rules = sorted(
            [rule for rule in rules.rules if rule.enabled],
            key=lambda r: (r.priority, r.id),
        )
        self.defaults = rules.defaults
        self.client = client
        self.logger = logger
        self.run_id = run_id
        self.processed_cache: Set[tuple[int, str]] = set()
        self.completed_messages: Set[int] = set()

    def process(self, messages: Iterable[Message]) -> EngineStats:
        stats = EngineStats()
        for message in messages:
            if message.uid in self.completed_messages:
                continue
            stats.scanned_messages += 1
            matched = self._process_message(message, stats)
            if matched:
                stats.matched_messages += 1
        return stats

    def _process_message(self, message: Message, stats: EngineStats) -> bool:
        matched_any = False
        for rule in self.rules:
            key = (message.uid, rule.id)
            if key in self.processed_cache or rule.id in message.applied_rules:
                continue
            if not self._matches(rule, message):
                continue
            matched_any = True
            rule_stats = stats.rule_stats.setdefault(rule.id, RuleStats())
            rule_stats.matches += 1
            actions = self._materialise_actions(rule, message)
            stop_requested = any(action.name == "stop_processing" for action in actions)
            for action in actions:
                try:
                    execute(action, client=self.client)
                    stats.actions_applied += 1
                    rule_stats.actions += 1
                except Exception:  # pragma: no cover - execution errors
                    rule_stats.errors += 1
                    self.logger.error("action_failed", uid=message.uid, rule_id=rule.id)
            self._mark_processed(message, rule)
            if stop_requested or self.defaults.stop_on_first_match:
                break
        return matched_any

    def _matches(self, rule: Rule, message: Message) -> bool:
        logic = rule.match
        result_any = True
        result_all = True
        result_none = True
        if logic.any:
            result_any = any(self._evaluate_condition(cond, message) for cond in logic.any)
        if logic.all:
            result_all = all(self._evaluate_condition(cond, message) for cond in logic.all)
        if logic.none:
            result_none = not any(
                self._evaluate_condition(cond, message) for cond in logic.none
            )
        return result_any and result_all and result_none

    def _evaluate_condition(self, condition: Dict[str, object], message: Message) -> bool:
        field, raw_payload = next(iter(condition.items()))
        if field == "has_attachment":
            return bool(message.has_attachment) == bool(raw_payload)
        payload = _normalize_payload(raw_payload)
        if field == "header":
            name = payload["name"].lower()
            value = message.headers.get(name, "")
            return self._compare_string(value, payload, default_case=self.defaults.case_sensitive)
        if field in {"subject", "body"}:
            target = message.headers.get("subject", "") if field == "subject" else message.body
            return self._compare_string(target, payload, default_case=self.defaults.case_sensitive)
        if field in {"from", "to", "cc", "bcc"}:
            value = message.headers.get(field, "")
            return self._compare_string(value, payload, default_case=False)
        if field == "mailbox":
            return self._compare_string(message.mailbox, payload, default_case=False)
        if field == "category_pred":
            label = payload["equals"]
            prob = payload.get("prob_gte", 0.0)
            return message.categories.get(label, 0.0) >= prob
        if field == "attachment_name":
            return False
        if field == "range":
            size_gt = payload.get("size_gt")
            size_lt = payload.get("size_lt")
            result = True
            if size_gt is not None:
                result = result and message.size > size_gt
            if size_lt is not None:
                result = result and message.size < size_lt
            return result
        return False

    def _compare_string(self, value: str, payload: Dict[str, object], *, default_case: bool) -> bool:
        if not default_case:
            value_cmp = value.lower()
        else:
            value_cmp = value
        if "equals" in payload:
            target = payload["equals"]
            return value_cmp == (target if default_case else str(target).lower())
        if "contains" in payload:
            target = payload["contains"]
            return (target if default_case else str(target).lower()) in value_cmp
        if "regex" in payload:
            target = payload["regex"]
            result = search(target, value, timeout_ms=50)
            return result.matched
        return False

    def _materialise_actions(self, rule: Rule, message: Message) -> List[ActionRequest]:
        actions: List[ActionRequest] = []
        quarantine = getattr(self.client, "quarantine_mailbox", None)
        for action in rule.actions:
            for name, value in _action_to_pairs(action):
                if self.defaults.dry_run and name in {"move_to", "delete"}:
                    if quarantine:
                        actions.append(ActionRequest(uid=message.uid, name="copy_to", value=quarantine))
                    continue
                actions.append(ActionRequest(uid=message.uid, name=name, value=value))
        if not self.defaults.dry_run:
            header_value = f"run={self.run_id}; rule={rule.id}"
            actions.append(ActionRequest(uid=message.uid, name="set_header", value=("X-MailAI", header_value)))
        return actions

    def _mark_processed(self, message: Message, rule: Rule) -> None:
        self.processed_cache.add((message.uid, rule.id))
        message.applied_rules.add(rule.id)
        self.completed_messages.add(message.uid)


def _action_to_pairs(action: Dict[str, object]) -> List[tuple[str, object]]:
    name, value = next(iter(action.items()))
    return [(name, value)]


def _normalize_payload(payload: object) -> Dict[str, object]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if isinstance(payload, dict):
        return payload
    raise TypeError("Unsupported payload type")


def load_active_rules(
    *,
    client: MailAIImapClient,
    status: StatusStore,
    backup: EncryptedRulesBackup,
    logger: JsonLogger,
    run_id: str,
) -> RulesV2:
    """Load the ruleset ensuring configuration safety guarantees."""

    previous = status.current_config_ref()
    latest = find_latest(client=client)
    runtime = get_runtime_config()
    if latest is None:
        restored = append_minimal_template(client=client)
        status.update_config_ref(restored)
        status.mark_restored(from_backup=False)
        status.append_event("config_restored", "missing")
        logger.warning(
            "config_restored",
            run_id=run_id,
            reason="missing",
            old_uid=previous.uid if previous else None,
            new_uid=restored.uid,
            old_checksum=previous.checksum if previous else None,
            new_checksum=restored.checksum,
        )
        backup.save(restored.body_text)
        return parse_and_validate(restored.body_text)

    size = len(latest.body_text.encode("utf-8"))
    hard_limit = runtime.mail.rules.limits.hard_limit
    if size > hard_limit:
        status.mark_invalid()
        status.append_event("config_invalid", f"oversize: {size}")
        logger.error(
            "config_invalid",
            run_id=run_id,
            reason="oversize",
            size=size,
            old_uid=previous.uid if previous else None,
            new_uid=latest.uid,
            old_checksum=previous.checksum if previous else None,
            new_checksum=latest.checksum,
        )
        restored = append_minimal_template(client=client)
        status.update_config_ref(restored, reset_errors=False)
        used_backup = backup.has_backup()
        status.mark_restored(from_backup=used_backup)
        status.append_event("config_restored", "oversize")
        logger.warning(
            "config_restored",
            run_id=run_id,
            reason="oversize",
            old_uid=previous.uid if previous else None,
            new_uid=restored.uid,
            old_checksum=previous.checksum if previous else None,
            new_checksum=restored.checksum,
        )
        if used_backup:
            fallback = backup.last_known_good()
        else:
            fallback = restored.body_text
            backup.save(restored.body_text)
        return parse_and_validate(fallback)

    if not has_changed(previous, latest):
        cached = backup.last_known_good()
        return parse_and_validate(cached)

    try:
        rules = parse_and_validate(latest.body_text)
    except ConfigLoadError as exc:
        status.mark_invalid()
        status.append_event("config_invalid", f"parse error: {exc}")
        logger.error(
            "config_invalid",
            run_id=run_id,
            reason="parse error",
            error=str(exc),
            old_uid=previous.uid if previous else None,
            new_uid=latest.uid,
            old_checksum=previous.checksum if previous else None,
            new_checksum=latest.checksum,
        )
        restored = append_minimal_template(client=client)
        status.update_config_ref(restored, reset_errors=False)
        used_backup = backup.has_backup()
        status.mark_restored(from_backup=used_backup)
        status.append_event("config_restored", "parse error")
        logger.warning(
            "config_restored",
            run_id=run_id,
            reason="parse error",
            old_uid=previous.uid if previous else None,
            new_uid=restored.uid,
            old_checksum=previous.checksum if previous else None,
            new_checksum=restored.checksum,
        )
        fallback = backup.last_known_good() if used_backup else restored.body_text
        if not used_backup:
            backup.save(restored.body_text)
        return parse_and_validate(fallback)

    backup.save(latest.body_text)
    status.update_config_ref(latest)
    status.mark_restored(from_backup=False)
    reason = change_reason(previous, latest)
    status.append_event("config_updated", reason)
    logger.info(
        "config_updated",
        run_id=run_id,
        reason=reason,
        old_uid=previous.uid if previous else None,
        new_uid=latest.uid,
        old_checksum=previous.checksum if previous else None,
        new_checksum=latest.checksum,
    )
    return rules

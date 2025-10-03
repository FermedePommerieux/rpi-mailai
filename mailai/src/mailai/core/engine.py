"""mailai.core.engine

What:
  Coordinate the deterministic rule engine that inspects IMAP messages, matches
  them against the YAML rule configuration, and schedules mailbox actions via
  the abstract action client. The module exposes data holders describing the
  evaluation context together with utilities for loading the active ruleset
  while preserving configuration safety guarantees.

Why:
  MailAI relies on a reproducible rule pipeline to avoid acting on messages
  twice and to provide an auditable trace of every automatic triage step.
  Centralising this logic here ensures configuration validation, backup
  recovery, and action scheduling follow the same invariants regardless of the
  caller (CLI once-off run, daemon loop, or tests).

How:
  - Normalise rule ordering and cache processed (message, rule) pairs to ensure
    idempotency across runs.
  - Evaluate rule predicates using resilient string and regex helpers with
    strict timeouts to avoid blocking the engine on adversarial content.
  - Materialise action requests while respecting dry-run defaults and ensuring
    every executed rule tags the message with an ``X-MailAI`` header for audit
    trails.
  - Load the most recent configuration email, enforce size limits, fall back to
    minimal templates or encrypted backups, and capture a structured status
    history for observability.

Interfaces:
  - :class:`Message`, :class:`RuleStats`, :class:`EngineStats` data containers.
  - :class:`Engine` orchestrating rule evaluation and action dispatch.
  - :func:`load_active_rules` providing a safe ruleset loader.

Invariants & Safety:
  - Never operate on IMAP sequence numbers; rule execution is keyed on stable
    UIDs and rule identifiers.
  - Every action emission in non dry-run mode must append an ``X-MailAI``
    header containing the run and rule identifiers for traceability.
  - Configuration retrieval enforces soft/hard size limits, preserves the last
    known good backup, and records recovery events for later audits.
"""
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
    """Container describing an IMAP message during rule evaluation.

    What:
      Model the subset of message metadata required by the rules engine when
      computing predicates and issuing actions.

    Why:
      Decouples engine tests from concrete IMAP objects while documenting the
      attributes an IMAP client must populate for deterministic rule matching.

    How:
      Stores headers, body, flags, mailbox metadata, and derived feature flags
      (attachments, categories, size) so predicate helpers can evaluate complex
      conditions without additional IMAP round-trips.

    Attributes:
      uid: Stable IMAP UID uniquely identifying the message.
      headers: Normalised header map used for subject and address predicates.
      body: Plain-text payload inspected by rules that trigger on content.
      mailbox: Source mailbox path used by ``mailbox`` predicates.
      flags: Mutable set of IMAP flags observed when the message was fetched.
      categories: Model-assigned category probabilities for ML-driven rules.
      applied_rules: Rules already executed for this message within the run.
      has_attachment: Whether the message reported attachments during fetch.
      size: Raw message size (bytes) for range-based rules.
    """

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
    """Per-rule counters maintained while processing a batch.

    What:
      Track how many messages matched a rule, how many actions were issued, and
      how many action executions raised errors.

    Why:
      Supports downstream observability (status emails, CLI summaries) without
      forcing the engine caller to recompute statistics from raw logs.

    How:
      The engine initialises a ``RuleStats`` entry lazily for every rule that
      matches a message and increments the counters during action dispatch.

    Attributes:
      matches: Number of messages that satisfied the rule predicates.
      actions: Number of action requests successfully executed.
      errors: Number of action executions that raised exceptions.
    """

    matches: int = 0
    actions: int = 0
    errors: int = 0


@dataclass
class EngineStats:
    """Summary statistics describing a processing run.

    What:
      Aggregate counts of scanned, matched, and acted-upon messages together
      with per-rule statistics.

    Why:
      Provide a compact representation of run results suitable for CLI output
      and for updating the persistent status store.

    How:
      Populated incrementally by :class:`Engine` while iterating over messages;
      the caller receives the final snapshot when processing completes.

    Attributes:
      scanned_messages: Total number of messages inspected this run.
      matched_messages: Count of messages with at least one rule match.
      actions_applied: Number of action requests executed.
      rule_stats: Mapping from rule identifiers to :class:`RuleStats`.
    """

    scanned_messages: int = 0
    matched_messages: int = 0
    actions_applied: int = 0
    rule_stats: Dict[str, RuleStats] = field(default_factory=dict)


class Engine:
    """Coordinate rule evaluation and action dispatch.

    What:
      Iterate over message snapshots, apply enabled rules in priority order, and
      send the resulting action requests to an IMAP action client.

    Why:
      Centralises rule evaluation, caching, and audit logging so that all entry
      points (daemon loop, manual invocation, tests) observe the same behaviour
      and safety guarantees.

    How:
      - Sort enabled rules by priority and identifier for deterministic
        processing.
      - Cache (UID, rule) combinations to prevent duplicate execution.
      - Evaluate predicates using helper functions, short-circuiting where
        applicable, and record per-rule statistics.
      - Materialise actions with dry-run handling and auditing headers before
        delegating to :func:`mailai.imap.actions.execute`.

    Attributes:
      rules: Ordered list of enabled rules to evaluate.
      defaults: Rule defaults controlling dry-run and match behaviour.
      client: Action client executing IMAP commands.
      logger: Structured logger capturing error/success events.
      run_id: Identifier included in audit headers and log lines.
    """

    def __init__(
        self,
        rules: RulesV2,
        *,
        client: SupportsActions,
        logger: JsonLogger,
        run_id: str,
    ) -> None:
        """Prepare the engine for a processing run.

        What:
          Capture the enabled ruleset, defaults, and action execution context.

        Why:
          Allows the caller to build the engine once per run while keeping the
          evaluation state isolated between runs for idempotency.

        How:
          Sort the enabled rules by priority/identifier, persist references to
          the action client and logger, and initialise tracking caches that
          prevent duplicate processing of the same (message, rule) pair.

        Args:
          rules: Fully validated ruleset including defaults.
          client: Action executor implementing ``SupportsActions``.
          logger: Structured logger for audit events and error reporting.
          run_id: Unique identifier for this execution, embedded in headers.
        """
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
        """Evaluate rules over the provided messages and run actions.

        What:
          Iterate over message snapshots, apply matching rules, and execute the
          corresponding actions while capturing run statistics.

        Why:
          Provides the primary entry point for callers that want to process a
          batch of IMAP messages without micromanaging rule evaluation state.

        How:
          Loop through messages, skip those already fully processed, delegate to
          :meth:`_process_message` for per-message logic, and increment counters
          representing engine progress.

        Args:
          messages: Iterable of :class:`Message` objects to inspect.

        Returns:
          Aggregated :class:`EngineStats` describing the processing run.
        """
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
        """Evaluate ``message`` against rules and execute resulting actions.

        What:
          Iterate over rules, determine matches, dispatch actions, and update
          processing statistics.

        Why:
          Centralising per-message orchestration ensures idempotence caches and
          statistics remain in sync, preventing duplicate actions.

        How:
          Skip rules already applied, call :meth:`_matches`, materialise actions,
          execute them with :func:`execute`, and mark the rule/message pair as
          processed. Honour ``stop_processing`` directives.

        Args:
          message: Message under evaluation.
          stats: Mutable statistics accumulator.

        Returns:
          ``True`` if any rule matched, else ``False``.
        """

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
        """Determine whether ``rule`` matches ``message``.

        What:
          Evaluate the rule's ``any``/``all``/``none`` clauses.

        Why:
          Encapsulating the logic provides a single place to enforce clause
          semantics and future extensions.

        How:
          Use :meth:`_evaluate_condition` for each clause and combine results to
          respect logical semantics.

        Args:
          rule: Rule to evaluate.
          message: Message context.

        Returns:
          ``True`` when the rule should trigger.
        """

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
        """Evaluate a single condition dictionary against ``message``.

        What:
          Support field-specific predicates such as headers, bodies, ranges, and
          category predictions.

        Why:
          Keeping condition evaluation encapsulated avoids duplicating parsing
          logic and ensures new predicate types are added coherently.

        How:
          Inspect the field key, normalise payloads via :func:`_normalize_payload`,
          and dispatch to helper comparisons, including :meth:`_compare_string`.

        Args:
          condition: Mapping from field name to predicate payload.
          message: Message instance under inspection.

        Returns:
          ``True`` if the predicate holds, otherwise ``False``.
        """

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
        """Apply string comparison operators to ``value``.

        What:
          Support ``equals``, ``contains``, and ``regex`` predicates with optional
          case sensitivity.

        Why:
          Rules rely heavily on textual matching; consolidating behaviour keeps
          case handling and regex timeouts consistent.

        How:
          Normalise ``value`` based on ``default_case`` and check the payload keys
          in priority order, invoking :func:`search` for regex evaluation.

        Args:
          value: Text to inspect.
          payload: Predicate definition.
          default_case: Whether comparisons should be case sensitive.

        Returns:
          ``True`` when the condition passes.
        """

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
        """Translate rule actions into concrete :class:`ActionRequest` objects.

        What:
          Expand action dictionaries, handle dry-run overrides, and append header
          stamping when not in dry-run mode.

        Why:
          Materialisation separates declarative rule definitions from executable
          requests, allowing the dispatcher to remain focused on IMAP mechanics.

        How:
          Iterate through ``rule.actions`` via :func:`_action_to_pairs`, redirect
          destructive operations to quarantine during dry-run, and append a
          ``set_header`` action for auditing.

        Args:
          rule: Rule being executed.
          message: Message triggering the actions.

        Returns:
          List of :class:`ActionRequest` instances ready for :func:`execute`.
        """

        actions: List[ActionRequest] = []
        quarantine = getattr(self.client, "quarantine_mailbox", None)
        for action in rule.actions:
            for name, value in _action_to_pairs(action):
                if self.defaults.dry_run and name in {"move_to", "delete"}:
                    if quarantine:
                        # SAFETY: During dry-run we redirect destructive actions to a
                        # dedicated mailbox when possible to avoid accidental data
                        # loss while still exercising the IMAP copy path.
                        actions.append(ActionRequest(uid=message.uid, name="copy_to", value=quarantine))
                    continue
                actions.append(ActionRequest(uid=message.uid, name=name, value=value))
        if not self.defaults.dry_run:
            header_value = f"run={self.run_id}; rule={rule.id}"
            actions.append(ActionRequest(uid=message.uid, name="set_header", value=("X-MailAI", header_value)))
        return actions

    def _mark_processed(self, message: Message, rule: Rule) -> None:
        """Record that ``rule`` has been applied to ``message``.

        What:
          Update caches tracking processed pairs and completed messages.

        Why:
          Prevents reprocessing loops and ensures stop-once semantics across
          daemon restarts.

        How:
          Add entries to ``processed_cache``, ``message.applied_rules``, and
          ``completed_messages``.

        Args:
          message: Message that triggered the rule.
          rule: Rule that was applied.
        """

        self.processed_cache.add((message.uid, rule.id))
        message.applied_rules.add(rule.id)
        self.completed_messages.add(message.uid)


def _action_to_pairs(action: Dict[str, object]) -> List[tuple[str, object]]:
    """Flatten a single action mapping into name/value pairs.

    What:
      Convert ``{"move_to": "Inbox"}`` style dictionaries into iterable tuples.

    Why:
      The engine stores actions as mappings for readability; execution benefits
      from explicit tuples.

    Args:
      action: Mapping representing one action.

    Returns:
      List containing a single ``(name, value)`` tuple.
    """

    name, value = next(iter(action.items()))
    return [(name, value)]


def _normalize_payload(payload: object) -> Dict[str, object]:
    """Convert payloads into dictionary form for comparison helpers.

    What:
      Accept dataclasses, Pydantic models, and raw dictionaries, returning a
      uniform ``dict``.

    Why:
      Condition payloads originate from multiple sources (YAML, models); a
      consistent structure simplifies evaluation.

    Args:
      payload: Object describing a predicate.

    Returns:
      Dictionary representation of ``payload``.

    Raises:
      TypeError: If the payload type is unsupported.
    """

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
    """Retrieve and validate the active rule configuration.

    What:
      Fetch the most recent configuration email, validate it, and return a
      parsed :class:`RulesV2` object ready for rule evaluation.

    Why:
      The configuration is stored in mailboxes and can be tampered with or grow
      beyond expected limits. This loader enforces MailAI's safety guarantees,
      including hard size limits, audit logging, and fallback to encrypted
      backups.

    How:
      - Retrieve the latest configuration message via ``find_latest``.
      - If missing, append a minimal template, persist status, and backup.
      - Reject oversized payloads, restore from backups, and record recovery
        events.
      - Parse YAML content with :func:`parse_and_validate`, capturing parse
        errors to restore known-good content.
      - Update the status store with the current configuration reference and
        change reason before returning the parsed rules.

    Args:
      client: IMAP client capable of reading/writing configuration messages.
      status: Persistent status store tracking configuration lineage.
      backup: Encrypted backup manager storing known good configurations.
      logger: Structured logger receiving audit events.
      run_id: Identifier used for logging to correlate configuration changes.

    Returns:
      A validated :class:`RulesV2` ready for engine consumption.

    Raises:
      ConfigLoadError: Propagated when validation fails even after recovery
        attempts.
    """

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

# TODO: Other modules require the same treatment (What/Why/How docstrings + module header).

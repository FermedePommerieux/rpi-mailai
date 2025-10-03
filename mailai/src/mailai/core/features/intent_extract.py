"""mailai.core.features.intent_extract

What:
  Combine lightweight heuristics with the structured local LLM interface to
  infer intent, tone, and scam risk indicators from sanitised message metadata.

Why:
  The feature store must enrich hashed records with high-level semantics while
  preserving MailAI's privacy guarantees. Mixing deterministic heuristics with a
  tightly constrained LLM prompt yields interpretable scores without persisting
  plaintext.

How:
  - Analyse aggregate counters such as exclamation density, uppercase ratios,
    reply depth, and attachment mentions to assign baseline scores and flags.
  - Build a privacy-safe prompt composed solely of structured metrics and invoke
    the local LLM for additional labels.
  - Validate the JSON response against closed vocabularies and bounded score
    ranges; raise :class:`PrivacyViolation` when the contract is breached.

Interfaces:
  - :func:`infer_intent_and_tone` returning a dictionary ready for
    :class:`IntentFeatures` construction.

Invariants & Safety:
  - No raw email text or arbitrary LLM output is returned; all identifiers stem
    from whitelisted vocabularies.
  - Scores remain inside the inclusive 0..3 range.
  - Violations trigger :class:`PrivacyViolation` so callers can abort persistence
    before privacy guarantees are compromised.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

from .schema import (
    INTENT_VOCAB,
    PERSUASION_VOCAB,
    SPEECH_ACT_VOCAB,
    SUSPICION_FLAG_VOCAB,
    ParsedMailMeta,
    TextStats,
    UrlInfo,
)
from ...utils.privacy import PrivacyViolation, assert_bounded_scores, assert_closed_vocab

_PROMPT_TEMPLATE = (
    "You are MailAI's privacy-locked classifier. You receive structured metrics "
    "derived from an email. Respond with JSON only, no prose. The JSON must "
    "include keys intent (string), speech_acts (array), persuasion (array), "
    "urgency_score (int), insistence_score (int), commercial_pressure (int), "
    "suspicion_flags (array), scam_singularity (int). Values must come from the "
    "allowed vocabularies: intent={intent_vocab}; speech_acts={speech_vocab}; "
    "persuasion={persuasion_vocab}; suspicion_flags={flag_vocab}. Scores must be "
    "integers between 0 and 3. Input metrics: {{metrics}}."
)


@dataclass(frozen=True)
class _HeuristicResult:
    """Bundle intermediate heuristic outputs for combination with LLM results."""

    intent: str
    speech_acts: Tuple[str, ...]
    persuasion: Tuple[str, ...]
    urgency_score: int
    insistence_score: int
    commercial_pressure: int
    suspicion_flags: Tuple[str, ...]
    scam_singularity: int


def infer_intent_and_tone(meta: ParsedMailMeta, stats: TextStats, urls: UrlInfo) -> Dict[str, object]:
    """Infer intent metadata while enforcing privacy constraints.

    What:
      Analyse sanitised metadata, apply deterministic heuristics, and optionally
      combine the outcome with a structured LLM completion to produce closed
      vocabulary annotations describing the message tone and risk posture.

    Why:
      The learner and routing logic benefit from intent and urgency signals, yet
      MailAI cannot persist raw content. This function bridges the gap by working
      entirely with aggregated counters and a tightly validated LLM response.

    How:
      - Compute heuristic scores based on ratios (exclamation density, capitals)
        and structural hints (reply depth, link mismatch, attachment pressure).
      - Optionally invoke the embedded LLM with a JSON-only prompt; validate the
        response strictly and merge it with the heuristics.
      - Return a mapping ready for :class:`IntentFeatures` construction.

    Args:
      meta: Sanitised metadata describing headers and attachment state.
      stats: Aggregated text statistics derived from the body.
      urls: Deduplicated domain information for hyperlinks in the body.

    Returns:
      Dictionary containing closed-vocabulary identifiers and bounded scores.

    Raises:
      PrivacyViolation: If the LLM returns unexpected keys or values.
    """

    heuristic = _run_heuristics(meta, stats, urls)
    combined = {
        "intent": heuristic.intent,
        "speech_acts": list(heuristic.speech_acts),
        "persuasion": list(heuristic.persuasion),
        "urgency_score": heuristic.urgency_score,
        "insistence_score": heuristic.insistence_score,
        "commercial_pressure": heuristic.commercial_pressure,
        "suspicion_flags": list(heuristic.suspicion_flags),
        "scam_singularity": heuristic.scam_singularity,
    }

    if meta.llm and meta.llm_settings:
        llm_payload = _call_llm(meta, stats, urls)
        _validate_llm_payload(llm_payload)
        _merge_llm_payload(combined, llm_payload)

    combined["speech_acts"] = tuple(dict.fromkeys(sorted(combined["speech_acts"])))
    combined["persuasion"] = tuple(dict.fromkeys(sorted(combined["persuasion"])))
    combined["suspicion_flags"] = tuple(dict.fromkeys(sorted(combined["suspicion_flags"])))
    combined["scam_singularity"] = int(min(3, max(0, combined["scam_singularity"])))
    return combined


def _run_heuristics(meta: ParsedMailMeta, stats: TextStats, urls: UrlInfo) -> _HeuristicResult:
    """Apply deterministic heuristics to derive baseline scores."""

    speech_acts: Set[str] = {"inform"}
    persuasion: Set[str] = {"none"}
    suspicion: Set[str] = set()
    urgency_score = 0
    insistence_score = 0
    commercial_pressure = 0

    if stats.exclamation_density > 0.015 or stats.call_to_action_score >= 2:
        urgency_score = max(urgency_score, 2)
        persuasion.add("urgency")
        suspicion.add("urgent_language")
    if stats.caps_ratio > 0.35:
        insistence_score = max(insistence_score, 2)
        suspicion.add("caps_excess")
    if meta.reply_depth + meta.relance_count >= 3:
        insistence_score = max(insistence_score, 3)
        speech_acts.add("demand")
    elif meta.reply_depth + meta.relance_count >= 1:
        insistence_score = max(insistence_score, 1)
        speech_acts.add("request")
    if meta.attachment_pressure or stats.attachment_mentions > 0:
        suspicion.add("attachment_push")
    if stats.call_to_action_score > 0:
        speech_acts.add("request")
        commercial_pressure = min(3, stats.call_to_action_score)
        if meta.has_attachments:
            persuasion.add("authority")

    if _domain_mismatch(meta.from_domain, urls.target_domains):
        suspicion.add("link_mismatch")

    scam_score = 0
    if "link_mismatch" in suspicion:
        scam_score += 1
    if stats.exclamation_density > 0.025:
        scam_score += 1
    if stats.caps_ratio > 0.45:
        scam_score += 1
    if meta.attachment_pressure:
        scam_score += 1
    scam_score = min(3, scam_score)

    return _HeuristicResult(
        intent="unknown",
        speech_acts=tuple(sorted(speech_acts)),
        persuasion=tuple(sorted(persuasion)),
        urgency_score=urgency_score,
        insistence_score=insistence_score,
        commercial_pressure=commercial_pressure,
        suspicion_flags=tuple(sorted(suspicion)),
        scam_singularity=scam_score,
    )


def _domain_mismatch(from_domain: str, targets: Tuple[str, ...]) -> bool:
    """Detect whether hyperlinks point to domains unrelated to the sender."""

    if not from_domain or not targets:
        return False
    sender = _effective_domain(from_domain)
    target_domains = {_effective_domain(domain) for domain in targets}
    return bool(target_domains and sender not in target_domains)


def _effective_domain(domain: str) -> str:
    """Reduce a domain to its last two labels for coarse comparison."""

    labels = [label for label in domain.lower().split(".") if label]
    if len(labels) >= 2:
        return ".".join(labels[-2:])
    return labels[0] if labels else ""


def _call_llm(meta: ParsedMailMeta, stats: TextStats, urls: UrlInfo) -> Dict[str, object]:
    """Invoke the structured LLM with a privacy-preserving prompt."""

    metrics = {
        "from_domain": _effective_domain(meta.from_domain),
        "reply_depth": meta.reply_depth,
        "relance_count": meta.relance_count,
        "has_attachments": meta.has_attachments,
        "attachment_pressure": meta.attachment_pressure,
        "caps_ratio": round(stats.caps_ratio, 4),
        "exclamation_density": round(stats.exclamation_density, 4),
        "call_to_action_score": stats.call_to_action_score,
        "attachment_mentions": stats.attachment_mentions,
        "target_domain_count": len(urls.target_domains),
        "link_domains": sorted({_effective_domain(d) for d in urls.target_domains}),
    }
    prompt = _PROMPT_TEMPLATE.format(
        intent_vocab=INTENT_VOCAB,
        speech_vocab=SPEECH_ACT_VOCAB,
        persuasion_vocab=PERSUASION_VOCAB,
        flag_vocab=SUSPICION_FLAG_VOCAB,
    ).replace("{metrics}", json.dumps(metrics, separators=(",", ":")))

    settings = meta.llm_settings
    assert settings is not None  # For mypy; guarded by caller.
    raw = meta.llm.structured_completion(
        prompt,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        timeout_s=settings.timeout_s,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise PrivacyViolation("LLM returned non-JSON payload") from exc
    if not isinstance(parsed, dict):
        raise PrivacyViolation("LLM payload must be a JSON object")
    return parsed


def _validate_llm_payload(payload: Dict[str, object]) -> None:
    """Ensure the LLM payload adheres to the closed vocabulary contract."""

    intent = payload.get("intent", "unknown")
    speech_acts = payload.get("speech_acts", [])
    persuasion = payload.get("persuasion", [])
    suspicion_flags = payload.get("suspicion_flags", [])
    urgency_score = payload.get("urgency_score", 0)
    insistence_score = payload.get("insistence_score", 0)
    commercial_pressure = payload.get("commercial_pressure", 0)
    scam_singularity = payload.get("scam_singularity", 0)

    assert_closed_vocab("intent", [intent], allowed=INTENT_VOCAB)
    assert_closed_vocab("speech_acts", speech_acts, allowed=SPEECH_ACT_VOCAB)
    assert_closed_vocab("persuasion", persuasion, allowed=PERSUASION_VOCAB)
    assert_closed_vocab("suspicion_flags", suspicion_flags, allowed=SUSPICION_FLAG_VOCAB)
    assert_bounded_scores("urgency_score", urgency_score, upper=3)
    assert_bounded_scores("insistence_score", insistence_score, upper=3)
    assert_bounded_scores("commercial_pressure", commercial_pressure, upper=3)
    assert_bounded_scores("scam_singularity", scam_singularity, upper=3)


def _merge_llm_payload(target: Dict[str, object], payload: Dict[str, object]) -> None:
    """Merge a validated LLM payload into the heuristic result."""

    if "intent" in payload:
        target["intent"] = payload["intent"]
    for key in ("speech_acts", "persuasion", "suspicion_flags"):
        if key in payload:
            values = payload[key]
            if isinstance(values, str):
                target[key].append(values)
            elif isinstance(values, Iterable):
                target[key].extend(str(value) for value in values)
            else:
                target[key].append(str(values))
    for key in ("urgency_score", "insistence_score", "commercial_pressure", "scam_singularity"):
        if key in payload:
            candidate = int(payload[key])
            target[key] = max(target[key], candidate)

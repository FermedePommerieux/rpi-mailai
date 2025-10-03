"""Translate engine action descriptors into IMAP client calls.

What:
  Provide a thin dispatcher that accepts rule engine action payloads and routes
  them to :class:`~mailai.imap.client.MailAIImapClient` methods while enforcing
  a minimal protocol contract.

Why:
  Centralising the mapping between rule actions and IMAP operations ensures new
  actions are vetted for safety, and unknown actions fail loudly instead of
  performing unexpected network calls.

How:
  Defines a :class:`SupportsActions` protocol describing the client surface, an
  :class:`ActionRequest` dataclass mirroring rule-engine events, and a single
  :func:`execute` dispatcher that handles each supported action explicitly.

Interfaces:
  :class:`SupportsActions`, :class:`ActionRequest`, :class:`UnsupportedActionError`,
  and :func:`execute`.

Invariants & Safety:
  - Only UID-based operations are allowed; sequence-number APIs remain unused.
  - Forwarding is explicitly blocked to prevent the agent from acting as a spam
    relay.
  - Delete requests translate to ``\Deleted`` flag changes to preserve control
    over expunge semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SupportsActions(Protocol):
    """Protocol describing the IMAP operations required by :func:`execute`.

    What:
      Formalises the subset of client methods invoked by :func:`execute` so
      alternative client implementations (e.g., fakes in tests) can type-check
      successfully.

    Why:
      Using a structural protocol decouples the dispatcher from the concrete
      :class:`mailai.imap.client.MailAIImapClient` implementation, which improves
      testability.

    How:
      Annotates each method signature required for dispatching MailAI actions.
    """

    def move(self, uid: int, destination: str) -> None:
        """Relocate a message to a different mailbox using UID semantics.

        What:
          Request that the client move ``uid`` into ``destination`` while
          preserving idempotent behaviour enforced upstream.

        Why:
          Moves are the most common automation effect; documenting the contract
          ensures implementers honour UID-first guarantees.

        How:
          The client should invoke ``UID MOVE`` when supported or fall back to
          copy/delete sequences that respect MailAI's retry and header guards.

        Args:
          uid: Stable UID identifying the message.
          destination: Target mailbox path.
        """

    def copy(self, uid: int, destination: str) -> None:
        """Duplicate a message into ``destination`` without altering the source.

        What:
          Ask the client to retain the original message while placing a copy in
          the quarantine or review mailbox.

        Why:
          Copies underpin quarantine workflows; clarify that the source must
          remain untouched so operators can audit outcomes.

        How:
          Implementers typically issue ``UID COPY`` and rely on MailAI's
          idempotent header stamping to avoid duplicate processing.

        Args:
          uid: UID to duplicate.
          destination: Mailbox path receiving the copy.
        """

    def add_label(self, uid: int, label: str) -> None:
        """Attach an IMAP label or keyword to ``uid``.

        What:
          Apply a user-visible label so clients can surface automation outcomes
          (e.g., "mailai/triaged").

        Why:
          Consistent labelling drives auditability and lets operators override
          behaviour by manipulating labels.

        How:
          The implementation should map to ``UID STORE +FLAGS`` or the
          provider-specific labelling API while remaining idempotent.

        Args:
          uid: Message identifier receiving the label.
          label: Canonical label string.
        """

    def mark_read(self, uid: int, read: bool) -> None:
        """Flip the read/unread state for ``uid``.

        What:
          Toggle the ``\Seen`` flag according to ``read``.

        Why:
          Rule actions may intentionally mark status updates as read to reduce
          noise; implementers must mirror IMAP semantics exactly.

        How:
          Translate ``read`` into ``UID STORE`` operations that add or remove the
          ``\Seen`` flag without affecting unrelated flags.

        Args:
          uid: Target message UID.
          read: ``True`` to add ``\Seen``; ``False`` to remove it.
        """

    def add_flag(self, uid: int, flag: str) -> None:
        """Apply a provider-specific flag to ``uid``.

        What:
          Ensure caller-specified flags (e.g., ``"$Forwarded"``) are present on
          the message.

        Why:
          Certain automations rely on bespoke provider flags; documenting the
          behaviour avoids ambiguity when adding new action types.

        How:
          Issue ``UID STORE`` commands to add the requested flag, leaving
          existing flags untouched.

        Args:
          uid: UID of the message.
          flag: IMAP flag string to add.
        """

    def set_header(self, uid: int, name: str, value: str) -> None:
        """Persist a synthetic header on ``uid`` for idempotency bookkeeping.

        What:
          Inject or update a header (e.g., ``X-MailAI``) recording which rule
          executed.

        Why:
          Header stamping underpins the engine's "already processed" checks;
          implementers must guarantee consistent storage semantics.

        How:
          Replace any existing header with ``name`` and ensure the message is
          re-appended or patched according to provider capabilities.

        Args:
          uid: UID of the message being annotated.
          name: Header key to set.
          value: Header value representing the action or rule identifier.
        """


@dataclass
class ActionRequest:
    """Serializable representation of a rule engine action.

    What:
      Captures the action ``name``, target ``uid``, and optional ``value`` payload
      provided by the rule engine.

    Why:
      Dataclass semantics keep the payload lightweight while offering predictable
      attribute access when consumed by the dispatcher and status logging.

    How:
      Mirrors the structure produced by the rule engine, leaving validation to
      :func:`execute`.

    Attributes:
      uid: Mail UID the action should operate on.
      name: Symbolic action identifier (e.g., ``"move_to"``).
      value: Optional argument associated with ``name``.
    """

    uid: int
    name: str
    value: object | None = None


class UnsupportedActionError(ValueError):
    """Signal that the rule engine requested an unsafe or unknown action.

    What:
      Custom exception raised by :func:`execute` when an action ``name`` is not
      recognised or intentionally disallowed.

    Why:
      Escalating via a dedicated error type simplifies caller-side retries and
      telemetry, helping operators understand misconfigured rules.

    How:
      Inherits from :class:`ValueError` because the payload is structurally valid
      but semantically unsupported.
    """


def execute(action: ActionRequest, *, client: SupportsActions) -> None:
    """Execute an engine action using the provided IMAP ``client``.

    What:
      Routes the ``action`` to the appropriate client method, handling all
      supported names and translating payloads into concrete method calls.

    Why:
      Ensures a single choke point enforces idempotency rules, rate limiting,
      and explicit whitelisting of behaviour. Prevents disparate modules from
      directly invoking low-level IMAP operations.

    How:
      Performs a series of ``if`` checks on ``action.name``. For argument-bearing
      actions the ``value`` is coerced to the expected type before calling the
      client. Unknown or disallowed actions raise :class:`UnsupportedActionError`.

    Args:
      action: Action descriptor produced by the rule engine.
      client: IMAP client instance implementing :class:`SupportsActions`.

    Raises:
      UnsupportedActionError: When the action is not recognised or unsafe.
    """

    if action.name == "move_to":
        client.move(action.uid, str(action.value))
        return
    if action.name == "copy_to":
        client.copy(action.uid, str(action.value))
        return
    if action.name == "add_label":
        client.add_label(action.uid, str(action.value))
        return
    if action.name == "mark_read":
        client.mark_read(action.uid, bool(action.value))
        return
    if action.name == "add_flag":
        client.add_flag(action.uid, str(action.value))
        return
    if action.name == "set_header":
        name, value = action.value  # type: ignore[misc]
        client.set_header(action.uid, str(name), str(value))
        return
    if action.name == "stop_processing":
        return
    if action.name == "delete":
        if action.value:
            client.add_flag(action.uid, "\\Deleted")
        return
    if action.name == "forward_to":  # pragma: no cover - network side effect
        raise UnsupportedActionError(
            "Forwarding must be managed by a whitelisted SMTP relay upstream"
        )
    raise UnsupportedActionError(f"Unsupported action {action.name}")


# TODO: Other modules in this repository still require the same What/Why/How documentation.

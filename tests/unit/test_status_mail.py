"""Status mail persistence unit tests.

What:
  Validate that :func:`mailai.imap.status_mail.upsert_status` truncates YAML
  payloads to the configured size limits while keeping the resulting status
  document parseable.

Why:
  Raspberry Pi deployments deliver ``status.yaml`` over IMAP, so runaway note or
  proposal lists could exceed provider size caps. The truncation logic must
  retain enough diagnostic context without overflowing the mailbox quota.

How:
  Generate an oversized :class:`~mailai.config.schema.StatusV2` instance, append
  it through the IMAP test double, then fetch the stored message and parse it
  back using the runtime config helpers. Assertions cover both hard byte limits
  and logical truncation markers for notes and proposals.

Interfaces:
  ``test_upsert_status_truncates_large_payload``.

Invariants & Safety:
  - Tests must keep fixtures side-effect free so other suites can reuse the
    IMAP backend without order dependencies.
  - Truncation markers count toward note totals to make the boundary explicit
    for auditors reviewing status digests.
"""

from mailai.config.loader import get_runtime_config, load_status
from mailai.config.schema import Proposal, StatusV2
from mailai.imap.status_mail import upsert_status


def test_upsert_status_truncates_large_payload(imap_client) -> None:
    """Ensure status messages respect byte limits after IMAP append.

    What:
      Builds a large status document, persists it through the IMAP client under
      test, and verifies the stored MIME payload satisfies byte caps while
      preserving parseability.

    Why:
      Oversized ``status.yaml`` bodies would be rejected by providers or stall
      synchronisation. The production implementation truncates notes and
      proposals before append, so the regression guard confirms that behaviour
      whenever limits or schema evolve.

    How:
      Requests a fake IMAP client fixture, populates 150 notes and 20 proposals,
      invokes :func:`upsert_status`, and downloads the stored message from the
      backend. The test asserts the byte length stays under the hard limit and
      that parsing the payload yields bounded note and proposal collections.

    Args:
      imap_client: Tuple fixture exposing the production IMAP client under test
        plus its fake backend.
    """

    client, backend = imap_client
    status = StatusV2.minimal()
    status.notes = [f"note {idx} " + "x" * 1024 for idx in range(150)]
    status.proposals = [
        Proposal(rule_id=f"auto-{idx}", diff="+ sample", why="test proposal")
        for idx in range(20)
    ]
    upsert_status(client, status)
    subject = get_runtime_config().mail.status.subject
    hard_limit = get_runtime_config().mail.status.limits.hard_limit
    record = next(
        entry
        for entry in backend.mailboxes[client.control_mailbox].values()
        if entry.subject == subject
    )
    payload = record.body_text.encode(record.charset)
    assert len(payload) <= hard_limit
    parsed = load_status(payload)
    assert len(parsed.model.notes) <= 21  # 20 original + truncation marker
    assert len(parsed.model.proposals) <= 8


# TODO: Other modules in this repository still require the same What/Why/How documentation.

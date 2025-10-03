"""Pytest configuration for end-to-end suites.

What:
  Establish project import paths and define fixtures that apply a canned runtime
  configuration to every test.

Why:
  The CLI integration tests execute the real ``mailai`` package as a module. To
  ensure imports resolve to the source tree rather than installed wheels, we
  prepend the ``mailai/src`` directory to ``sys.path``. The autouse fixture keeps
  configuration state deterministic between tests.

How:
  Compute the project root relative to the file, inject the source directory into
  ``sys.path`` when available, and define :func:`runtime_config` to manage the
  ``MAILAI_CONFIG_PATH`` environment variable while resetting the shared runtime
  cache before and after each test.

Interfaces:
  :func:`runtime_config` (pytest fixture).

Invariants & Safety:
  - The path injection runs once at import time and only when the source tree is
    present, avoiding pollution when the package is installed as a dependency.
  - The autouse fixture always resets the runtime configuration to prevent tests
    from leaking state across modules.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "mailai" / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

import pytest

from mailai.config.loader import reset_runtime_config

CONFIG_PATH = Path(__file__).resolve().parent / "data" / "config.yaml"


@pytest.fixture(autouse=True)
def runtime_config(monkeypatch: pytest.MonkeyPatch):
    """Apply the canned configuration file for every test.

    What:
      Sets ``MAILAI_CONFIG_PATH`` to the repository fixture, clears the runtime
      configuration cache, and ensures both happen before and after each test.

    Why:
      MailAI stores configuration globally. Without explicit resets, tests could
      interfere with each other or depend on execution order.

    How:
      Uses :class:`pytest.MonkeyPatch` to manipulate the environment, invokes
      :func:`reset_runtime_config` prior to yielding control, and guarantees the
      cache is cleared again in the ``finally`` block.

    Args:
      monkeypatch: Pytest helper injected automatically for environment control.
    """

    monkeypatch.setenv("MAILAI_CONFIG_PATH", str(CONFIG_PATH))
    reset_runtime_config()
    try:
        yield
    finally:
        reset_runtime_config()


# TODO: Other modules in this repository still require the same What/Why/How documentation.

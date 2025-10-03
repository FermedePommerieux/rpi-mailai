import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "mailai" / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

import pytest

from mailai.config.loader import reset_runtime_config

CONFIG_PATH = Path(__file__).resolve().parent / "data" / "config.cfg"


@pytest.fixture(autouse=True)
def runtime_config(monkeypatch):
    monkeypatch.setenv("MAILAI_CONFIG_PATH", str(CONFIG_PATH))
    reset_runtime_config()
    try:
        yield
    finally:
        reset_runtime_config()

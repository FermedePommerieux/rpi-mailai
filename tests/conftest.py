import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "mailai" / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

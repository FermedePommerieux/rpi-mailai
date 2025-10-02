import os
import pathlib
import subprocess
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "mailai.cli", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'mailai' / 'src'}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(cmd, text=True, capture_output=True, cwd=PROJECT_ROOT, env=env)


def test_cli_once(tmp_path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text((PROJECT_ROOT / "examples" / "rules.yaml").read_text())
    result = _run_cli("once", str(rules_path))
    assert result.returncode == 0
    assert "Loaded rules" in result.stdout


def test_cli_watch(tmp_path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text((PROJECT_ROOT / "examples" / "rules.yaml").read_text())
    result = _run_cli("watch", str(rules_path), "--interval", "5")
    assert result.returncode == 0
    assert "Watching mailbox" in result.stdout

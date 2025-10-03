"""End-to-end tests asserting CLI commands execute successfully.

What:
  Launch the ``mailai.cli`` module through ``python -m`` and validate observable
  behaviour for the ``once`` and ``watch`` commands using fixture rule files.

Why:
  These tests ensure the packaging metadata, entry point wiring, and environment
  bootstrapping work when invoked the same way operators do on Raspberry Pi
  deployments.

How:
  Construct subprocess invocations with a controlled ``PYTHONPATH`` pointing to
  the in-repo source tree, feed example rules, and assert that return codes and
  stdout messages match expectations.

Interfaces:
  ``test_cli_once``, ``test_cli_watch``.

Invariants & Safety:
  - Tests run against the local source tree to avoid depending on installed
    packages.
  - Commands must succeed without requiring external network access.
"""

import os
import pathlib
import subprocess
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Execute the MailAI CLI with the provided arguments.

    What:
      Spawns ``python -m mailai.cli`` as a subprocess and returns the completed
      process handle.

    Why:
      Running commands through a subprocess mirrors real operator usage and
      verifies packaging metadata beyond what unit tests cover.

    How:
      Builds the command list, clones the current environment while overriding
      ``PYTHONPATH`` to point at the repository source tree, and executes
      :func:`subprocess.run` capturing stdout/stderr for assertions.

    Args:
      *args: Command-line arguments to pass to ``mailai.cli``.

    Returns:
      Completed subprocess result containing return code and output.
    """

    cmd = [sys.executable, "-m", "mailai.cli", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'mailai' / 'src'}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(cmd, text=True, capture_output=True, cwd=PROJECT_ROOT, env=env)


def test_cli_once(tmp_path: pathlib.Path) -> None:
    """Verify the ``once`` command loads rules successfully.

    What:
      Executes ``mailai.cli once`` with a temporary rules file and inspects the
      resulting stdout.

    Why:
      Confirms that the CLI entry point can bootstrap the runtime configuration
      and process YAML rules without requiring IMAP connectivity.

    How:
      Writes the example rules to ``tmp_path``, invokes :func:`_run_cli`, and
      asserts that the command exits cleanly with the expected log message.

    Args:
      tmp_path: Pytest-provided temporary directory used to store rules.
    """

    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text((PROJECT_ROOT / "examples" / "rules.yaml").read_text())
    result = _run_cli("once", str(rules_path))
    assert result.returncode == 0
    assert "Loaded rules" in result.stdout


def test_cli_watch(tmp_path: pathlib.Path) -> None:
    """Ensure the ``watch`` command enters the monitoring loop.

    What:
      Invokes ``mailai.cli watch`` with a rules file and a short interval.

    Why:
      Validates that watch mode starts successfully, proving scheduler wiring and
      logging remain intact.

    How:
      Creates a rules file, runs :func:`_run_cli` with ``watch`` arguments, and
      checks the return code and stdout banner.

    Args:
      tmp_path: Temporary directory containing the generated rules file.
    """

    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text((PROJECT_ROOT / "examples" / "rules.yaml").read_text())
    result = _run_cli("watch", str(rules_path), "--interval", "5")
    assert result.returncode == 0
    assert "Watching mailbox" in result.stdout


# TODO: Other modules in this repository still require the same What/Why/How documentation.

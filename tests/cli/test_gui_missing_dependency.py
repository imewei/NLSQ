"""The GUI entry points must give an actionable hint when the gui extra is absent."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_BLOCK = 'import sys; sys.modules["PySide6"] = None; '


def _run(code: str, block: str = _BLOCK) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "NLSQ_SKIP_GPU_CHECK": "1"}
    return subprocess.run(
        [sys.executable, "-c", block + code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


@pytest.mark.serial
def test_run_desktop_without_pyside6_exits_with_hint():
    proc = _run("import nlsq.gui_qt; raise SystemExit(nlsq.gui_qt.run_desktop())")
    assert proc.returncode != 0
    assert "nlsq[gui]" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.serial
def test_cli_gui_command_without_pyside6_exits_with_hint():
    proc = _run("from nlsq.cli.main import main; raise SystemExit(main(['gui']))")
    assert proc.returncode != 0
    assert "Launching" not in proc.stdout
    assert "nlsq[gui]" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.serial
def test_run_desktop_without_pyqtgraph_exits_with_hint():
    # PySide6 stays importable; only pyqtgraph is missing.
    proc = _run(
        "import nlsq.gui_qt; raise SystemExit(nlsq.gui_qt.run_desktop())",
        block='import sys; sys.modules["pyqtgraph"] = None; ',
    )
    assert proc.returncode != 0
    assert "nlsq[gui]" in proc.stderr
    assert "Traceback" not in proc.stderr

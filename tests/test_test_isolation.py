"""Exercise the actual xdist collection hook and configured scheduler."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


@pytest.mark.serial
def test_serial_tests_in_different_modules_share_one_worker(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    options = config["tool"]["pytest"]["ini_options"]["addopts"]
    scheduler = options[options.index("--dist") + 1]
    (tmp_path / "conftest.py").write_text(
        "import runpy\n"
        f"pytest_collection_modifyitems = runpy.run_path({str(root / 'tests' / 'conftest.py')!r})"
        "['pytest_collection_modifyitems']\n"
    )
    for name in ("first", "second"):
        (tmp_path / f"test_{name}.py").write_text(
            "import pytest\n"
            "from pathlib import Path\n"
            "@pytest.mark.serial\n"
            "def test_worker(worker_id):\n"
            f"    Path({str(tmp_path / name)!r}).write_text(worker_id)\n"
        )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path),
            "-c",
            str(root / "pyproject.toml"),
            "-o",
            f"addopts=-n 2 --dist={scheduler}",
        ],
        cwd=root,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "first").read_text() == (tmp_path / "second").read_text()

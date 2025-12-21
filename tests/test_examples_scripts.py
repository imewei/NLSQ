from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "examples" / "scripts"


def discover_scripts() -> list[Path]:
    return sorted(SCRIPTS_ROOT.rglob("*.py"))


SCRIPT_PARAMS = [
    pytest.param(path, id=str(path.relative_to(SCRIPTS_ROOT)))
    for path in discover_scripts()
]


@pytest.mark.parametrize("script_path", SCRIPT_PARAMS)
def test_example_script_runs(
    script_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    env = os.environ.copy()
    env["NLSQ_EXAMPLES_QUICK"] = env.get("NLSQ_EXAMPLES_QUICK", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("NLSQ_EXAMPLES_TMPDIR", str(tmp_path))
    env.setdefault("NLSQ_EXAMPLES_MAX_SAMPLES", "100")
    env.setdefault("JAX_DISABLE_JIT", "1")
    env.setdefault("NLSQ_EXAMPLES_SKIP_ADVANCED", "0")

    if env["NLSQ_EXAMPLES_SKIP_ADVANCED"] == "1" and "09_gallery_advanced" in str(
        script_path
    ):
        pytest.skip("Skipped advanced gallery in quick mode")

    # Ensure sitecustomize quick patches are loaded
    extra_path = REPO_ROOT / "scripts" / "quick_sitecustomize"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(extra_path), env.get("PYTHONPATH", "")]
    )

    # Execute a copy of the script inside an isolated temp directory
    local_script = tmp_path / script_path.relative_to(REPO_ROOT)
    local_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_path, local_script)

    result = subprocess.run(
        [sys.executable, str(local_script)],
        check=False,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",  # Fix Windows cp1252 encoding issues with emoji
        errors="replace",  # Replace undecodable bytes instead of failing
        timeout=60,
    )

    if result.returncode != 0:
        stdout_snip = (result.stdout or "")[-800:]
        stderr_snip = (result.stderr or "")[-800:]
        pytest.fail(
            f"{script_path} failed with code {result.returncode}\n"
            f"stdout:\n{stdout_snip}\n\nstderr:\n{stderr_snip}"
        )

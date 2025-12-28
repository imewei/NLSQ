"""
Tests for example scripts execution.

These tests spawn external Python processes to validate example scripts.
Scripts are filtered at collection time for efficiency.

Performance optimizations:
- Heavy/slow scripts excluded at collection time (not runtime)
- Duplicate gallery scripts (09_gallery_advanced) excluded
- Streaming scripts excluded (tested separately in streaming tests)
- CLI model scripts excluded (not standalone examples)

Serial execution prevents resource contention with pytest-xdist.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "examples" / "scripts"

# Scripts/patterns to EXCLUDE at collection time (performance optimization)
EXCLUDED_PATTERNS = {
    # Streaming scripts - too slow, functionality tested in streaming tests
    "06_streaming",
    # Advanced gallery - duplicates 04_gallery with longer execution times
    "09_gallery_advanced",
    # Global optimization scripts - slow due to multiple fits
    "07_global_optimization",
    # CLI model definitions - not standalone example scripts
    "10_cli-commands/models",
    "10_cli-commands/data",
    "10_cli-commands/output",
}

# Scripts that need longer timeout (complex demonstrations)
LONG_TIMEOUT_PATTERNS = {
    "03_advanced",
    "08_workflow_system",
}


def should_exclude(path: Path) -> bool:
    """Check if script should be excluded from testing."""
    path_str = str(path)
    return any(pattern in path_str for pattern in EXCLUDED_PATTERNS)


def get_timeout(path: Path) -> int:
    """Get appropriate timeout for script based on complexity."""
    path_str = str(path)
    if any(pattern in path_str for pattern in LONG_TIMEOUT_PATTERNS):
        return 120  # 2 minutes for complex scripts
    return 60  # 1 minute for simple scripts


def discover_scripts() -> list[Path]:
    """Discover scripts, excluding heavy/duplicate ones at collection time."""
    all_scripts = sorted(SCRIPTS_ROOT.rglob("*.py"))
    return [s for s in all_scripts if not should_exclude(s)]


SCRIPT_PARAMS = [
    pytest.param(path, id=str(path.relative_to(SCRIPTS_ROOT)))
    for path in discover_scripts()
]


@pytest.mark.slow  # Skip in fast tests (-m "not slow")
@pytest.mark.serial  # Run on single xdist worker to prevent resource contention
@pytest.mark.parametrize("script_path", SCRIPT_PARAMS)
def test_example_script_runs(
    script_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Execute a script and verify it completes without errors."""
    env = os.environ.copy()
    env["NLSQ_EXAMPLES_QUICK"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONHASHSEED"] = "0"
    env["NLSQ_EXAMPLES_TMPDIR"] = str(tmp_path)
    env["NLSQ_EXAMPLES_MAX_SAMPLES"] = "100"
    env["JAX_DISABLE_JIT"] = "1"

    # Ensure sitecustomize quick patches are loaded
    extra_path = REPO_ROOT / "scripts" / "quick_sitecustomize"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(extra_path), env.get("PYTHONPATH", "")]
    )

    # Execute a copy of the script inside an isolated temp directory
    local_script = tmp_path / script_path.relative_to(REPO_ROOT)
    local_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_path, local_script)

    timeout = get_timeout(script_path)
    result = subprocess.run(
        [sys.executable, str(local_script)],
        check=False,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )

    if result.returncode != 0:
        stdout_snip = (result.stdout or "")[-800:]
        stderr_snip = (result.stderr or "")[-800:]
        pytest.fail(
            f"{script_path} failed with code {result.returncode}\n"
            f"stdout:\n{stdout_snip}\n\nstderr:\n{stderr_snip}"
        )

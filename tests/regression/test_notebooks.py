"""
Tests for Jupyter notebook execution.

These tests validate example notebooks by executing them via nbclient.
Notebooks are filtered at collection time for efficiency.

Performance optimizations:
- Heavy/slow notebooks excluded at collection time (not runtime)
- Duplicate gallery notebooks (09_gallery_advanced) excluded
- Streaming notebooks excluded (tested separately in streaming tests)
- Short timeout for simple notebooks, longer for complex ones

Serial execution prevents resource contention with pytest-xdist.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parents[2]
NB_ROOT = REPO_ROOT / "examples" / "notebooks"

# Notebooks/patterns to EXCLUDE at collection time (performance optimization)
EXCLUDED_PATTERNS = {
    # Streaming notebooks - too slow, functionality tested in streaming tests
    "06_streaming",
    # Advanced gallery - duplicates 04_gallery with longer execution times
    "09_gallery_advanced",
    # Specific heavy notebooks that timeout
    "defense_layers_demo",
    # Global optimization notebooks - slow due to multiple fits
    "07_global_optimization",
}

# Notebooks that need longer timeout (complex demonstrations)
LONG_TIMEOUT_PATTERNS = {
    "03_advanced",
    "large_dataset_demo",
    "performance_optimization_demo",
    "08_workflow_system",
}


def should_exclude(path: Path) -> bool:
    """Check if notebook should be excluded from testing."""
    path_str = str(path)
    return any(pattern in path_str for pattern in EXCLUDED_PATTERNS)


def get_timeout(path: Path) -> int:
    """Get appropriate timeout for notebook based on complexity."""
    path_str = str(path)
    if any(pattern in path_str for pattern in LONG_TIMEOUT_PATTERNS):
        return 180  # 3 minutes for complex notebooks
    return 90  # 90 seconds for simple notebooks


def discover_notebooks() -> list[Path]:
    """Discover notebooks, excluding heavy/duplicate ones at collection time."""
    all_notebooks = sorted(NB_ROOT.rglob("*.ipynb"))
    return [nb for nb in all_notebooks if not should_exclude(nb)]


NOTEBOOK_PARAMS = [
    pytest.param(path, id=str(path.relative_to(NB_ROOT)))
    for path in discover_notebooks()
]


@pytest.mark.slow  # Skip in fast tests (-m "not slow")
@pytest.mark.serial  # Run on single xdist worker to prevent resource contention
@pytest.mark.parametrize("notebook_path", NOTEBOOK_PARAMS)
def test_notebook_executes(notebook_path: Path, tmp_path: Path):
    """Execute a notebook and verify it completes without errors."""
    # Set environment variables for quick execution mode
    old_env = {}
    env_vars = {
        "NLSQ_EXAMPLES_QUICK": "1",
        "NLSQ_EXAMPLES_MAX_SAMPLES": "10",
        "JAX_DISABLE_JIT": "1",
        "PYTHONHASHSEED": "0",
        "MPLBACKEND": "Agg",
    }
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    # Ensure sitecustomize quick patches are discoverable
    quick_path = REPO_ROOT / "scripts" / "quick_sitecustomize"
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(quick_path), old_pythonpath]
    )

    try:
        # Copy notebook to temp directory for isolated execution
        local_nb = tmp_path / notebook_path.relative_to(REPO_ROOT)
        local_nb.parent.mkdir(parents=True, exist_ok=True)
        (local_nb.parent / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(notebook_path, local_nb)

        nb = nbformat.read(local_nb, as_version=4)
        timeout = get_timeout(notebook_path)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(local_nb.parent)}},
        )

        client.execute()
    except CellExecutionError as exc:
        raise AssertionError(f"Notebook failed: {notebook_path}\n{exc}") from exc
    finally:
        # Restore original environment
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if old_pythonpath:
            os.environ["PYTHONPATH"] = old_pythonpath
        else:
            os.environ.pop("PYTHONPATH", None)

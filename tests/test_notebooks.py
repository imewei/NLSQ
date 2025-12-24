from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_ROOT = REPO_ROOT / "examples" / "notebooks"


def discover_notebooks() -> list[Path]:
    return sorted(NB_ROOT.rglob("*.ipynb"))


NOTEBOOK_PARAMS = [
    pytest.param(path, id=str(path.relative_to(NB_ROOT)))
    for path in discover_notebooks()
]


@pytest.mark.slow  # Skip in integration tests (-m "not slow"); run in dedicated Validate Notebooks job
@pytest.mark.parametrize("notebook_path", NOTEBOOK_PARAMS)
def test_notebook_executes(notebook_path: Path, tmp_path: Path):
    # Set environment variables directly - NotebookClient doesn't use the env param,
    # the kernel inherits from the parent process
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

    # Handle skip flags
    skip_advanced = os.environ.get("NLSQ_EXAMPLES_SKIP_ADVANCED", "0")
    skip_heavy = os.environ.get("NLSQ_NOTEBOOKS_SKIP_HEAVY", "0")

    # Ensure sitecustomize quick patches are discoverable
    quick_path = REPO_ROOT / "scripts" / "quick_sitecustomize"
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(quick_path), old_pythonpath]
    )

    try:
        # Skip heavy advanced gallery notebooks in quick mode
        if skip_advanced == "1" and "09_gallery_advanced" in str(notebook_path):
            pytest.skip("Skipped advanced gallery notebook in quick mode")
        if skip_heavy == "1" and "07_global_optimization" in str(notebook_path):
            pytest.skip("Skipped heavy global optimization notebook in quick mode")

        local_nb = tmp_path / notebook_path.relative_to(REPO_ROOT)
        local_nb.parent.mkdir(parents=True, exist_ok=True)
        (local_nb.parent / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(notebook_path, local_nb)

        nb = nbformat.read(local_nb, as_version=4)
        client = NotebookClient(
            nb,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(local_nb.parent)}},
        )

        client.execute()
    except CellExecutionError as exc:
        # Truncate outputs for readability
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

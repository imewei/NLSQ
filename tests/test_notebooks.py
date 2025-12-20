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


@pytest.mark.parametrize("notebook_path", NOTEBOOK_PARAMS)
def test_notebook_executes(notebook_path: Path, tmp_path: Path):
    env = os.environ.copy()
    env.setdefault("NLSQ_EXAMPLES_QUICK", "1")
    env.setdefault("NLSQ_EXAMPLES_MAX_SAMPLES", "10")
    env.setdefault("JAX_DISABLE_JIT", "1")
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault(
        "NLSQ_NOTEBOOKS_SKIP_ADVANCED", env.get("NLSQ_EXAMPLES_SKIP_ADVANCED", "0")
    )
    env.setdefault("NLSQ_NOTEBOOKS_SKIP_HEAVY", "0")

    # Skip heavy advanced gallery notebooks in quick mode
    if env["NLSQ_NOTEBOOKS_SKIP_ADVANCED"] == "1" and "09_gallery_advanced" in str(
        notebook_path
    ):
        pytest.skip("Skipped advanced gallery notebook in quick mode")
    if env["NLSQ_NOTEBOOKS_SKIP_HEAVY"] == "1" and "07_global_optimization" in str(
        notebook_path
    ):
        pytest.skip("Skipped heavy global optimization notebook in quick mode")

    # Ensure sitecustomize quick patches are discoverable
    quick_path = REPO_ROOT / "tools" / "quick_sitecustomize"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(quick_path), env.get("PYTHONPATH", "")]
    )

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
        env=env,
    )

    try:
        client.execute()
    except CellExecutionError as exc:
        # Truncate outputs for readability
        raise AssertionError(f"Notebook failed: {notebook_path}\n{exc}") from exc

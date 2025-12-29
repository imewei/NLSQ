"""
Execute example notebooks sequentially with strict warning handling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import localinterfaces
from nbclient import NotebookClient


def _patch_localinterfaces() -> None:
    def _load_ips_no_warn(suppress_exceptions: bool = True) -> None:
        localinterfaces._load_ips_dumb()

    localinterfaces._load_ips = localinterfaces._only_once(_load_ips_no_warn)


def _kernel_env() -> dict[str, str]:
    env = os.environ.copy()
    sitecustomize_dir = Path(__file__).parent / "notebook_sitecustomize"
    env["PYTHONPATH"] = (
        f"{sitecustomize_dir}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    )
    env.setdefault("JAX_DISABLE_JIT", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("NLSQ_EXAMPLES_MAX_SAMPLES", "10")
    env.setdefault("NLSQ_EXAMPLES_QUICK", "1")
    env.setdefault("PYTHONHASHSEED", "0")
    env["PYTHONWARNINGS"] = (
        "error,ignore:There is no current event loop:DeprecationWarning,"
        "ignore::PendingDeprecationWarning"
    )
    return env


def _execute_notebook(notebook_path: Path, env: dict[str, str]) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        kernel_manager_kwargs={"env": env},
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()


def _collect_notebooks(args: list[str]) -> list[Path]:
    if args:
        return [Path(arg) for arg in args]
    return sorted(Path("examples/notebooks").rglob("*.ipynb"))


def main() -> int:
    _patch_localinterfaces()
    env = _kernel_env()
    for key in (
        "JAX_DISABLE_JIT",
        "MPLBACKEND",
        "NLSQ_EXAMPLES_MAX_SAMPLES",
        "NLSQ_EXAMPLES_QUICK",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "PYTHONWARNINGS",
    ):
        os.environ[key] = env[key]
    notebooks = _collect_notebooks(sys.argv[1:])
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1
    for notebook_path in notebooks:
        print(f"Running {notebook_path}...")
        _execute_notebook(notebook_path, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    scripts_dir = Path(__file__).parent
    sitecustomize_dir = scripts_dir / "notebook_sitecustomize"
    # Add both sitecustomize (for other potential future uses) and scripts_dir (for io_patch)
    env["PYTHONPATH"] = (
        f"{scripts_dir}{os.pathsep}{sitecustomize_dir}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
            os.pathsep
        )
    )
    env.setdefault("JAX_DISABLE_JIT", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("NLSQ_EXAMPLES_MAX_SAMPLES", "10")
    env.setdefault("NLSQ_EXAMPLES_QUICK", "1")
    env.setdefault("PYTHONHASHSEED", "0")
    env["PYTHONWARNINGS"] = (
        "error,ignore:There is no current event loop:DeprecationWarning,"
        "ignore::PendingDeprecationWarning,"
        "ignore:Jupyter is migrating its paths:DeprecationWarning,"
        "ignore:Unable to import Axes3D:UserWarning"
    )
    return env


def _get_output_path(notebook_path: Path) -> Path:
    relative_path = notebook_path.relative_to("examples/notebooks")
    output_dir = Path("examples/outputs/notebooks") / relative_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / notebook_path.name


def _execute_notebook(notebook_path: Path, env: dict[str, str]) -> bool:
    print(f"Running {notebook_path}...")
    output_path = _get_output_path(notebook_path)

    # Load notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Inject patch cell
    patch_code = """
# Auto-injected patch to redirect figures to examples/outputs
try:
    import io_patch
    io_patch.patch_savefig()
    print("Figures will be saved to examples/outputs")
except ImportError:
    print("Could not import io_patch. Figures might be saved in-place.")
except Exception as e:
    print(f"Error patching savefig: {e}")
"""
    nb.cells.insert(0, nbformat.v4.new_code_cell(patch_code))

    # Set current notebook name for patcher
    # We must copy env to avoid polluting the global env for other notebooks
    run_env = env.copy()
    run_env["NLSQ_CURRENT_SCRIPT"] = notebook_path.stem

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        kernel_manager_kwargs={"env": run_env},
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )

    success = True
    try:
        client.execute()
    except Exception as e:
        print(f"Error executing {notebook_path}: {e}", file=sys.stderr)
        success = False
    finally:
        # Save the executed notebook
        with open(output_path, "w") as f:
            nbformat.write(nb, f)
            print(f"Saved executed notebook to {output_path}")

    return success


def _collect_notebooks(args: list[str]) -> list[Path]:
    if args:
        return [Path(arg) for arg in args]
    return sorted(Path("examples/notebooks").rglob("*.ipynb"))


def main() -> int:
    _patch_localinterfaces()
    env = _kernel_env()
    # Explicitly set NLSQ_OUTPUT_DIR so notebooks can use it if they want
    env["NLSQ_OUTPUT_DIR"] = str(Path("examples/outputs").absolute())

    for key in (
        "JAX_DISABLE_JIT",
        "MPLBACKEND",
        "NLSQ_EXAMPLES_MAX_SAMPLES",
        "NLSQ_EXAMPLES_QUICK",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "PYTHONWARNINGS",
        "NLSQ_OUTPUT_DIR",
    ):
        if key in env:
            os.environ[key] = env[key]

    notebooks = _collect_notebooks(sys.argv[1:])
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    failures = [
        notebook_path
        for notebook_path in notebooks
        if not _execute_notebook(notebook_path, env)
    ]

    if failures:
        print(f"\nFailures in {len(failures)} notebooks:")
        for p in failures:
            print(f"- {p}")
        return 1

    print("\nAll notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

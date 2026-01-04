"""
Execute example scripts sequentially with strict warning handling.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _script_paths(args: list[str]) -> list[Path]:
    if args:
        return [Path(arg) for arg in args]
    return sorted(Path("examples/scripts").rglob("*.py"))


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(
        "PYTHONWARNINGS",
        "error,ignore:Jupyter is migrating its paths:DeprecationWarning,ignore:Unable to import Axes3D:UserWarning",
    )
    env.setdefault("NLSQ_EXAMPLES_QUICK", "1")
    return env


def _get_output_log_path(script_path: Path) -> Path:
    relative_path = script_path.relative_to("examples/scripts")
    output_dir = Path("examples/outputs/scripts") / relative_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (script_path.stem + ".log")


def main() -> int:
    scripts = _script_paths(sys.argv[1:])
    if not scripts:
        print("No scripts found.", file=sys.stderr)
        return 1

    env = _build_env()
    # Explicitly set NLSQ_OUTPUT_DIR so scripts can use it if they want
    env["NLSQ_OUTPUT_DIR"] = str(Path("examples/outputs").absolute())

    failures = []

    runner_script = Path(__file__).parent / "script_runner.py"

    for script in scripts:
        print(f"Running {script}...")
        log_path = _get_output_log_path(script)

        # Set current script name for the patcher
        script_env = env.copy()
        script_env["NLSQ_CURRENT_SCRIPT"] = script.stem

        with open(log_path, "w") as log_file:
            try:
                # We pipe both stdout and stderr to the log file
                # Use script_runner.py wrapper
                subprocess.run(
                    [sys.executable, str(runner_script), str(script)],
                    check=True,
                    env=script_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                print(f"Success. Output saved to {log_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error running {script}. Check {log_path}", file=sys.stderr)
                failures.append(script)
            except Exception as e:
                print(f"Unexpected error running {script}: {e}", file=sys.stderr)
                failures.append(script)

    if failures:
        print(f"\nFailures in {len(failures)} scripts:")
        for p in failures:
            print(f"- {p}")
        return 1

    print("\nAll scripts executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

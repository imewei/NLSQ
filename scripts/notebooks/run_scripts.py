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
    env.setdefault("PYTHONWARNINGS", "error")
    env.setdefault("NLSQ_EXAMPLES_QUICK", "1")
    return env


def main() -> int:
    scripts = _script_paths(sys.argv[1:])
    if not scripts:
        print("No scripts found.", file=sys.stderr)
        return 1
    env = _build_env()
    for script in scripts:
        print(f"Running {script}...")
        subprocess.run([sys.executable, str(script)], check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

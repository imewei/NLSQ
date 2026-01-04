import os
import runpy
import sys
from pathlib import Path

# Add the directory containing io_patch to path
sys.path.append(str(Path(__file__).parent))

import io_patch


def main():
    if len(sys.argv) < 2:
        print("Usage: script_runner.py <script_path> [args...]")
        sys.exit(1)

    script_path = sys.argv[1]

    # Apply patches
    try:
        io_patch.patch_savefig()
    except Exception as e:
        print(f"Failed to patch savefig: {e}")

    # Remove script_runner.py from argv so the target script sees its own args
    sys.argv = sys.argv[1:]

    # Run the script
    # We use run_path which executes in __main__
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()

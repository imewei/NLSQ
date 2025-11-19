#!/usr/bin/env python3
"""
Fix missing IPython.display imports in notebooks.

Adds 'from IPython.display import display' to notebooks that use display()
but don't have the import.

This script uses the shared notebook_utils package to avoid code duplication.
"""

import logging
import sys
from pathlib import Path

from notebook_utils import (
    create_ipython_display_import_cell,
    find_cell_with_pattern,
    has_ipython_display_import,
    read_notebook,
    uses_display,
    write_notebook,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_notebook(notebook_path: Path, dry_run: bool = False) -> bool:
    """
    Fix a single notebook by adding missing display import.

    Args:
        notebook_path: Path to notebook file
        dry_run: If True, don't write changes

    Returns:
        True if notebook was modified
    """
    # Read notebook using shared utility
    notebook = read_notebook(notebook_path)
    if notebook is None:
        return False

    cells = notebook.get("cells", [])
    if not cells:
        return False

    # Check if fix is needed
    if not uses_display(cells):
        return False

    if has_ipython_display_import(cells):
        return False

    # Add import after %matplotlib inline
    matplotlib_idx = find_cell_with_pattern(cells, "%matplotlib inline")
    if matplotlib_idx is not None:
        insert_idx = matplotlib_idx + 1
    else:
        # Fallback: add at beginning
        insert_idx = 0

    import_cell = create_ipython_display_import_cell()
    cells.insert(insert_idx, import_cell)

    # Save modified notebook using shared utility with atomic write
    if not dry_run:
        success = write_notebook(notebook_path, notebook, backup=False)
        if not success:
            logger.warning(f"Failed to save changes to {notebook_path}")
            return False

    return True


def main():
    """Fix all notebooks in examples/notebooks directory."""
    repo_root = Path(__file__).parent.parent
    notebooks_dir = repo_root / "examples" / "notebooks"

    if not notebooks_dir.exists():
        print(f"❌ Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)

    notebooks = sorted(notebooks_dir.rglob("*.ipynb"))

    if not notebooks:
        print(f"❌ No notebooks found in {notebooks_dir}")
        sys.exit(1)

    print(f"🔍 Found {len(notebooks)} notebooks to check\n")

    fixed_count = 0
    for notebook_path in notebooks:
        rel_path = notebook_path.relative_to(notebooks_dir)
        print(f"Checking: {rel_path}")

        try:
            if fix_notebook(notebook_path, dry_run=False):
                fixed_count += 1
                print("  ✓ Added IPython.display import")
            else:
                print("  • No fix needed")
        except Exception as e:
            logger.exception(f"Error processing {notebook_path}: {e}")
            print(f"  ❌ Error: {e}")

    print(f"\n{'=' * 60}")
    print(f"Fixed {fixed_count} notebook(s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

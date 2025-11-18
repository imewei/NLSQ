#!/usr/bin/env python3
"""
Fix missing IPython.display imports in notebooks.

Adds 'from IPython.display import display' to notebooks that use display()
but don't have the import.
"""

import json
import sys
from pathlib import Path


def has_ipython_display_import(cells: list[dict]) -> bool:
    """Check if notebook already imports display from IPython.display."""
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "from IPython.display import display" in source:
                return True
    return False


def uses_display(cells: list[dict]) -> bool:
    """Check if notebook uses display() function."""
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "display(" in source:
                return True
    return False


def find_matplotlib_magic_index(cells: list[dict]) -> int | None:
    """Find the index of the cell with %matplotlib inline."""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "%matplotlib inline" in source:
                return i
    return None


def create_ipython_display_import_cell() -> dict:
    """Create cell with IPython.display import."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["from IPython.display import display"],
    }


def fix_notebook(notebook_path: Path, dry_run: bool = False) -> bool:
    """
    Fix a single notebook by adding missing display import.

    Returns: True if notebook was modified
    """
    # Read notebook
    with open(notebook_path, encoding="utf-8") as f:
        notebook = json.load(f)

    cells = notebook.get("cells", [])
    if not cells:
        return False

    # Check if fix is needed
    if not uses_display(cells):
        return False

    if has_ipython_display_import(cells):
        return False

    # Add import after %matplotlib inline
    matplotlib_idx = find_matplotlib_magic_index(cells)
    if matplotlib_idx is not None:
        insert_idx = matplotlib_idx + 1
    else:
        # Fallback: add at beginning
        insert_idx = 0

    import_cell = create_ipython_display_import_cell()
    cells.insert(insert_idx, import_cell)

    # Save modified notebook
    if not dry_run:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
            f.write("\n")

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
            print(f"  ❌ Error: {e}")

    print(f"\n{'=' * 60}")
    print(f"Fixed {fixed_count} notebook(s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

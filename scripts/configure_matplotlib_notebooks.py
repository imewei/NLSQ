#!/usr/bin/env python3
"""
Configure matplotlib inline plotting in Jupyter notebooks.

Adds %matplotlib inline magic command at the beginning of notebooks
and replaces plt.show() calls with proper display pattern.

This script uses the shared notebook_utils package for common operations.
"""

import re
import sys
from pathlib import Path

from notebook_utils import (
    NotebookStats,
    create_ipython_display_import_cell,
    create_matplotlib_config_cell,
    find_cell_with_pattern,
    find_first_code_cell_index,
    has_ipython_display_import,
    has_matplotlib_magic,
    read_notebook,
    uses_display,
    write_notebook,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_figure_variable(source: list[str], show_line_idx: int) -> str:
    """
    Find the figure variable name by looking backwards from plt.show().

    Returns: figure variable name or 'plt.gcf()' as fallback
    """
    # Look backwards up to 20 lines for figure assignment
    start_idx = max(0, show_line_idx - 20)
    for i in range(show_line_idx - 1, start_idx - 1, -1):
        line = source[i]

        # Match patterns like: fig = plt.figure() or fig, ax = plt.subplots()
        fig_match = re.search(r"(\w+)\s*[,=].*plt\.(figure|subplots)", line)
        if fig_match:
            return fig_match.group(1)

    # Default: use plt.gcf() to get current figure
    return "plt.gcf()"


def replace_plt_show(source: list[str]) -> tuple[list[str], int]:
    """
    Replace plt.show() with display pattern using context-aware logic.

    Avoids false positives in comments and strings.
    Detects figure variable names dynamically.

    Returns: (modified_source, num_replacements)
    """
    modified = []
    replacements = 0

    for i, line in enumerate(source):
        # Skip comments (lines that start with # after stripping whitespace)
        stripped = line.lstrip()
        if stripped.startswith("#"):
            modified.append(line)
            continue

        # Check if line contains plt.show()
        if "plt.show()" not in line:
            modified.append(line)
            continue

        # Check for plt.show() inside string literals (basic check)
        # Count quotes before plt.show() to detect if inside string
        before_show = line.split("plt.show()")[0]
        double_quotes = before_show.count('"')
        single_quotes = before_show.count("'")

        # If odd number of quotes, we're inside a string literal
        if double_quotes % 2 == 1 or single_quotes % 2 == 1:
            modified.append(line)
            continue

        # Use regex to match plt.show() as a standalone statement
        # Pattern: optional whitespace, optional code, plt.show(), optional whitespace/newline
        match = re.match(r"^(\s*)(.*)plt\.show\(\)(.*)$", line)

        if match:
            indent = match.group(1)
            before = match.group(2).strip()
            after = match.group(3).strip()

            # Only replace if it's a standalone call (not part of complex expression)
            if before == "" and after in ["", "\n"]:
                # Find the figure variable name
                fig_var = find_figure_variable(source, i)

                # Replace with three-line pattern
                modified.append(f"{indent}plt.tight_layout()\n")
                modified.append(f"{indent}display({fig_var})\n")
                modified.append(f"{indent}plt.close({fig_var})\n")
                replacements += 1
            else:
                # Complex case - don't replace to avoid breaking code
                modified.append(line)
        else:
            modified.append(line)

    return modified, replacements


def process_notebook(notebook_path: Path, dry_run: bool = False) -> NotebookStats:
    """
    Process a single notebook.

    Returns: dict with statistics
    """
    stats = NotebookStats(
        matplotlib_magic_added=0,
        ipython_display_import_added=0,
        plt_show_replaced=0,
        cells_modified=0,
    )

    # Read notebook using shared utility
    notebook = read_notebook(notebook_path)
    if notebook is None:
        logger.warning(f"Skipping {notebook_path} due to read error")
        return stats

    cells = notebook.get("cells", [])
    if not cells:
        return stats

    # Track if we need display import (will check after replacements)
    needs_display_import = False

    # Add %matplotlib inline if not present
    matplotlib_magic_idx = None
    if not has_matplotlib_magic(cells):
        config_cell = create_matplotlib_config_cell()
        # Insert at beginning or before first code cell
        first_code_idx = find_first_code_cell_index(cells)
        cells.insert(first_code_idx, config_cell)
        matplotlib_magic_idx = first_code_idx
        stats["matplotlib_magic_added"] = 1
    else:
        # Find where %matplotlib inline is
        matplotlib_magic_idx = find_cell_with_pattern(cells, "%matplotlib inline")

    # Replace plt.show() in all code cells
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, str):
                source = [source]

            modified_source, num_replacements = replace_plt_show(source)

            if num_replacements > 0:
                cell["source"] = modified_source
                stats["plt_show_replaced"] += num_replacements
                stats["cells_modified"] += 1
                needs_display_import = True

    # Check if notebook already uses display()
    if uses_display(cells):
        needs_display_import = True

    # Add IPython.display import if needed and not present
    if needs_display_import and not has_ipython_display_import(cells):
        import_cell = create_ipython_display_import_cell()
        # Insert after %matplotlib inline magic
        if matplotlib_magic_idx is not None:
            cells.insert(matplotlib_magic_idx + 1, import_cell)
        else:
            # Fallback: insert at first code cell
            first_code_idx = find_first_code_cell_index(cells)
            cells.insert(first_code_idx, import_cell)
        stats["ipython_display_import_added"] = 1

    # Save modified notebook using shared utility with atomic write
    if not dry_run:
        success = write_notebook(notebook_path, notebook, backup=False)
        if not success:
            logger.warning(f"Failed to save changes to {notebook_path}")

    return stats


def main():
    """Process all notebooks in examples/notebooks directory."""
    # Get notebooks directory
    repo_root = Path(__file__).parent.parent
    notebooks_dir = repo_root / "examples" / "notebooks"

    if not notebooks_dir.exists():
        print(f"❌ Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)

    # Find all notebooks
    notebooks = sorted(notebooks_dir.rglob("*.ipynb"))

    if not notebooks:
        print(f"❌ No notebooks found in {notebooks_dir}")
        sys.exit(1)

    print(f"🔍 Found {len(notebooks)} notebooks to process\n")

    # Process each notebook
    total_stats = NotebookStats(
        notebooks_processed=0,
        notebooks_modified=0,
        matplotlib_magic_added=0,
        ipython_display_import_added=0,
        plt_show_replaced=0,
        cells_modified=0,
    )

    for notebook_path in notebooks:
        rel_path = notebook_path.relative_to(notebooks_dir)
        print(f"Processing: {rel_path}")

        try:
            stats = process_notebook(notebook_path, dry_run=False)
            total_stats["notebooks_processed"] += 1

            if (
                stats["matplotlib_magic_added"]
                or stats["ipython_display_import_added"]
                or stats["plt_show_replaced"]
            ):
                total_stats["notebooks_modified"] += 1
                total_stats["matplotlib_magic_added"] += stats["matplotlib_magic_added"]
                total_stats["ipython_display_import_added"] += stats[
                    "ipython_display_import_added"
                ]
                total_stats["plt_show_replaced"] += stats["plt_show_replaced"]
                total_stats["cells_modified"] += stats["cells_modified"]

                changes = []
                if stats["matplotlib_magic_added"]:
                    changes.append("added %matplotlib inline")
                if stats["ipython_display_import_added"]:
                    changes.append("added IPython.display import")
                if stats["plt_show_replaced"]:
                    changes.append(
                        f"replaced {stats['plt_show_replaced']} plt.show() calls"
                    )

                print(f"  ✓ {', '.join(changes)}")
            else:
                print("  • No changes needed")

        except Exception as e:
            logger.exception(f"Error processing {notebook_path}: {e}")
            print(f"  ❌ Error: {e}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("📊 Summary:")
    print(f"{'=' * 60}")
    print(f"Notebooks processed:          {total_stats['notebooks_processed']}")
    print(f"Notebooks modified:           {total_stats['notebooks_modified']}")
    print(f"Matplotlib magic added:       {total_stats['matplotlib_magic_added']}")
    print(
        f"IPython.display import added: {total_stats['ipython_display_import_added']}"
    )
    print(f"plt.show() replaced:          {total_stats['plt_show_replaced']}")
    print(f"Code cells modified:          {total_stats['cells_modified']}")
    print(f"{'=' * 60}")

    if total_stats["notebooks_modified"] > 0:
        print("\n✅ Notebooks successfully configured for inline plotting!")
    else:
        print("\n✅ All notebooks already properly configured!")


if __name__ == "__main__":
    main()

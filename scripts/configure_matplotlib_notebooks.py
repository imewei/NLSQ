#!/usr/bin/env python3
"""
Configure matplotlib inline plotting in Jupyter notebooks.

Adds %matplotlib inline magic command at the beginning of notebooks
and replaces plt.show() calls with proper display pattern.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def has_matplotlib_magic(cells: List[Dict]) -> bool:
    """Check if notebook already has %matplotlib inline."""
    for cell in cells:
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if '%matplotlib inline' in source:
                return True
    return False


def find_first_code_cell_index(cells: List[Dict]) -> int:
    """Find index of first code cell."""
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            return i
    return 0


def create_matplotlib_config_cell() -> Dict:
    """Create cell with matplotlib configuration."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configure matplotlib for inline plotting in VS Code/Jupyter\n",
            "# MUST come before importing matplotlib\n",
            "%matplotlib inline"
        ]
    }


def replace_plt_show(source: List[str]) -> Tuple[List[str], int]:
    """
    Replace plt.show() with display pattern.

    Returns: (modified_source, num_replacements)
    """
    modified = []
    replacements = 0
    i = 0

    while i < len(source):
        line = source[i]

        # Check if this line contains plt.show()
        if 'plt.show()' in line:
            # Get indentation
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent

            # Replace with three-line pattern
            modified.append(f"{indent_str}plt.tight_layout()\n")
            modified.append(f"{indent_str}display(fig)\n")
            modified.append(f"{indent_str}plt.close(fig)\n")
            replacements += 1
        else:
            modified.append(line)

        i += 1

    return modified, replacements


def process_notebook(notebook_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Process a single notebook.

    Returns: dict with statistics
    """
    stats = {
        'matplotlib_magic_added': 0,
        'plt_show_replaced': 0,
        'cells_modified': 0
    }

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook.get('cells', [])
    if not cells:
        return stats

    # Add %matplotlib inline if not present
    if not has_matplotlib_magic(cells):
        config_cell = create_matplotlib_config_cell()
        # Insert at beginning or before first code cell
        first_code_idx = find_first_code_cell_index(cells)
        cells.insert(first_code_idx, config_cell)
        stats['matplotlib_magic_added'] = 1

    # Replace plt.show() in all code cells
    for cell in cells:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, str):
                source = [source]

            modified_source, num_replacements = replace_plt_show(source)

            if num_replacements > 0:
                cell['source'] = modified_source
                stats['plt_show_replaced'] += num_replacements
                stats['cells_modified'] += 1

    # Save modified notebook
    if not dry_run:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
            f.write('\n')  # Add trailing newline

    return stats


def main():
    """Process all notebooks in examples/notebooks directory."""
    # Get notebooks directory
    repo_root = Path(__file__).parent.parent
    notebooks_dir = repo_root / 'examples' / 'notebooks'

    if not notebooks_dir.exists():
        print(f"❌ Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)

    # Find all notebooks
    notebooks = sorted(notebooks_dir.rglob('*.ipynb'))

    if not notebooks:
        print(f"❌ No notebooks found in {notebooks_dir}")
        sys.exit(1)

    print(f"🔍 Found {len(notebooks)} notebooks to process\n")

    # Process each notebook
    total_stats = {
        'notebooks_processed': 0,
        'notebooks_modified': 0,
        'matplotlib_magic_added': 0,
        'plt_show_replaced': 0,
        'cells_modified': 0
    }

    for notebook_path in notebooks:
        rel_path = notebook_path.relative_to(notebooks_dir)
        print(f"Processing: {rel_path}")

        try:
            stats = process_notebook(notebook_path, dry_run=False)
            total_stats['notebooks_processed'] += 1

            if stats['matplotlib_magic_added'] or stats['plt_show_replaced']:
                total_stats['notebooks_modified'] += 1
                total_stats['matplotlib_magic_added'] += stats['matplotlib_magic_added']
                total_stats['plt_show_replaced'] += stats['plt_show_replaced']
                total_stats['cells_modified'] += stats['cells_modified']

                changes = []
                if stats['matplotlib_magic_added']:
                    changes.append("added %matplotlib inline")
                if stats['plt_show_replaced']:
                    changes.append(f"replaced {stats['plt_show_replaced']} plt.show() calls")

                print(f"  ✓ {', '.join(changes)}")
            else:
                print(f"  • No changes needed")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print("📊 Summary:")
    print(f"{'='*60}")
    print(f"Notebooks processed:        {total_stats['notebooks_processed']}")
    print(f"Notebooks modified:         {total_stats['notebooks_modified']}")
    print(f"Matplotlib magic added:     {total_stats['matplotlib_magic_added']}")
    print(f"plt.show() replaced:        {total_stats['plt_show_replaced']}")
    print(f"Code cells modified:        {total_stats['cells_modified']}")
    print(f"{'='*60}")

    if total_stats['notebooks_modified'] > 0:
        print("\n✅ Notebooks successfully configured for inline plotting!")
    else:
        print("\n✅ All notebooks already properly configured!")


if __name__ == '__main__':
    main()

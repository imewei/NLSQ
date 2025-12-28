#!/usr/bin/env python3
"""
Bidirectional conversion utility for NLSQ examples.

Converts between Jupyter notebooks (.ipynb) and Python scripts (.py).
For scripts: Replaces display() with plt.savefig() for non-interactive execution.
Also supports configuring matplotlib for inline plotting in notebooks.
"""

import json
import re
import sys
import textwrap
from pathlib import Path

# ============================================================================
# Matplotlib Configuration for Notebooks
# ============================================================================


def has_matplotlib_import(cell_source):
    """Check if cell imports matplotlib."""
    source = "".join(cell_source) if isinstance(cell_source, list) else cell_source
    patterns = [
        r"import matplotlib",
        r"from matplotlib",
        r"import.*pyplot",
    ]
    return any(re.search(pattern, source) for pattern in patterns)


def has_matplotlib_magic(cell_source):
    """Check if cell already has %matplotlib inline."""
    source = "".join(cell_source) if isinstance(cell_source, list) else cell_source
    return "%matplotlib inline" in source or "%matplotlib widget" in source


def configure_matplotlib_inline(notebook_data):
    """Add %matplotlib inline before first matplotlib import.

    Returns:
        Tuple of (modified: bool, message: str)
    """
    cells = notebook_data.get("cells", [])

    # Find first cell with matplotlib import
    matplotlib_cell_idx = None
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if has_matplotlib_import(source):
                matplotlib_cell_idx = idx
                break

    if matplotlib_cell_idx is None:
        return False, "No matplotlib imports found"

    # Check if magic already exists in earlier cells
    for idx in range(matplotlib_cell_idx + 1):
        cell = cells[idx]
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if has_matplotlib_magic(source):
                return False, "Already has %matplotlib magic"

    # Insert magic comment cell before matplotlib import
    magic_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configure matplotlib for inline plotting in VS Code/Jupyter\n",
            "# MUST come before importing matplotlib\n",
            "%matplotlib inline\n",
        ],
    }

    cells.insert(matplotlib_cell_idx, magic_cell)
    notebook_data["cells"] = cells

    return True, f"Added %matplotlib inline before cell {matplotlib_cell_idx}"


def replace_plt_show_in_cell(cell_source):
    """Replace plt.show() with inline display pattern in a cell.

    Replaces:
        plt.show()

    With:
        plt.tight_layout()
        display(plt.gcf())
        plt.close()

    Returns:
        Tuple of (new_source, modified: bool)
    """
    if isinstance(cell_source, list):
        source = "".join(cell_source)
    else:
        source = cell_source

    if "plt.show()" not in source:
        return cell_source, False

    # Check if tight_layout already exists on the line before plt.show()
    already_has_tight = bool(
        re.search(r"plt\.tight_layout\(\)\s*\n\s*plt\.show\(\)", source)
    )

    if already_has_tight:
        new_source = re.sub(
            r"plt\.tight_layout\(\)\s*\n\s*plt\.show\(\)",
            "plt.tight_layout()\ndisplay(plt.gcf())\nplt.close()",
            source,
        )
    else:
        new_source = re.sub(
            r"plt\.show\(\)",
            "plt.tight_layout()\ndisplay(plt.gcf())\nplt.close()",
            source,
        )

    # Convert back to list of lines
    if isinstance(cell_source, list):
        lines = new_source.split("\n")
        if cell_source and cell_source[-1].endswith("\n"):
            new_source_list = [line + "\n" for line in lines[:-1]]
            if lines[-1]:
                new_source_list.append(lines[-1])
        else:
            new_source_list = [line + "\n" for line in lines[:-1]]
            if lines[-1]:
                new_source_list.append(lines[-1])
        return new_source_list, True
    else:
        return new_source, True


def configure_notebook_matplotlib(notebook_path):
    """Configure a notebook for matplotlib inline plotting.

    Adds %matplotlib inline and replaces plt.show() calls.
    """
    print(f"Configuring: {notebook_path}")

    with open(notebook_path, encoding="utf-8") as f:
        notebook_data = json.load(f)

    # Add %matplotlib inline
    magic_added, magic_msg = configure_matplotlib_inline(notebook_data)

    # Replace plt.show() in all cells
    cells = notebook_data.get("cells", [])
    show_count = 0
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            new_source, modified = replace_plt_show_in_cell(source)
            if modified:
                cell["source"] = new_source
                show_count += 1

    if magic_added or show_count > 0:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook_data, f, indent=1, ensure_ascii=False)
            f.write("\n")

        print(f"  ✓ {magic_msg}")
        print(f"  ✓ Replaced {show_count} plt.show() calls")
        return True
    else:
        print(f"  ⊘ No changes needed ({magic_msg}, no plt.show())")
        return False


# ============================================================================
# Notebook/Script Conversion
# ============================================================================


def post_process_script_code(code: str, script_name: str, fig_counter: dict) -> str:
    """Post-process script code to replace display() with savefig().

    Args:
        code: Source code from notebook cell
        script_name: Name of the script (without extension)
        fig_counter: Dictionary to track figure numbers across cells

    Returns:
        Modified code with savefig() instead of display()
    """
    # Skip %matplotlib magic (only needed in notebooks)
    if "%matplotlib" in code:
        code = re.sub(r"^%matplotlib.*\n?", "", code, flags=re.MULTILINE)

    # Replace display(plt.gcf()) with plt.savefig()
    if "display(plt.gcf())" in code:
        # Increment figure counter
        fig_counter["count"] += 1
        fig_num = fig_counter["count"]

        # Create figure path
        _fig_path = f"figures/{script_name}/fig_{fig_num:02d}.png"

        # Replace the pattern
        replacement = (
            f"# Save figure to file\n"
            f"fig_dir = Path(__file__).parent / 'figures' / '{script_name}'\n"
            f"fig_dir.mkdir(parents=True, exist_ok=True)\n"
            f"plt.savefig(fig_dir / 'fig_{fig_num:02d}.png', dpi=300, bbox_inches='tight')"
        )

        code = code.replace("display(plt.gcf())", replacement)

        # Ensure Path is imported
        if "from pathlib import Path" not in fig_counter.get("imports", set()):
            fig_counter.setdefault("imports", set()).add("from pathlib import Path")

    return code


def notebook_to_script(notebook_path: Path, output_path: Path | None = None) -> Path:
    """Convert Jupyter notebook to Python script.

    For scripts, this function:
    - Removes %matplotlib magic commands (notebook-only)
    - Replaces display(plt.gcf()) with plt.savefig()
    - Creates figure directories automatically
    - Numbers figures sequentially
    """
    with open(notebook_path) as f:
        notebook = json.load(f)

    if output_path is None:
        # Determine output path based on notebook structure
        # notebooks/ -> scripts/
        notebook_str = str(notebook_path)
        if "/notebooks/" in notebook_str:
            output_path = Path(
                notebook_str.replace("/notebooks/", "/scripts/")
            ).with_suffix(".py")
        else:
            output_path = notebook_path.with_suffix(".py")

    script_name = output_path.stem

    python_lines = [
        '"""',
        f"Converted from {notebook_path.name}",
        "",
        "This script was automatically generated from a Jupyter notebook.",
        "Plots are saved to the figures/ directory instead of displayed inline.",
        '"""',
        "",
    ]

    # Track figure counter and required imports
    fig_counter = {"count": 0, "imports": set()}

    # Process all cells first to collect imports
    processed_cells = []
    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else source

        if not source_text.strip():
            continue

        if cell_type == "markdown":
            processed_cells.append(("markdown", source_text))
        elif cell_type == "code":
            processed_code = post_process_script_code(
                source_text, script_name, fig_counter
            )
            processed_cells.append(("code", processed_code))

    # Add required imports at the top
    if fig_counter.get("imports"):
        python_lines.extend(sorted(fig_counter["imports"]))
        python_lines.append("")

    # Add processed cells
    for cell_type, content in processed_cells:
        if cell_type == "markdown":
            python_lines.extend(
                [
                    "",
                    "# " + "=" * 70,
                    *[f"# {line}" for line in content.split("\n")],
                    "# " + "=" * 70,
                    "",
                ]
            )
        else:  # code
            python_lines.extend(["", content.rstrip(), ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(python_lines))
    return output_path


def script_to_notebook(script_path: Path, output_path: Path | None = None) -> Path:
    """Convert Python script to Jupyter notebook."""
    content = script_path.read_text()

    if output_path is None:
        output_path = script_path.with_suffix(".ipynb")

    cells = []
    lines = content.split("\n")
    i = 0

    # Skip module docstring
    if lines[0].strip().startswith('"""'):
        while i < len(lines) and '"""' not in lines[i][1:]:
            i += 1
        i += 1

    current_block = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("# " + "=" * 70):
            # Start of markdown section
            if current_block:
                # Dedent code block to remove common leading whitespace
                code_text = "\n".join(current_block)
                code_text = textwrap.dedent(code_text)
                cells.append(
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [code_text + "\n"],
                    }
                )
                current_block = []

            # Collect markdown
            i += 1
            markdown_lines = []
            while i < len(lines) and not lines[i].strip().startswith("# " + "=" * 70):
                markdown_lines.append(lines[i].lstrip("# "))
                i += 1

            if markdown_lines:
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [line + "\n" for line in markdown_lines],
                    }
                )

        elif line.strip() and not line.strip().startswith("#"):
            # Code line
            current_block.append(line)

        i += 1

    if current_block:
        # Dedent code block to remove common leading whitespace
        code_text = "\n".join(current_block)
        code_text = textwrap.dedent(code_text)
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [code_text + "\n"],
            }
        )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)

    return output_path


def convert_directory(directory: Path, mode: str):
    """Convert all files in a directory."""
    if mode == "notebook-to-script":
        pattern = "*.ipynb"
        converter = notebook_to_script
    elif mode == "script-to-notebook":
        pattern = "*.py"
        converter = script_to_notebook
    elif mode == "configure-matplotlib":
        pattern = "*.ipynb"
        converter = configure_notebook_matplotlib
    else:
        print(f"Error: Invalid mode '{mode}'")
        return

    files = list(directory.rglob(pattern))

    if not files:
        print(f"No {pattern} files found in {directory}")
        return

    if mode == "configure-matplotlib":
        print(f"Configuring {len(files)} notebooks for matplotlib inline plotting...")
        modified_count = 0
        for file_path in files:
            try:
                if converter(file_path):
                    modified_count += 1
            except Exception as e:
                print(f"  ✗ Error configuring {file_path.name}: {e}")
        print("=" * 80)
        print(f"Modified {modified_count} / {len(files)} notebooks")
    else:
        print(f"Converting {len(files)} files...")
        for file_path in files:
            try:
                output = converter(file_path)
                print(f"  ✓ {file_path.name} → {output.name}")
            except Exception as e:
                print(f"  ✗ Error converting {file_path.name}: {e}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_examples.py <mode> [path]")
        print()
        print("Modes:")
        print("  notebook-to-script     Convert .ipynb to .py")
        print("  script-to-notebook     Convert .py to .ipynb")
        print(
            "  configure-matplotlib   Configure notebooks for inline matplotlib plotting"
        )
        print()
        print("Examples:")
        print("  python convert_examples.py notebook-to-script example.ipynb")
        print("  python convert_examples.py script-to-notebook example.py")
        print("  python convert_examples.py notebook-to-script examples/notebooks/")
        print("  python convert_examples.py configure-matplotlib examples/notebooks/")
        sys.exit(1)

    mode = sys.argv[1]

    # For configure-matplotlib, path is optional (defaults to examples/notebooks/)
    if len(sys.argv) < 3:
        if mode == "configure-matplotlib":
            path = Path("examples/notebooks")
        else:
            print("Error: Path argument required for this mode")
            sys.exit(1)
    else:
        path = Path(sys.argv[2])

    valid_modes = ["notebook-to-script", "script-to-notebook", "configure-matplotlib"]
    if mode not in valid_modes:
        print(f"Error: Invalid mode '{mode}'")
        print(f"Valid modes: {', '.join(valid_modes)}")
        sys.exit(1)

    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)

    if path.is_dir():
        convert_directory(path, mode)
    elif mode == "notebook-to-script":
        output = notebook_to_script(path)
        print(f"✓ Converted: {path.name} → {output.name}")
    elif mode == "script-to-notebook":
        output = script_to_notebook(path)
        print(f"✓ Converted: {path.name} → {output.name}")
    elif mode == "configure-matplotlib":
        configure_notebook_matplotlib(path)
    else:
        print(f"Error: Mode '{mode}' not supported for single files")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bidirectional conversion utility for NLSQ examples.

Converts between Jupyter notebooks (.ipynb) and Python scripts (.py).
"""

import json
import sys
from pathlib import Path


def notebook_to_script(notebook_path: Path, output_path: Path | None = None) -> Path:
    """Convert Jupyter notebook to Python script."""
    with open(notebook_path) as f:
        notebook = json.load(f)

    if output_path is None:
        output_path = notebook_path.with_suffix(".py")

    python_lines = [
        '"""',
        f"Converted from {notebook_path.name}",
        "",
        "This script was automatically generated from a Jupyter notebook.",
        '"""',
        "",
    ]

    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else source

        if not source_text.strip():
            continue

        if cell_type == "markdown":
            python_lines.extend(
                [
                    "",
                    "# " + "=" * 70,
                    *[f"# {line}" for line in source_text.split("\n")],
                    "# " + "=" * 70,
                    "",
                ]
            )
        elif cell_type == "code":
            python_lines.extend(["", source_text.rstrip(), ""])

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
    in_comment_section = False

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("# " + "=" * 70):
            # Start of markdown section
            if current_block:
                cells.append(
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": ["\n".join(current_block) + "\n"],
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
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["\n".join(current_block) + "\n"],
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
    else:
        pattern = "*.py"
        converter = script_to_notebook

    files = list(directory.rglob(pattern))

    if not files:
        print(f"No {pattern} files found in {directory}")
        return

    print(f"Converting {len(files)} files...")
    for file_path in files:
        try:
            output = converter(file_path)
            print(f"  ✓ {file_path.name} → {output.name}")
        except Exception as e:
            print(f"  ✗ Error converting {file_path.name}: {e}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 3:
        print("Usage: python convert_examples.py <mode> <path>")
        print()
        print("Modes:")
        print("  notebook-to-script  Convert .ipynb to .py")
        print("  script-to-notebook  Convert .py to .ipynb")
        print()
        print("Examples:")
        print("  python convert_examples.py notebook-to-script example.ipynb")
        print("  python convert_examples.py script-to-notebook example.py")
        print("  python convert_examples.py notebook-to-script examples/notebooks/")
        sys.exit(1)

    mode = sys.argv[1]
    path = Path(sys.argv[2])

    if mode not in ["notebook-to-script", "script-to-notebook"]:
        print(f"Error: Invalid mode '{mode}'")
        print("Valid modes: notebook-to-script, script-to-notebook")
        sys.exit(1)

    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)

    if path.is_dir():
        convert_directory(path, mode)
    else:
        if mode == "notebook-to-script":
            output = notebook_to_script(path)
        else:
            output = script_to_notebook(path)
        print(f"✓ Converted: {path.name} → {output.name}")


if __name__ == "__main__":
    main()

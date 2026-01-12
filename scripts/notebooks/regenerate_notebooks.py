#!/usr/bin/env python3
"""
Regenerate notebooks from scripts with proper structure.

This handles Python scripts by:
1. Creating a proper header cell with metadata from docstring
2. Creating import cells
3. Creating function definition cells
4. Creating the main logic cells from top-level code or main() body
"""

import ast
import json
import re
import sys
import textwrap
from pathlib import Path


def extract_docstring(source: str) -> str | None:
    """Extract module docstring from source."""
    try:
        tree = ast.parse(source)
        return ast.get_docstring(tree)
    except SyntaxError:
        return None


def split_script_sections(source: str) -> dict:
    """Split script into sections: docstring, imports, constants, functions, main_body."""
    lines = source.split("\n")

    sections = {
        "docstring": "",
        "imports": [],
        "constants": [],  # Top-level assignments like QUICK
        "functions": [],
        "main_body": [],
    }

    # Extract docstring
    docstring = extract_docstring(source)
    if docstring:
        sections["docstring"] = docstring

    # Parse the source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sections

    # Track line ranges for different sections
    import_lines = set()
    constant_ranges = []  # (start, end, name) for top-level assignments
    function_ranges = []
    main_start = None
    main_end = None
    docstring_end = 0

    # Find docstring end line
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    ):
        docstring_end = tree.body[0].end_lineno

    # First pass: find where the first function starts
    first_func_line = float("inf")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name != "main":
                first_func_line = min(first_func_line, node.lineno)
                break

    # Walk through top-level nodes only
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for lineno in range(node.lineno, node.end_lineno + 1):
                import_lines.add(lineno)
        elif isinstance(node, ast.FunctionDef):
            if node.name == "main":
                main_start = node.lineno
                main_end = node.end_lineno
            else:
                function_ranges.append((node.lineno, node.end_lineno, node.name))
        elif isinstance(node, ast.Assign):
            # Capture top-level assignments like QUICK = ...
            # Only capture if BEFORE the first function definition
            # Skip FIG_DIR, fig_dir, __file__ related ones
            if node.lineno < first_func_line:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name.upper() not in (
                            "FIG_DIR",
                            "SCRIPT_DIR",
                        ) and "__file__" not in ast.unparse(node):
                            constant_ranges.append((node.lineno, node.end_lineno, name))
        elif isinstance(node, ast.If):
            # Capture short if QUICK: blocks (early announcements, not main body logic)
            # Only before first function
            if node.lineno < first_func_line:
                if isinstance(node.test, ast.Name) and node.test.id == "QUICK":
                    # Only capture short blocks (1-3 lines) that are likely just announcements
                    block_size = node.end_lineno - node.lineno + 1
                    if block_size <= 3:
                        constant_ranges.append(
                            (node.lineno, node.end_lineno, "QUICK_CHECK")
                        )

    # Find the last function/import/constant line to determine where main body starts
    last_preamble_line = docstring_end
    for start, end, _ in function_ranges:
        last_preamble_line = max(last_preamble_line, end)
    for start, end, _ in constant_ranges:
        last_preamble_line = max(last_preamble_line, end)
    if import_lines:
        max_import_line = max(import_lines)
        last_preamble_line = max(last_preamble_line, max_import_line)

    # Build imports section
    for i, line in enumerate(lines, 1):
        if i in import_lines:
            sections["imports"].append(line)

    # Extract constants
    constant_lines = set()
    for start, end, _ in constant_ranges:
        for i in range(start, end + 1):
            constant_lines.add(i)
    for i, line in enumerate(lines, 1):
        if i in constant_lines:
            sections["constants"].append(line)

    # Extract functions (excluding main)
    for start, end, name in function_ranges:
        func_lines = lines[start - 1 : end]
        sections["functions"].append("\n".join(func_lines))

    # Extract main body
    if main_start and main_end:
        # Script has a main() function - extract and dedent its body
        main_lines = lines[main_start:main_end]  # Skip the 'def main():' line

        # Skip docstring at the start of main() if present
        body_start = 0
        in_docstring = False
        for i, line in enumerate(main_lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                if stripped.count(quote) >= 2 and len(stripped) > 3:
                    continue  # Single-line docstring, skip it
                # Multi-line docstring - find the end
                in_docstring = True
                for j in range(i + 1, len(main_lines)):
                    if quote in main_lines[j]:
                        body_start = j + 1
                        break
                break
            elif stripped and not stripped.startswith("#"):
                body_start = i
                break

        # Dedent by 4 spaces (standard Python function indentation)
        # We can't use textwrap.dedent because multi-line strings inside may have 0 indent
        main_body = []
        for line in main_lines[body_start:]:
            if len(line) >= 4 and line[:4] == "    ":
                main_body.append(line[4:])
            elif not line.strip():
                main_body.append("")  # Preserve empty lines
            else:
                main_body.append(
                    line
                )  # Keep as-is (e.g., content inside multi-line strings)

        sections["main_body"] = main_body
    else:
        # Script has no main() function - use all top-level code after functions/imports
        # Collect all lines that are not imports, functions, or docstring
        used_lines = set()

        # Mark docstring lines
        for i in range(1, docstring_end + 1):
            used_lines.add(i)

        # Mark import lines
        used_lines.update(import_lines)

        # Mark function lines
        for start, end, _ in function_ranges:
            for i in range(start, end + 1):
                used_lines.add(i)

        # Also mark if __name__ == "__main__": block
        for node in tree.body:
            if isinstance(node, ast.If):
                # Check if this is the if __name__ == "__main__": block
                if isinstance(node.test, ast.Compare):
                    left = node.test.left
                    if isinstance(left, ast.Name) and left.id == "__name__":
                        for i in range(node.lineno, node.end_lineno + 1):
                            used_lines.add(i)

        # Collect remaining lines as main body
        main_body_lines = []
        for i, line in enumerate(lines, 1):
            if i not in used_lines:
                main_body_lines.append(line)

        sections["main_body"] = main_body_lines

    return sections


def create_notebook_cell(
    cell_type: str, source: str | list, cell_id: str = None
) -> dict:
    """Create a notebook cell."""
    if isinstance(source, list):
        source_list = [
            line + "\n" if not line.endswith("\n") else line for line in source
        ]
    else:
        source_list = [source + "\n"] if not source.endswith("\n") else [source]

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_list,
    }

    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

    if cell_id:
        cell["id"] = cell_id

    return cell


def extract_title_from_docstring(docstring: str) -> tuple[str, str]:
    """Extract title and description from docstring."""
    lines = docstring.strip().split("\n")
    title = lines[0] if lines else "Untitled"
    description = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return title, description


def script_to_notebook_robust(
    script_path: Path, output_path: Path | None = None
) -> Path:
    """Convert Python script to Jupyter notebook with robust handling."""
    content = script_path.read_text()

    if output_path is None:
        output_path = script_path.with_suffix(".ipynb")

    sections = split_script_sections(content)
    cells = []

    # 1. Header markdown cell
    title, description = extract_title_from_docstring(sections["docstring"])
    header_md = f"# {title}\n\n{description}" if description else f"# {title}"
    cells.append(create_notebook_cell("markdown", header_md, "header"))

    # 2. Setup/matplotlib config cell (for notebooks)
    setup_cell = """# Configure matplotlib for inline plotting
%matplotlib inline"""
    cells.append(create_notebook_cell("code", setup_cell, "setup"))

    # 3. Imports cell
    if sections["imports"]:
        imports = [line for line in sections["imports"] if line.strip()]
        if imports:
            cells.append(create_notebook_cell("code", "\n".join(imports), "imports"))

    # 4. Constants cell (QUICK, etc.)
    if sections["constants"]:
        constants = [line for line in sections["constants"] if line.strip()]
        if constants:
            cells.append(
                create_notebook_cell("code", "\n".join(constants), "constants")
            )

    # 5. Function definition cells
    for i, func in enumerate(sections["functions"]):
        cells.append(create_notebook_cell("code", func, f"func_{i}"))

    # 5. Main body cells - split on section markers
    main_body = "\n".join(sections["main_body"])

    # Protect multi-line strings from section splitting
    # Find all triple-quoted strings and replace with placeholders
    string_placeholders = {}
    placeholder_pattern = "___STRING_PLACEHOLDER_{}_"

    def protect_multiline_strings(text: str) -> str:
        """Replace triple-quoted strings with placeholders."""
        result = text
        # Match triple-quoted strings (both """ and ''')
        for quote in ['"""', "'''"]:
            pattern = re.escape(quote) + r"[\s\S]*?" + re.escape(quote)
            matches = list(re.finditer(pattern, result))
            for idx, match in enumerate(reversed(matches)):
                placeholder = placeholder_pattern.format(len(string_placeholders))
                string_placeholders[placeholder] = match.group(0)
                result = result[: match.start()] + placeholder + result[match.end() :]
        return result

    def restore_multiline_strings(text: str) -> str:
        """Restore triple-quoted strings from placeholders."""
        result = text
        for placeholder, original in string_placeholders.items():
            result = result.replace(placeholder, original)
        return result

    main_body_protected = protect_multiline_strings(main_body)

    # Split on section markers (# ========= or # --------- or print("=" * ))
    section_pattern = r'^(?:\s*#\s*[=\-]{40,}.*$|print\s*\(\s*["\'][=\-]{50,})'
    parts = re.split(section_pattern, main_body_protected, flags=re.MULTILINE)

    for i, raw_part in enumerate(parts):
        # Restore multi-line strings before processing
        code = restore_multiline_strings(raw_part)
        code = code.strip()
        if not code:
            continue

        # Skip if __name__ == "__main__" blocks
        if "__name__" in code and "__main__" in code:
            continue

        # Clean up the code
        # Remove FIG_DIR references and replace savefig with show
        code = re.sub(r"plt\.savefig\s*\([^)]+\)", "plt.show()", code)
        code = re.sub(r'print\s*\(\s*f?["\'].*Saved:.*\)', "", code)
        # Remove fig_dir/FIG_DIR related lines
        code = re.sub(r"^\s*fig_dir\s*=.*\n?", "", code, flags=re.MULTILINE)
        code = re.sub(r"^\s*fig_dir\.mkdir.*\n?", "", code, flags=re.MULTILINE)
        code = re.sub(r"^\s*FIG_DIR\s*=.*\n?", "", code, flags=re.MULTILINE)
        code = re.sub(
            r"^\s*out_path\s*=\s*FIG_DIR\s*/.*\n?", "", code, flags=re.MULTILINE
        )
        code = re.sub(
            r"^\s*out_path\s*=\s*fig_dir\s*/.*\n?", "", code, flags=re.MULTILINE
        )
        # Remove sys.exit() calls (not valid in notebooks)
        code = re.sub(r"sys\.exit\s*\([^)]*\)", "pass", code)
        # Remove bare 'return' statements (not valid outside functions)
        code = re.sub(
            r"^(\s*)return\s*$",
            r"\1pass  # early exit in quick mode",
            code,
            flags=re.MULTILINE,
        )

        if code.strip():
            cells.append(create_notebook_cell("code", code, f"code_{i}"))

    # Build notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
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


def main():
    """Process all scripts in the specified directories."""
    if len(sys.argv) < 2:
        print("Usage: python regenerate_notebooks.py <directory>")
        print(
            "Example: python regenerate_notebooks.py examples/scripts/07_global_optimization"
        )
        sys.exit(1)

    script_dir = Path(sys.argv[1])
    if not script_dir.exists():
        print(f"Error: Directory not found: {script_dir}")
        sys.exit(1)

    # Get corresponding notebook directory
    notebook_dir = Path(str(script_dir).replace("/scripts/", "/notebooks/"))
    notebook_dir.mkdir(parents=True, exist_ok=True)

    scripts = list(script_dir.glob("*.py"))
    if not scripts:
        # Try recursive
        scripts = list(script_dir.rglob("*.py"))

    print(f"Converting {len(scripts)} scripts...")

    for script in scripts:
        # Determine output path
        rel_path = script.relative_to(script_dir)
        output_path = notebook_dir / rel_path.with_suffix(".ipynb")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = script_to_notebook_robust(script, output_path)
            print(f"  ✓ {script.name} → {result.name}")
        except Exception as e:
            print(f"  ✗ {script.name}: {e}")


if __name__ == "__main__":
    main()

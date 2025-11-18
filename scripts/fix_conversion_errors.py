#!/usr/bin/env python3
"""Fix syntax errors in converted notebook scripts.

This script fixes two types of errors from the notebook conversion:
1. Indentation errors in figure-saving code blocks
2. Multiple module docstrings (removes secondary docstrings)
"""

from pathlib import Path


def fix_indentation_errors():
    """Fix indentation errors in figure-saving code blocks."""

    fixes = [
        {
            "file": "examples/scripts/02_core_tutorials/performance_optimization_demo.py",
            "line": 340,
            "old_indent": 0,
            "new_indent": 4,
            "num_lines": 7,  # Lines 340-346
        },
        {
            "file": "examples/scripts/01_getting_started/nlsq_interactive_tutorial.py",
            "line": 636,
            "old_indent": 0,
            "new_indent": 4,
            "num_lines": 5,  # Lines 636-640
        },
        {
            "file": "examples/scripts/02_core_tutorials/advanced_features_demo.py",
            "line": 793,
            "old_indent": 0,
            "new_indent": 8,
            "num_lines": 7,  # Lines 793-799
        },
        {
            "file": "examples/scripts/02_core_tutorials/nlsq_2d_gaussian_demo.py",
            "line": 520,
            "old_indent": 0,
            "new_indent": 8,
            "num_lines": 9,  # Lines 520-528
        },
    ]

    repo_root = Path(__file__).parent.parent

    for fix in fixes:
        file_path = repo_root / fix["file"]
        print(f"Fixing {file_path.name} at line {fix['line']}...")

        # Read file
        with open(file_path) as f:
            lines = f.readlines()

        # Fix indentation
        start_idx = fix["line"] - 1  # Convert to 0-indexed
        end_idx = start_idx + fix["num_lines"]

        for i in range(start_idx, end_idx):
            if i < len(lines):
                # Remove old indentation and add new
                line = lines[i].lstrip()
                if line and not line.startswith("#"):  # Don't indent empty lines
                    lines[i] = " " * fix["new_indent"] + line
                elif line.startswith("#"):
                    # Comments should also be indented
                    lines[i] = " " * fix["new_indent"] + line

        # Write back
        with open(file_path, "w") as f:
            f.writelines(lines)

        print("  ✓ Fixed indentation")


def fix_multiple_docstrings():
    """Remove secondary docstrings that appear after the initial module docstring."""

    files_with_double_docstrings = [
        "examples/scripts/03_advanced/nlsq_challenges.py",
        "examples/scripts/03_advanced/gpu_optimization_deep_dive.py",
        "examples/scripts/03_advanced/troubleshooting_guide.py",
        "examples/scripts/03_advanced/custom_algorithms_advanced.py",
        "examples/scripts/03_advanced/time_series_analysis.py",
        "examples/scripts/03_advanced/research_workflow_case_study.py",
        "examples/scripts/02_core_tutorials/large_dataset_demo.py",
    ]

    repo_root = Path(__file__).parent.parent

    for file_rel_path in files_with_double_docstrings:
        file_path = repo_root / file_rel_path
        print(f"Fixing {file_path.name}...")

        # Read file
        with open(file_path) as f:
            content = f.read()

        # Find and remove the secondary docstring
        # Pattern: After first docstring and imports, there's a standalone """..."""
        lines = content.split("\n")

        # Find the second docstring (after the module docstring)
        in_first_docstring = False
        first_docstring_ended = False
        second_docstring_start = None
        second_docstring_end = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Track first docstring
            if i == 0 and stripped.startswith('"""'):
                in_first_docstring = True
            elif in_first_docstring and '"""' in line and i > 0:
                first_docstring_ended = True
                in_first_docstring = False

            # Find second docstring (after first is done)
            elif (
                first_docstring_ended
                and stripped == '"""'
                and second_docstring_start is None
            ):
                second_docstring_start = i
            elif second_docstring_start is not None and second_docstring_end is None:
                if '"""' in stripped or stripped.startswith('"""'):
                    second_docstring_end = i
                    break

        if second_docstring_start is not None and second_docstring_end is not None:
            # Remove the second docstring lines
            lines_to_remove = list(
                range(second_docstring_start, second_docstring_end + 1)
            )
            new_lines = [
                line for i, line in enumerate(lines) if i not in lines_to_remove
            ]

            # Also remove the blank line before the docstring if present
            if (
                second_docstring_start > 0
                and not lines[second_docstring_start - 1].strip()
            ):
                new_lines.pop(second_docstring_start - 1)

            # Write back
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))

            print(
                f"  ✓ Removed secondary docstring at lines {second_docstring_start + 1}-{second_docstring_end + 1}"
            )
        else:
            print("  ! Could not find secondary docstring pattern")


if __name__ == "__main__":
    print("Fixing conversion errors...")
    print("\n1. Fixing indentation errors...")
    fix_indentation_errors()

    print("\n2. Fixing multiple docstrings...")
    fix_multiple_docstrings()

    print("\n✅ All fixes complete!")

#!/usr/bin/env python3
"""AST-based code analysis for documentation generation.

Extracts module structures, classes, functions, and docstrings from Python files.
"""

import ast
import json
from pathlib import Path
from typing import Any


def extract_docstring(node: ast.AST) -> str | None:
    """Extract docstring from an AST node."""
    return ast.get_docstring(node)


def analyze_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Analyze a function definition."""
    # Extract parameters
    params = []
    for arg in node.args.args:
        param_info = {"name": arg.arg}
        if arg.annotation:
            param_info["type"] = ast.unparse(arg.annotation)
        params.append(param_info)

    # Extract return type
    returns = None
    if node.returns:
        returns = ast.unparse(node.returns)

    # Extract decorators
    decorators = [ast.unparse(dec) for dec in node.decorator_list]

    return {
        "name": node.name,
        "docstring": extract_docstring(node),
        "params": params,
        "returns": returns,
        "decorators": decorators,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "lineno": node.lineno,
    }


def analyze_class(node: ast.ClassDef) -> dict[str, Any]:
    """Analyze a class definition."""
    methods = []
    attributes = []

    for item in node.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            methods.append(analyze_function(item))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            # Class attribute with type annotation
            attr_info = {"name": item.target.id}
            if item.annotation:
                attr_info["type"] = ast.unparse(item.annotation)
            attributes.append(attr_info)

    # Extract base classes
    bases = [ast.unparse(base) for base in node.bases]

    # Extract decorators
    decorators = [ast.unparse(dec) for dec in node.decorator_list]

    return {
        "name": node.name,
        "docstring": extract_docstring(node),
        "bases": bases,
        "decorators": decorators,
        "methods": methods,
        "attributes": attributes,
        "lineno": node.lineno,
    }


def analyze_module(file_path: Path, base_path: Path) -> dict[str, Any]:
    """Analyze a Python module file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError) as e:
        return {"error": str(e), "file": str(file_path)}

    # Extract module docstring
    module_docstring = extract_docstring(tree)

    # Extract imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                [
                    {"type": "import", "name": alias.name, "asname": alias.asname}
                    for alias in node.names
                ]
            )
        elif isinstance(node, ast.ImportFrom):
            imports.extend(
                [
                    {
                        "type": "from",
                        "module": node.module,
                        "name": alias.name,
                        "asname": alias.asname,
                    }
                    for alias in node.names
                ]
            )

    # Extract top-level functions and classes
    functions = []
    classes = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            functions.append(analyze_function(node))
        elif isinstance(node, ast.ClassDef):
            classes.append(analyze_class(node))

    return {
        "file": str(file_path.relative_to(base_path)),
        "docstring": module_docstring,
        "imports": imports[:10],  # Limit imports to avoid clutter
        "functions": functions,
        "classes": classes,
    }


def main():
    """Analyze all Python files in the nlsq/ directory."""
    base_path = Path.cwd()
    nlsq_dir = base_path / "nlsq"

    if not nlsq_dir.exists():
        print(f"Error: {nlsq_dir} directory not found")
        return

    results = []

    # Analyze all Python files
    for py_file in sorted(nlsq_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        print(f"Analyzing {py_file}...")
        module_info = analyze_module(py_file, base_path)
        results.append(module_info)

    # Save results
    output_file = Path("docs/ast_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis complete! Results saved to {output_file}")

    # Print summary
    total_classes = sum(len(m.get("classes", [])) for m in results if "error" not in m)
    total_functions = sum(
        len(m.get("functions", [])) for m in results if "error" not in m
    )
    total_methods = sum(
        sum(len(c.get("methods", [])) for c in m.get("classes", []))
        for m in results
        if "error" not in m
    )

    print("\nSummary:")
    print(f"  Modules analyzed: {len(results)}")
    print(f"  Classes found: {total_classes}")
    print(f"  Functions found: {total_functions}")
    print(f"  Methods found: {total_methods}")


if __name__ == "__main__":
    main()

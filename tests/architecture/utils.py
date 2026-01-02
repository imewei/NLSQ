"""Architecture testing utilities for NLSQ.

This module provides utilities for detecting circular dependencies and other
architectural issues in the NLSQ package.
"""

import importlib
import sys
from collections import defaultdict
from pathlib import Path


def detect_circular_deps(package_name: str = "nlsq") -> list[tuple[str, str]]:
    """Detect circular import dependencies in a package.

    This function analyzes the import structure of a Python package to find
    circular dependencies. It uses static analysis of import statements
    rather than dynamic import tracing.

    Parameters
    ----------
    package_name : str, default="nlsq"
        The name of the package to analyze.

    Returns
    -------
    list[tuple[str, str]]
        List of circular dependency pairs (module_a, module_b) where
        module_a imports module_b and module_b imports module_a.

    Examples
    --------
    >>> cycles = detect_circular_deps("nlsq")
    >>> if cycles:
    ...     print(f"Found {len(cycles)} circular dependencies:")
    ...     for a, b in cycles:
    ...         print(f"  {a} <-> {b}")

    Notes
    -----
    This is a simplified detector that finds direct circular dependencies.
    It does not detect longer cycles (A -> B -> C -> A) or conditional
    imports that may not cause runtime issues.
    """
    import ast

    # Find the package root
    try:
        pkg = importlib.import_module(package_name)
        if not hasattr(pkg, "__path__"):
            return []  # Not a package
        pkg_path = Path(pkg.__path__[0])
    except ImportError:
        return []

    # Build import graph
    imports: dict[str, set[str]] = defaultdict(set)

    for py_file in pkg_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        # Get module name relative to package
        rel_path = py_file.relative_to(pkg_path.parent)
        if rel_path.name == "__init__.py":
            module_name = str(rel_path.parent).replace("/", ".")
        else:
            module_name = str(rel_path.with_suffix("")).replace("/", ".")

        # Parse the file and extract only module-level imports
        # (not inside functions or TYPE_CHECKING blocks)
        try:
            with open(py_file, encoding="utf-8") as f:
                source = f.read()
                tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        # Only look at top-level statements (not inside functions/classes)
        for node in tree.body:
            # Skip TYPE_CHECKING blocks
            if isinstance(node, ast.If):
                # Check if this is "if TYPE_CHECKING:"
                if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                    continue

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(package_name):
                        imports[module_name].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(package_name):
                    imports[module_name].add(node.module)

    # Find direct circular dependencies
    cycles: list[tuple[str, str]] = []
    checked: set[tuple[str, str]] = set()

    for module_a, imported_by_a in imports.items():
        for module_b in imported_by_a:
            # Skip self-references (module importing itself or its __init__)
            if module_a == module_b:
                continue
            # Skip if one is parent of another (e.g., nlsq.core imports nlsq.core.minpack)
            if module_a.startswith(module_b + ".") or module_b.startswith(
                module_a + "."
            ):
                continue

            if module_b in imports and module_a in imports[module_b]:
                # Found a cycle
                pair = tuple(sorted([module_a, module_b]))
                if pair not in checked:
                    checked.add(pair)
                    cycles.append((module_a, module_b))

    return cycles


def get_module_dependencies(module_name: str) -> set[str]:
    """Get direct dependencies of a module within the nlsq package.

    Parameters
    ----------
    module_name : str
        Full module name (e.g., "nlsq.core.minpack").

    Returns
    -------
    set[str]
        Set of module names that this module imports from nlsq.

    Examples
    --------
    >>> deps = get_module_dependencies("nlsq.core.minpack")
    >>> print(f"minpack.py has {len(deps)} internal dependencies")
    """
    import ast

    try:
        mod = importlib.import_module(module_name)
        if not hasattr(mod, "__file__") or mod.__file__ is None:
            return set()
        file_path = Path(mod.__file__)
    except ImportError:
        return set()

    dependencies: set[str] = set()

    try:
        with open(file_path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except SyntaxError:
        return set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nlsq"):
                    dependencies.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("nlsq"):
                dependencies.add(node.module)

    return dependencies


def count_module_dependencies(module_name: str) -> int:
    """Count the number of direct nlsq dependencies for a module.

    This is useful for measuring "god module" status - modules with
    too many dependencies are harder to maintain.

    Parameters
    ----------
    module_name : str
        Full module name (e.g., "nlsq.core.minpack").

    Returns
    -------
    int
        Number of unique nlsq modules imported by this module.

    Examples
    --------
    >>> count = count_module_dependencies("nlsq.core.minpack")
    >>> assert count < 15, f"God module detected: {count} dependencies"
    """
    return len(get_module_dependencies(module_name))

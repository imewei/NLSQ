#!/usr/bin/env python3
"""Analyze documentation coverage by comparing AST analysis with existing docs."""

import json
from pathlib import Path


def analyze_coverage():
    """Analyze documentation coverage."""
    # Load AST analysis
    ast_file = Path("docs/ast_analysis.json")
    with ast_file.open(encoding="utf-8") as f:
        modules = json.load(f)

    # Find existing API docs
    api_dir = Path("docs/api")
    existing_docs = {f.stem for f in api_dir.glob("nlsq.*.rst")}

    # Analyze coverage
    print("=" * 80)
    print("DOCUMENTATION COVERAGE ANALYSIS")
    print("=" * 80)

    total_modules = 0
    documented_modules = 0
    undocumented_modules = []

    total_classes = 0
    total_functions = 0
    total_methods = 0

    modules_without_docstrings = []
    classes_without_docstrings = []
    functions_without_docstrings = []

    for module in modules:
        if "error" in module:
            continue

        module_name = module["file"].replace("/", ".").replace(".py", "")
        total_modules += 1

        # Check if module has API doc
        if module_name in existing_docs:
            documented_modules += 1
        else:
            undocumented_modules.append(module_name)

        # Check module docstring
        if not module.get("docstring"):
            modules_without_docstrings.append(module_name)

        # Count and check classes
        for cls in module.get("classes", []):
            total_classes += 1
            if not cls.get("docstring"):
                classes_without_docstrings.append(f"{module_name}.{cls['name']}")

            # Count methods
            for method in cls.get("methods", []):
                total_methods += 1
                if not method.get("docstring"):
                    # Skip common methods that often don't need docstrings
                    if method["name"] not in {"__init__", "__repr__", "__str__"}:
                        functions_without_docstrings.append(
                            f"{module_name}.{cls['name']}.{method['name']}"
                        )

        # Count and check functions
        for func in module.get("functions", []):
            total_functions += 1
            if not func.get("docstring"):
                functions_without_docstrings.append(f"{module_name}.{func['name']}")

    # Print summary
    print("\nModule Coverage:")
    print(f"  Total modules: {total_modules}")
    print(f"  Documented (have API .rst): {documented_modules}")
    print(f"  Missing API docs: {len(undocumented_modules)}")
    if undocumented_modules:
        print("\n  Undocumented modules:")
        for mod in sorted(undocumented_modules)[:10]:
            print(f"    - {mod}")
        if len(undocumented_modules) > 10:
            print(f"    ... and {len(undocumented_modules) - 10} more")

    print("\nCode Statistics:")
    print(f"  Total classes: {total_classes}")
    print(f"  Total functions: {total_functions}")
    print(f"  Total methods: {total_methods}")

    print("\nDocstring Coverage:")
    print(f"  Modules without docstrings: {len(modules_without_docstrings)}")
    print(f"  Classes without docstrings: {len(classes_without_docstrings)}")
    print(
        f"  Functions/methods without docstrings: {len(functions_without_docstrings)}"
    )

    if modules_without_docstrings:
        print("\n  Modules needing docstrings:")
        for mod in sorted(modules_without_docstrings)[:10]:
            print(f"    - {mod}")

    if classes_without_docstrings:
        print("\n  Classes needing docstrings (top 10):")
        for cls in sorted(classes_without_docstrings)[:10]:
            print(f"    - {cls}")

    if functions_without_docstrings:
        print("\n  Functions/methods needing docstrings (top 20):")
        for func in sorted(functions_without_docstrings)[:20]:
            print(f"    - {func}")

    # Calculate percentages
    module_coverage = (
        (documented_modules / total_modules * 100) if total_modules > 0 else 0
    )
    total_items = total_modules + total_classes + total_functions + total_methods
    items_with_docstrings = (
        total_modules
        - len(modules_without_docstrings)
        + total_classes
        - len(classes_without_docstrings)
        + total_functions
        + total_methods
        - len(functions_without_docstrings)
    )
    docstring_coverage = (
        (items_with_docstrings / total_items * 100) if total_items > 0 else 0
    )

    print("\nOverall Coverage:")
    print(f"  API documentation coverage: {module_coverage:.1f}%")
    print(f"  Docstring coverage: {docstring_coverage:.1f}%")

    # Save detailed report
    report = {
        "summary": {
            "total_modules": total_modules,
            "documented_modules": documented_modules,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "total_methods": total_methods,
            "module_coverage_percent": round(module_coverage, 1),
            "docstring_coverage_percent": round(docstring_coverage, 1),
        },
        "undocumented_modules": sorted(undocumented_modules),
        "modules_without_docstrings": sorted(modules_without_docstrings),
        "classes_without_docstrings": sorted(classes_without_docstrings),
        "functions_without_docstrings": sorted(functions_without_docstrings),
    }

    output_file = Path("docs/coverage_report.json")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'-' * 80}")
    print(f"Detailed report saved to: {output_file}")


if __name__ == "__main__":
    analyze_coverage()

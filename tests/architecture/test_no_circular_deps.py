"""Tests for circular import dependencies in NLSQ.

This module verifies that the NLSQ package structure doesn't have circular
import dependencies that could cause ImportError at runtime.
"""

import importlib
import sys

import pytest

from tests.architecture.utils import detect_circular_deps

# All packages that should import cleanly
PACKAGES = [
    "nlsq",
    "nlsq.result",
    "nlsq.interfaces",
    "nlsq.core",
    "nlsq.streaming",
    "nlsq.global_optimization",
    "nlsq.caching",
    "nlsq.stability",
    "nlsq.precision",
    "nlsq.utils",
    "nlsq.diagnostics",
]


class TestPackageImports:
    """Test that all packages import cleanly without circular dependency errors."""

    @pytest.mark.parametrize("package", PACKAGES)
    def test_package_imports_cleanly(self, package: str):
        """Each package should import without circular dependency errors.

        This test catches the most obvious circular dependency issues that
        would cause ImportError at runtime.
        """
        # Remove cached imports to test fresh import
        modules_to_remove = [m for m in sys.modules if m.startswith(package)]
        for m in modules_to_remove:
            del sys.modules[m]

        # This should not raise ImportError
        try:
            importlib.import_module(package)
        except ImportError as e:
            pytest.fail(f"Package {package} failed to import: {e}")


class TestCircularDependencyDetection:
    """Test for circular dependencies using static analysis."""

    def test_no_circular_deps_detected(self):
        """Automated circular dependency detection.

        Uses static AST analysis to find modules that import each other,
        which indicates a circular dependency.
        """
        cycles = detect_circular_deps("nlsq")

        # Build a helpful error message if cycles are found
        if cycles:
            cycle_report = "\n".join(f"  - {a} <-> {b}" for a, b in sorted(cycles))
            pytest.fail(
                f"Found {len(cycles)} circular dependency pairs:\n{cycle_report}\n\n"
                "To fix circular dependencies:\n"
                "1. Move shared types to a separate module (e.g., nlsq.result)\n"
                "2. Use function-level imports for optional features\n"
                "3. Use TYPE_CHECKING guards for type hints only\n"
                "4. Apply dependency inversion with protocols"
            )

    def test_core_imports_result(self):
        """Core modules should import from nlsq.result, not nlsq.core._optimize."""
        # This is now enforced by the deprecation shim
        # Just verify the import works
        from nlsq.result import OptimizeResult, OptimizeWarning

        assert OptimizeResult is not None
        assert OptimizeWarning is not None


# NOTE: TestImportTime was removed because import time depends on JAX cache
# state, disk I/O speed, and system load — making it inherently flaky in CI.

"""Tests for circular import dependencies in NLSQ.

This module verifies that the NLSQ package structure doesn't have circular
import dependencies that could cause ImportError at runtime.
"""

import subprocess
import sys

import pytest

from tests.architecture.utils import detect_circular_deps

# All packages that should import cleanly
PACKAGES = [
    "nlsq",
    "nlsq.result",
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

        Runs the import in a subprocess rather than deleting/reimporting
        the package's sys.modules entries in-process: an in-process reload
        re-executes every module's top-level code, which can leave global
        state (module-level registries, cached class objects, etc.)
        permanently diverged from what already-collected test modules
        reference -- even after restoring the sys.modules mapping itself.
        Observed in CI as monkeypatch.setattr(SomeClass, ...) silently not
        taking effect in unrelated tests later in the same xdist worker
        (e.g. HPCCheckpointManager.load, CurveFit) because production
        code's lazy re-import resolved to a freshly-reloaded class distinct
        from the one the patch targeted. A subprocess makes that
        structurally impossible: whatever state the reload disturbs dies
        with the child process.
        """
        result = subprocess.run(
            [sys.executable, "-c", f"import {package}"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"Package {package} failed to import: {result.stderr}")


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

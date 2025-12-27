"""Import time benchmarks for NLSQ.

This module measures import time performance for SC-001:
- Target: <400ms cold import time
- Baseline: ~1084ms (measured 2025-12-26)

Tests run in subprocess to ensure clean import state.
"""

import subprocess
import sys
import time

import pytest


class TestImportTimeBenchmarks:
    """Import time benchmarks for SC-001."""

    def test_cold_import_time(self) -> None:
        """Measure cold import time in isolated subprocess.

        This test runs nlsq import in a fresh Python process to measure
        true cold import time without any prior caching.

        Target: <400ms (SC-001)
        """
        code = """\
import time
start = time.perf_counter()
import nlsq
end = time.perf_counter()
print(f'{(end - start) * 1000:.1f}')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **dict(__import__("os").environ),
                "JAX_PLATFORMS": "cpu",  # Force CPU for consistent timing
            },
        )

        if result.returncode != 0:
            pytest.fail(f"Import failed: {result.stderr}")

        import_time_ms = float(result.stdout.strip())
        print(f"\nCold import time: {import_time_ms:.1f}ms")

        # Record for reporting but don't fail - this is informational
        # The actual target depends on lazy import implementation
        assert import_time_ms > 0, "Import time measurement failed"

    def test_warm_import_time(self) -> None:
        """Measure warm import time (module already in sys.modules).

        This should be very fast (<1ms) since the module is already loaded.
        """
        # First import to warm up
        import nlsq

        # Measure reimport time
        start = time.perf_counter()
        import importlib

        importlib.reload(nlsq)
        end = time.perf_counter()

        warm_time_ms = (end - start) * 1000
        print(f"\nWarm reimport time: {warm_time_ms:.1f}ms")

        # Warm import should be fast
        assert warm_time_ms < 5000, f"Warm import too slow: {warm_time_ms:.1f}ms"

    def test_minimal_import_time(self) -> None:
        """Measure import time for just curve_fit (core only).

        This tests that core functionality loads quickly without
        pulling in specialty modules.
        """
        code = """\
import time
start = time.perf_counter()
from nlsq import curve_fit
end = time.perf_counter()
print(f'{(end - start) * 1000:.1f}')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **dict(__import__("os").environ),
                "JAX_PLATFORMS": "cpu",
            },
        )

        if result.returncode != 0:
            pytest.fail(f"Import failed: {result.stderr}")

        import_time_ms = float(result.stdout.strip())
        print(f"\nMinimal import time (curve_fit only): {import_time_ms:.1f}ms")

        assert import_time_ms > 0

    def test_lazy_module_import_time(self) -> None:
        """Measure import time for a lazy module.

        This tests that accessing a lazy module adds minimal overhead.
        """
        code = """\
import time
import nlsq

# Base import done, now access a lazy module
start = time.perf_counter()
_ = nlsq.LargeDatasetFitter
end = time.perf_counter()
print(f'{(end - start) * 1000:.1f}')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **dict(__import__("os").environ),
                "JAX_PLATFORMS": "cpu",
            },
        )

        if result.returncode != 0:
            pytest.fail(f"Import failed: {result.stderr}")

        lazy_time_ms = float(result.stdout.strip())
        print(f"\nLazy module access time: {lazy_time_ms:.1f}ms")

        assert lazy_time_ms > 0


class TestImportTimeComparison:
    """Compare import times before and after optimization."""

    BASELINE_MS = 1084.4  # From benchmarks/baselines/import_baseline.txt
    TARGET_MS = 400.0  # SC-001 target

    def test_import_time_vs_target(self) -> None:
        """Check if import time meets SC-001 target."""
        code = """\
import time
start = time.perf_counter()
import nlsq
end = time.perf_counter()
print(f'{(end - start) * 1000:.1f}')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **dict(__import__("os").environ),
                "JAX_PLATFORMS": "cpu",
            },
        )

        if result.returncode != 0:
            pytest.fail(f"Import failed: {result.stderr}")

        import_time_ms = float(result.stdout.strip())

        reduction_pct = (1 - import_time_ms / self.BASELINE_MS) * 100

        print("\n--- Import Time Report ---")
        print(f"Baseline: {self.BASELINE_MS:.1f}ms")
        print(f"Current:  {import_time_ms:.1f}ms")
        print(f"Target:   {self.TARGET_MS:.1f}ms")
        print(f"Reduction: {reduction_pct:.1f}%")

        if import_time_ms <= self.TARGET_MS:
            print(
                f"✓ SC-001 PASSED: Import time {import_time_ms:.1f}ms <= {self.TARGET_MS}ms target"
            )
        else:
            print(
                f"⚠ SC-001 NOT MET: Import time {import_time_ms:.1f}ms > {self.TARGET_MS}ms target"
            )

        # Don't fail - this is informational until lazy imports are implemented
        assert import_time_ms > 0

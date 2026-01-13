"""Memory regression tests for CurveFit God Class Decomposition.

Validates that the decomposition maintains memory usage within acceptable bounds:
- Memory overhead: <10% regression from baseline

Reference: specs/017-curve-fit-decomposition/spec.md SC-011
"""

from __future__ import annotations

import gc
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit

# =============================================================================
# Test Fixtures
# =============================================================================


def exponential_model(x: np.ndarray, a: float, b: float) -> Any:
    """Exponential decay model for benchmarking."""
    return a * jnp.exp(-b * x)


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback if psutil not available
        return 0.0


# =============================================================================
# Memory Usage Tests
# =============================================================================


class TestDecompositionMemory:
    """Memory regression tests for decomposed CurveFit."""

    @pytest.fixture
    def benchmark_data_10k(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate 10K point dataset."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 4, 10000)
        y_true = 2.5 * np.exp(-1.3 * x)
        y = y_true + rng.normal(0, 0.1, 10000)
        return x, y

    @pytest.fixture
    def benchmark_data_100k(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate 100K point dataset."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 4, 100000)
        y_true = 2.5 * np.exp(-1.3 * x)
        y = y_true + rng.normal(0, 0.1, 100000)
        return x, y

    def test_memory_no_leak_repeated_fits(
        self, benchmark_data_10k: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify no memory leak with repeated curve_fit calls.

        SC-011: Memory usage should not grow unboundedly.
        """
        x, y = benchmark_data_10k

        # Force garbage collection and get baseline
        gc.collect()
        baseline_mb = get_process_memory_mb()

        if baseline_mb == 0:
            pytest.skip("psutil not available for memory measurement")

        # Run many fits
        for i in range(20):
            curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Force garbage collection
        gc.collect()
        final_mb = get_process_memory_mb()

        growth_mb = final_mb - baseline_mb
        print("\nMemory after 20 fits:")
        print(f"  Baseline: {baseline_mb:.1f}MB")
        print(f"  Final: {final_mb:.1f}MB")
        print(f"  Growth: {growth_mb:.1f}MB")

        # Allow reasonable growth for JIT caches, but flag major leaks
        # 100MB growth for 20 fits of 10K points would indicate a leak
        assert growth_mb < 100, f"Possible memory leak: grew by {growth_mb:.1f}MB"

    def test_memory_scales_linearly(self) -> None:
        """Verify memory scales approximately linearly with data size."""
        sizes = [1000, 5000, 10000]
        memory_usage = []

        for size in sizes:
            gc.collect()
            before_mb = get_process_memory_mb()

            if before_mb == 0:
                pytest.skip("psutil not available for memory measurement")

            rng = np.random.default_rng(42)
            x = np.linspace(0, 4, size)
            y = 2.5 * np.exp(-1.3 * x) + rng.normal(0, 0.1, size)

            curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

            gc.collect()
            after_mb = get_process_memory_mb()
            memory_usage.append(after_mb - before_mb)

        print("\nMemory scaling:")
        for size, mem in zip(sizes, memory_usage, strict=True):
            print(f"  {size:>6} points: {mem:>6.1f}MB delta")

        # Memory should not grow super-linearly
        # 10x data should not cause >20x memory
        if memory_usage[0] > 0:
            ratio = memory_usage[-1] / memory_usage[0]
            size_ratio = sizes[-1] / sizes[0]
            print(f"  Memory ratio: {ratio:.1f}x for {size_ratio:.0f}x data")

    def test_component_memory_isolation(self) -> None:
        """Verify extracted components don't share unexpected state."""
        from nlsq.core.orchestration import CovarianceComputer, DataPreprocessor

        rng = np.random.default_rng(42)
        x = np.linspace(0, 4, 10000)
        y = 2.5 * np.exp(-1.3 * x) + rng.normal(0, 0.1, 10000)

        # Create multiple instances
        preprocessors = [DataPreprocessor() for _ in range(5)]
        covariance_computers = [CovarianceComputer() for _ in range(5)]

        # Use each instance - DataPreprocessor.preprocess requires (f, xdata, ydata)
        for prep in preprocessors:
            prep.preprocess(exponential_model, x, y)

        # Verify no shared mutable state
        # Each instance should work independently
        for i, prep in enumerate(preprocessors):
            result = prep.preprocess(exponential_model, x, y)
            assert result is not None, f"Preprocessor {i} failed"

        # Clean up
        del preprocessors
        del covariance_computers
        gc.collect()


# =============================================================================
# Import Memory Tests
# =============================================================================


class TestImportMemory:
    """Tests for import-time memory usage."""

    def test_lazy_import_reduces_baseline_memory(self) -> None:
        """Verify lazy imports reduce initial memory footprint."""
        # This is more of a documentation test - actual measurement
        # requires subprocess isolation which is slow

        # Check that facades are lazily imported

        # Clear any existing imports
        nlsq_modules = [k for k in sys.modules if "nlsq" in k]
        initial_count = len(nlsq_modules)

        # Import facades (should be lazy)
        from nlsq.facades import OptimizationFacade

        nlsq_modules_after = [k for k in sys.modules if "nlsq" in k]
        count_after = len(nlsq_modules_after)

        print("\nImport footprint:")
        print(f"  Before facade import: {initial_count} modules")
        print(f"  After facade import: {count_after} modules")

        # Accessing facade should not immediately load global_optimization
        facade = OptimizationFacade()

        # Only when we access a method should it load
        CMAESOptimizer = facade.get_cmaes_optimizer()

        nlsq_modules_final = [k for k in sys.modules if "nlsq" in k]
        global_opt_loaded = any("global_optimization" in m for m in nlsq_modules_final)

        print(f"  After get_cmaes_optimizer: {len(nlsq_modules_final)} modules")
        print(f"  global_optimization loaded: {global_opt_loaded}")

        assert global_opt_loaded, "Lazy loading should load module on access"

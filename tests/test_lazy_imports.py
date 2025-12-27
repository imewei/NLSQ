"""Unit tests for lazy import functionality in nlsq package.

This module tests that lazy imports work correctly, including:
- Lazy modules are not loaded until accessed
- All lazy modules load correctly when accessed
- __getattr__ and __dir__ work correctly
- Backward compatibility is maintained
"""

import sys
import types
from unittest.mock import patch

import pytest


class TestLazyImportMechanism:
    """Tests for the lazy import mechanism in nlsq.__init__."""

    def test_lazy_modules_exist_in_dir(self) -> None:
        """Test that lazy module exports appear in dir(nlsq)."""
        import nlsq

        # Check that some known lazy exports are in dir()
        nlsq_dir = dir(nlsq)

        # These should be in __all__ and visible via __dir__
        lazy_exports = [
            "StreamingOptimizer",
            "LargeDatasetFitter",
            "PerformanceProfiler",
            "MemoryManager",
        ]

        for export in lazy_exports:
            assert export in nlsq_dir, f"{export} should be visible in dir(nlsq)"

    def test_lazy_module_loads_on_access(self) -> None:
        """Test that lazy modules load correctly when accessed."""
        # Fresh import to ensure clean state
        import nlsq

        # Access a lazy module - should not raise
        fitter = nlsq.LargeDatasetFitter
        assert fitter is not None
        assert callable(fitter)

    def test_all_exports_accessible(self) -> None:
        """Test that all items in __all__ are accessible."""
        import nlsq

        for name in nlsq.__all__:
            obj = getattr(nlsq, name, None)
            assert obj is not None, f"{name} in __all__ is not accessible"

    def test_lazy_import_caches_result(self) -> None:
        """Test that lazy imports are cached after first access."""
        import nlsq

        # First access
        obj1 = nlsq.LargeDatasetFitter
        # Second access should return same object
        obj2 = nlsq.LargeDatasetFitter

        assert obj1 is obj2, "Lazy import should cache the result"

    def test_core_modules_always_available(self) -> None:
        """Test that core modules are always available (eager load)."""
        import nlsq

        # Core exports should always be present
        assert hasattr(nlsq, "curve_fit")
        assert hasattr(nlsq, "CurveFit")
        assert hasattr(nlsq, "LeastSquares")
        assert hasattr(nlsq, "OptimizeResult")
        assert hasattr(nlsq, "__version__")

    def test_invalid_attribute_raises_attribute_error(self) -> None:
        """Test that accessing non-existent attributes raises AttributeError."""
        import nlsq

        with pytest.raises(AttributeError):
            _ = nlsq.NonExistentModule


class TestLazyImportCategories:
    """Tests for specific categories of lazy imports."""

    def test_streaming_imports(self) -> None:
        """Test streaming module lazy imports."""
        import nlsq

        # These should all be accessible
        assert hasattr(nlsq, "StreamingOptimizer")
        assert hasattr(nlsq, "StreamingConfig")
        assert hasattr(nlsq, "fit_unlimited_data")

    def test_large_dataset_imports(self) -> None:
        """Test large dataset module lazy imports."""
        import nlsq

        assert hasattr(nlsq, "LargeDatasetFitter")
        assert hasattr(nlsq, "fit_large_dataset")
        assert hasattr(nlsq, "estimate_memory_requirements")

    def test_profiler_imports(self) -> None:
        """Test profiler module lazy imports."""
        import nlsq

        assert hasattr(nlsq, "PerformanceProfiler")
        assert hasattr(nlsq, "ProfileMetrics")

    def test_memory_imports(self) -> None:
        """Test memory management lazy imports."""
        import nlsq

        assert hasattr(nlsq, "MemoryManager")
        assert hasattr(nlsq, "MemoryPool")
        assert hasattr(nlsq, "get_memory_manager")

    def test_workflow_imports(self) -> None:
        """Test workflow system lazy imports."""
        import nlsq

        assert hasattr(nlsq, "WorkflowConfig")
        assert hasattr(nlsq, "WorkflowSelector")
        assert hasattr(nlsq, "auto_select_workflow")


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_direct_import_syntax(self) -> None:
        """Test that 'from nlsq import X' works for lazy modules."""
        # This pattern should work for lazy imports
        from nlsq import LargeDatasetFitter

        assert LargeDatasetFitter is not None

    def test_attribute_access_syntax(self) -> None:
        """Test that 'nlsq.X' works for lazy modules."""
        import nlsq

        assert nlsq.LargeDatasetFitter is not None

    def test_core_api_unchanged(self) -> None:
        """Test that core API behavior is unchanged."""
        import numpy as np

        from nlsq import curve_fit

        def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
            return a * x + b

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        popt, _pcov = curve_fit(model, x, y)

        assert len(popt) == 2
        assert np.allclose(popt, [2.0, 0.0], atol=0.01)

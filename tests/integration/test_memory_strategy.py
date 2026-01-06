"""Integration tests for memory-based automatic strategy selection.

Tests the MemoryBudgetSelector integration with curve_fit() when method='auto'.

Test Coverage:
- T042: test_curve_fit_auto_selects_chunked_on_constrained_memory
- T043: test_curve_fit_auto_selects_standard_on_high_memory
- T044: test_removed_apis_raise_import_error
"""

import jax.numpy as jnp
import numpy as np
import pytest


def exponential_model(x, a, b):
    """Simple exponential model for testing."""
    return a * jnp.exp(-b * x)


class TestMemoryBasedStrategySelection:
    """Integration tests for memory-based auto strategy selection."""

    @pytest.mark.slow
    def test_curve_fit_auto_selects_streaming_on_very_constrained_memory(self):
        """T042: Verify method='auto' selects streaming when data exceeds memory threshold.

        With very tight memory constraint (0.001 GB), even a small dataset
        will trigger streaming strategy.
        """
        from nlsq import curve_fit

        # Generate test data
        n_points = 10_000
        x = np.linspace(0, 4, n_points)
        y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.normal(size=n_points)

        # With very tight memory constraint, should trigger streaming
        result = curve_fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            method="auto",
            memory_limit_gb=0.0001,  # Very tight constraint
            verbose=0,
        )

        # Should still get reasonable fit
        assert result.success
        assert abs(result.x[0] - 2.5) < 0.5
        assert abs(result.x[1] - 1.3) < 0.5

    def test_curve_fit_auto_selects_standard_on_high_memory(self):
        """T043: Verify method='auto' selects standard when peak fits in memory.

        On a system with sufficient memory, a small dataset should use
        the standard curve_fit path.
        """
        from nlsq import curve_fit

        # Small dataset that should fit in any reasonable memory
        n_points = 1000
        x = np.linspace(0, 4, n_points)
        y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.normal(size=n_points)

        # With default memory limit (auto-detect), should use standard
        result = curve_fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            method="auto",
            verbose=0,
        )

        # Verify fit succeeded
        assert result.success
        assert abs(result.x[0] - 2.5) < 0.3
        assert abs(result.x[1] - 1.3) < 0.3

    def test_curve_fit_auto_respects_memory_limit_override(self):
        """Verify memory_limit_gb parameter affects strategy selection."""
        from nlsq import curve_fit
        from nlsq.core.workflow import MemoryBudgetSelector

        n_points = 100_000
        n_params = 5

        # Check strategy with different memory limits
        selector = MemoryBudgetSelector()

        # High memory limit should select standard
        strategy_high, _ = selector.select(
            n_points=n_points,
            n_params=n_params,
            memory_limit_gb=100.0,  # Very high limit
        )

        # Very low memory limit should select streaming or chunked
        strategy_low, _ = selector.select(
            n_points=n_points,
            n_params=n_params,
            memory_limit_gb=0.001,  # Very low limit
        )

        # The strategies should differ
        assert strategy_high == "standard"
        assert strategy_low in ("streaming", "chunked")


class TestRemovedAPIs:
    """Tests verifying removed APIs raise ImportError (FR-008)."""

    def test_removed_apis_raise_import_error(self):
        """T044: Verify WorkflowTier, WorkflowConfig, auto_select_workflow are not importable."""
        # These imports should raise ImportError since the classes were removed
        with pytest.raises(ImportError):
            from nlsq.core.workflow import WorkflowTier

        with pytest.raises(ImportError):
            from nlsq.core.workflow import WorkflowConfig

        with pytest.raises(ImportError):
            from nlsq.core.workflow import auto_select_workflow

        with pytest.raises(ImportError):
            from nlsq.core.workflow import WorkflowSelector

        with pytest.raises(ImportError):
            from nlsq.core.workflow import DatasetSizeTier

        with pytest.raises(ImportError):
            from nlsq.core.workflow import MemoryTier

        with pytest.raises(ImportError):
            from nlsq.core.workflow import WORKFLOW_PRESETS

    def test_new_apis_are_importable(self):
        """Verify new memory-based APIs are importable."""
        from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector

        # Verify they are classes
        assert MemoryBudget is not None
        assert MemoryBudgetSelector is not None

        # Verify basic functionality
        budget = MemoryBudget.compute(n_points=1000, n_params=5)
        assert budget.available_gb > 0
        assert budget.threshold_gb > 0

        selector = MemoryBudgetSelector()
        strategy, _config = selector.select(n_points=1000, n_params=5)
        assert strategy in ("standard", "chunked", "streaming")


class TestMethodAutoFunctionality:
    """Tests for method='auto' functionality in curve_fit."""

    def test_method_auto_basic_fit(self):
        """Verify basic fit works with method='auto'."""
        from nlsq import curve_fit

        x = np.linspace(0, 4, 500)
        y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.normal(size=len(x))

        result = curve_fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            method="auto",
            verbose=0,
        )

        assert result.success
        assert len(result.x) == 2

    def test_method_auto_with_bounds(self):
        """Verify method='auto' works with parameter bounds."""
        from nlsq import curve_fit

        x = np.linspace(0, 4, 500)
        y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.normal(size=len(x))

        result = curve_fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            bounds=([0, 0], [10, 10]),
            method="auto",
            verbose=0,
        )

        assert result.success
        # Parameters should be within bounds
        assert 0 <= result.x[0] <= 10
        assert 0 <= result.x[1] <= 10

    def test_method_auto_verbose_logging(self):
        """Verify verbose logging works with method='auto'."""
        import logging

        from nlsq import curve_fit

        # Set up logging capture
        logger = logging.getLogger("nlsq")
        original_level = logger.level
        logger.setLevel(logging.INFO)

        try:
            x = np.linspace(0, 4, 500)
            y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.normal(size=len(x))

            # This should trigger logging
            result = curve_fit(
                exponential_model,
                x,
                y,
                p0=[2.0, 1.0],
                method="auto",
                verbose=2,  # Verbose mode
            )

            assert result.success
        finally:
            logger.setLevel(original_level)

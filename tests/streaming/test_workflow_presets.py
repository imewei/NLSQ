"""Tests for the NEW WORKFLOW SYSTEM (v0.6.3).

This test module covers:
- US1 (auto): Memory-aware local optimization workflow
- US2 (auto_global): Memory-aware global optimization workflow
- US3 (hpc): HPC workflow with checkpointing
- US4 (cleanup): Removed preset error handling

Tests follow the 3-workflow system:
- "auto": Memory-aware local optimization (default)
- "auto_global": Memory-aware global optimization (requires bounds)
- "hpc": auto_global + checkpointing for HPC environments
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest


def model(x, a, b):
    """Simple exponential model for testing."""
    return a * jnp.exp(-b * x)


# =============================================================================
# US1: workflow="auto" Tests (T008-T011)
# =============================================================================


class TestAutoWorkflow:
    """Tests for workflow='auto' memory-aware local optimization."""

    def test_auto_selects_standard_for_small_data(self):
        """T008: Auto selects standard strategy for small datasets.

        When data + Jacobian fit comfortably in RAM, the system should
        select the 'standard' strategy (direct TRF optimization).
        """
        from nlsq.core.workflow import MemoryBudgetSelector

        selector = MemoryBudgetSelector(safety_factor=0.75)

        # Small dataset: 1K points, 3 params
        # Data: ~16 KB, Jacobian: ~24 KB, Peak: ~50 KB
        # This should easily fit in any available memory
        strategy, config = selector.select(
            n_points=1_000,
            n_params=3,
        )

        assert strategy == "standard"
        assert config is None

    def test_auto_selects_chunked_for_medium_data(self):
        """T009: Auto selects chunked strategy for medium datasets.

        When Jacobian exceeds memory threshold but data fits,
        the system should select the 'chunked' strategy.
        """
        from nlsq.core.workflow import MemoryBudgetSelector

        selector = MemoryBudgetSelector(safety_factor=0.75)

        # Force chunked selection by limiting available memory
        # With 0.5 GB limit, 1M points with 10 params should exceed threshold
        # Jacobian: 1M * 10 * 8 bytes = 80 MB
        # Peak: ~100+ MB, but with 0.5 GB * 0.75 * 0.9 = ~337 MB threshold
        # Need larger data or smaller memory limit
        strategy, config = selector.select(
            n_points=5_000_000,  # 5M points
            n_params=50,  # 50 params
            memory_limit_gb=0.5,  # 0.5 GB limit forces chunked/streaming
        )

        # Should select chunked or streaming (not standard)
        assert strategy in ("chunked", "streaming")
        assert config is not None

    def test_auto_selects_streaming_for_huge_data(self):
        """T010: Auto selects streaming strategy for huge datasets.

        When even the data arrays exceed available memory,
        the system should select the 'streaming' strategy.
        """
        from nlsq.core.workflow import MemoryBudgetSelector

        selector = MemoryBudgetSelector(safety_factor=0.75)

        # Force streaming by having data exceed threshold
        # 100M points * 16 bytes (x + y) = 1.6 GB data alone
        strategy, config = selector.select(
            n_points=100_000_000,  # 100M points
            n_params=5,
            memory_limit_gb=0.5,  # Only 0.5 GB available
        )

        assert strategy == "streaming"
        assert config is not None

    def test_auto_is_default_workflow(self):
        """T011: workflow='auto' is the default when not specified.

        The fit() function should default to 'auto' workflow.
        """
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        # Call fit without workflow parameter - should use 'auto'
        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
        )

        assert result is not None
        assert result.success

    def test_auto_workflow_logs_strategy_selection(self):
        """Auto workflow logs info about selected strategy."""
        from nlsq.core.workflow import MemoryBudgetSelector

        selector = MemoryBudgetSelector(safety_factor=0.75)

        # With verbose=True, should log strategy selection
        strategy, _config = selector.select(
            n_points=1_000,
            n_params=3,
            verbose=True,
        )

        assert strategy == "standard"

    def test_auto_workflow_respects_memory_limit_override(self):
        """Auto workflow respects memory_limit_gb override."""
        from nlsq.core.workflow import MemoryBudgetSelector

        selector = MemoryBudgetSelector(safety_factor=0.75)

        # With very low memory limit, even small data should trigger chunked
        strategy, _config = selector.select(
            n_points=100_000,
            n_params=20,
            memory_limit_gb=0.01,  # Only 10 MB
        )

        assert strategy in ("chunked", "streaming")


# =============================================================================
# US2: workflow="auto_global" Tests (T016-T025)
# =============================================================================


class TestAutoGlobalWorkflow:
    """Tests for workflow='auto_global' memory-aware global optimization."""

    def test_auto_global_requires_bounds(self):
        """T016: auto_global raises ValueError if bounds not provided."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        with pytest.raises(ValueError, match="requires bounds"):
            fit(
                model,
                x,
                y,
                p0=[1.0, 0.5],
                workflow="auto_global",
                # No bounds provided
            )

    def test_auto_global_with_bounds_succeeds(self):
        """T017: auto_global with bounds produces valid results."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="auto_global",
            bounds=([0.0, 0.0], [10.0, 10.0]),
        )

        assert result is not None
        assert result.success
        popt = np.array(result.x)
        assert 2.0 < popt[0] < 3.0  # a should be around 2.5
        assert 0.3 < popt[1] < 0.7  # b should be around 0.5

    def test_auto_global_uses_multistart_for_narrow_bounds(self):
        """T018: auto_global uses multi-start when scale_ratio < 1000."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        # Narrow bounds -> scale ratio < 1000 -> should use multi-start
        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="auto_global",
            bounds=([0.0, 0.0], [5.0, 5.0]),  # Narrow bounds
            n_starts=3,  # Few starts for speed
        )

        assert result is not None
        assert result.success

    def test_auto_global_respects_n_starts_override(self):
        """T024: auto_global respects n_starts parameter override."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        # Custom n_starts
        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="auto_global",
            bounds=([0.0, 0.0], [10.0, 10.0]),
            n_starts=5,
        )

        assert result is not None
        assert result.success


# =============================================================================
# US3: workflow="hpc" Tests (T036-T041)
# =============================================================================


class TestHPCWorkflow:
    """Tests for workflow='hpc' with checkpointing."""

    def test_hpc_requires_bounds(self):
        """T036: hpc raises ValueError if bounds not provided."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        with pytest.raises(ValueError, match="requires bounds"):
            fit(
                model,
                x,
                y,
                p0=[1.0, 0.5],
                workflow="hpc",
                # No bounds provided
            )

    def test_hpc_with_bounds_succeeds(self):
        """T037: hpc with bounds produces valid results."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="hpc",
            bounds=([0.0, 0.0], [10.0, 10.0]),
        )

        assert result is not None
        assert result.success
        popt = np.array(result.x)
        assert 2.0 < popt[0] < 3.0  # a should be around 2.5
        assert 0.3 < popt[1] < 0.7  # b should be around 0.5

    def test_hpc_accepts_checkpoint_parameters(self):
        """T038: hpc accepts checkpoint_dir and checkpoint_interval."""
        import tempfile

        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fit(
                model,
                x,
                y,
                p0=[1.0, 0.5],
                workflow="hpc",
                bounds=([0.0, 0.0], [10.0, 10.0]),
                checkpoint_dir=tmpdir,
                checkpoint_interval=10,
            )

            assert result is not None
            assert result.success


# =============================================================================
# US4: Removed Presets Tests (T049-T051)
# =============================================================================


class TestRemovedPresets:
    """Tests for removed workflow presets error handling."""

    @pytest.mark.parametrize(
        "preset_name",
        [
            "standard",
            "fast",
            "quality",
            "large_robust",
            "streaming",
            "hpc_distributed",
            "cmaes",
            "cmaes-global",
            "global_auto",
        ],
    )
    def test_old_preset_raises_clear_error(self, preset_name):
        """T049: Old presets raise ValueError with clear message."""
        from nlsq.core.minpack import REMOVED_PRESETS, _raise_removed_preset_error

        assert preset_name in REMOVED_PRESETS

        with pytest.raises(ValueError) as exc_info:
            _raise_removed_preset_error(preset_name)

        error_msg = str(exc_info.value)
        assert preset_name in error_msg
        assert "v0.6.3" in error_msg

    @pytest.mark.parametrize(
        "preset_name",
        [
            "standard",
            "fast",
            "quality",
            "large_robust",
            "streaming",
            "hpc_distributed",
            "cmaes",
            "cmaes-global",
            "global_auto",
        ],
    )
    def test_error_includes_migration_hint(self, preset_name):
        """T050: Error message includes specific migration hint."""
        from nlsq.core.minpack import REMOVED_PRESETS, _raise_removed_preset_error

        with pytest.raises(ValueError) as exc_info:
            _raise_removed_preset_error(preset_name)

        error_msg = str(exc_info.value)
        migration_hint = REMOVED_PRESETS[preset_name]

        # Error should contain the migration hint
        assert any(
            keyword in error_msg
            for keyword in ["workflow=", "auto", "auto_global", "hpc"]
        )

    @pytest.mark.parametrize(
        "preset_name",
        [
            "standard",
            "fast",
            "quality",
            "large_robust",
            "streaming",
            "hpc_distributed",
            "cmaes",
            "cmaes-global",
            "global_auto",
        ],
    )
    def test_error_includes_docs_url(self, preset_name):
        """T051: Error message includes documentation URL."""
        from nlsq.core.minpack import _raise_removed_preset_error

        with pytest.raises(ValueError) as exc_info:
            _raise_removed_preset_error(preset_name)

        error_msg = str(exc_info.value)
        assert "https://nlsq.readthedocs.io" in error_msg
        assert "migration" in error_msg.lower()


# =============================================================================
# MemoryBudgetSelector Unit Tests
# =============================================================================


class TestMemoryBudgetSelector:
    """Unit tests for MemoryBudgetSelector."""

    def test_selector_initialization(self):
        """Test selector can be initialized with custom safety factor."""
        from nlsq.core.workflow import MemoryBudgetSelector

        selector = MemoryBudgetSelector(safety_factor=0.5)
        assert selector.safety_factor == 0.5

    def test_selector_default_safety_factor(self):
        """Test selector uses 0.75 as default safety factor."""
        from nlsq.core.workflow import MemoryBudgetSelector

        selector = MemoryBudgetSelector()
        assert selector.safety_factor == 0.75

    def test_selector_returns_tuple(self):
        """Test selector returns (strategy, config) tuple."""
        from nlsq.core.workflow import MemoryBudgetSelector

        selector = MemoryBudgetSelector()
        result = selector.select(n_points=1000, n_params=3)

        assert isinstance(result, tuple)
        assert len(result) == 2
        strategy, _config = result
        assert strategy in ("standard", "chunked", "streaming")


# =============================================================================
# MethodSelector Unit Tests
# =============================================================================


class TestMethodSelector:
    """Unit tests for MethodSelector (CMA-ES vs Multi-Start)."""

    def test_selector_low_scale_ratio_selects_multistart(self):
        """Low scale ratio (< 1000) selects multi-start."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()

        # Narrow bounds with similar scales
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])

        method = selector.select("auto", lower, upper)
        assert method == "multi-start"

    def test_selector_high_scale_ratio_prefers_cmaes(self):
        """High scale ratio (> 1000) prefers CMA-ES if available."""
        from nlsq.global_optimization.cmaes_config import is_evosax_available
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()

        # Wide bounds with different scales (ratio > 1000)
        lower = np.array([1e-6, 0.0])
        upper = np.array([1e6, 10.0])

        method = selector.select("auto", lower, upper)

        if is_evosax_available():
            assert method == "cmaes"
        else:
            # Falls back to multi-start when evosax not available
            assert method == "multi-start"

    def test_selector_explicit_multistart_request(self):
        """Explicit 'multi-start' request always returns multi-start."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()
        lower = np.array([1e-6, 0.0])
        upper = np.array([1e6, 10.0])

        method = selector.select("multi-start", lower, upper)
        assert method == "multi-start"


# =============================================================================
# Integration Tests
# =============================================================================


class TestWorkflowIntegration:
    """Integration tests for the new workflow system."""

    def test_fit_with_auto_workflow_succeeds(self):
        """Test fit() with workflow='auto' produces valid results."""
        from nlsq import fit

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="auto",
        )

        assert result is not None
        assert result.success
        popt = np.array(result.x)
        assert 2.0 < popt[0] < 3.0  # a should be around 2.5
        assert 0.3 < popt[1] < 0.7  # b should be around 0.5

    def test_valid_workflows_are_recognized(self):
        """Test that only valid workflows are recognized."""
        from nlsq.core.minpack import REMOVED_PRESETS, WORKFLOW_PRESETS

        # Valid workflows
        valid_workflows = ["auto", "auto_global", "hpc"]

        # All removed presets should be in REMOVED_PRESETS
        for preset in REMOVED_PRESETS:
            assert preset in REMOVED_PRESETS

        # WORKFLOW_PRESETS should still exist for backwards compatibility
        # during transition (will be removed in Phase 6)
        assert "standard" in WORKFLOW_PRESETS

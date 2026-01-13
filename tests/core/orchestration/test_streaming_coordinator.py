"""Unit tests for StreamingCoordinator component.

Tests for memory estimation, strategy selection, and streaming configuration.

Reference: specs/017-curve-fit-decomposition/spec.md FR-004, FR-022
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.orchestration.streaming_coordinator import StreamingCoordinator
from nlsq.interfaces.orchestration_protocol import StreamingDecision

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def coordinator() -> StreamingCoordinator:
    """Create a StreamingCoordinator instance."""
    return StreamingCoordinator()


@pytest.fixture
def small_data() -> tuple[np.ndarray, np.ndarray]:
    """Small test data (fits in memory without streaming)."""
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + 1.0
    return x, y


@pytest.fixture
def medium_data() -> tuple[np.ndarray, np.ndarray]:
    """Medium test data (may need chunking)."""
    x = np.linspace(0, 10, 50_000)
    y = 2.0 * x + 1.0
    return x, y


# =============================================================================
# Test StreamingDecision Result
# =============================================================================


class TestStreamingDecisionResult:
    """Tests for StreamingDecision return type."""

    def test_returns_streaming_decision(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test decide returns StreamingDecision instance."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
        )

        assert isinstance(result, StreamingDecision)

    def test_streaming_decision_has_required_fields(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test StreamingDecision has all required attributes."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
        )

        assert hasattr(result, "strategy")
        assert hasattr(result, "reason")
        assert hasattr(result, "estimated_memory_mb")
        assert hasattr(result, "available_memory_mb")
        assert hasattr(result, "memory_pressure")
        assert hasattr(result, "chunk_size")
        assert hasattr(result, "n_chunks")
        assert hasattr(result, "hybrid_config")


# =============================================================================
# Test Strategy Selection
# =============================================================================


class TestStrategySelection:
    """Tests for streaming strategy selection."""

    def test_small_data_uses_direct(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test small data uses direct (non-streaming) strategy."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
        )

        assert result.strategy == "direct"
        assert result.chunk_size is None
        assert result.hybrid_config is None

    def test_strategy_values_are_valid(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test strategy is one of valid options."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
        )

        valid_strategies = {"direct", "chunked", "hybrid", "auto_memory"}
        assert result.strategy in valid_strategies

    def test_force_streaming_overrides_decision(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test force_streaming parameter forces streaming strategy."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
            force_streaming=True,
        )

        # When forced, should use hybrid or chunked strategy
        assert result.strategy in ("hybrid", "chunked")


# =============================================================================
# Test Memory Estimation
# =============================================================================


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_memory_basic(self, coordinator: StreamingCoordinator) -> None:
        """Test basic memory estimation."""
        mem_mb = coordinator.estimate_memory(
            n_data=10_000,
            n_params=5,
        )

        assert mem_mb > 0
        assert isinstance(mem_mb, float)

    def test_estimate_memory_scales_with_data(
        self, coordinator: StreamingCoordinator
    ) -> None:
        """Test memory estimate scales with data size."""
        mem_small = coordinator.estimate_memory(n_data=1_000, n_params=5)
        mem_large = coordinator.estimate_memory(n_data=100_000, n_params=5)

        assert mem_large > mem_small

    def test_estimate_memory_scales_with_params(
        self, coordinator: StreamingCoordinator
    ) -> None:
        """Test memory estimate scales with parameter count."""
        mem_few_params = coordinator.estimate_memory(n_data=10_000, n_params=2)
        mem_many_params = coordinator.estimate_memory(n_data=10_000, n_params=50)

        assert mem_many_params > mem_few_params

    def test_estimate_memory_jacobian_dominates(
        self, coordinator: StreamingCoordinator
    ) -> None:
        """Test Jacobian memory dominates for large problems."""
        # Jacobian is n_data x n_params x 8 bytes
        n_data = 100_000
        n_params = 10
        jacobian_mb = (n_data * n_params * 8) / (1024 * 1024)

        estimated = coordinator.estimate_memory(n_data=n_data, n_params=n_params)

        # Estimated should be at least Jacobian size
        assert estimated >= jacobian_mb * 0.9  # Allow 10% tolerance


# =============================================================================
# Test Available Memory Detection
# =============================================================================


class TestAvailableMemoryDetection:
    """Tests for available memory detection."""

    def test_get_available_memory_positive(
        self, coordinator: StreamingCoordinator
    ) -> None:
        """Test available memory is positive."""
        mem_mb = coordinator.get_available_memory()

        assert mem_mb > 0
        assert isinstance(mem_mb, float)

    def test_get_available_memory_reasonable(
        self, coordinator: StreamingCoordinator
    ) -> None:
        """Test available memory is reasonable (not astronomical)."""
        mem_mb = coordinator.get_available_memory()

        # Should be less than 1TB (reasonable upper bound)
        assert mem_mb < 1_000_000  # 1TB in MB


# =============================================================================
# Test Memory Pressure
# =============================================================================


class TestMemoryPressure:
    """Tests for memory pressure calculation."""

    def test_memory_pressure_range(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test memory pressure is between 0 and 1."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
        )

        assert 0.0 <= result.memory_pressure <= 1.0

    def test_small_data_low_memory_pressure(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test small data has low memory pressure."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
        )

        # Small data should have low memory pressure
        assert result.memory_pressure < 0.5


# =============================================================================
# Test Workflow Hints
# =============================================================================


class TestWorkflowHints:
    """Tests for workflow hint handling."""

    def test_workflow_auto(self, coordinator: StreamingCoordinator, small_data) -> None:
        """Test workflow='auto' uses automatic selection."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
            workflow="auto",
        )

        # Auto should choose direct for small data
        assert result.strategy == "direct"

    def test_workflow_streaming_hint(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test workflow='streaming' suggests streaming strategy."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
            workflow="streaming",
        )

        # Streaming hint should use streaming-compatible strategy
        assert result.strategy in ("hybrid", "chunked", "direct")


# =============================================================================
# Test Memory Limit Override
# =============================================================================


class TestMemoryLimitOverride:
    """Tests for memory limit override."""

    def test_custom_memory_limit(
        self, coordinator: StreamingCoordinator, medium_data
    ) -> None:
        """Test custom memory limit is respected."""
        x, y = medium_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=10,
            memory_limit_mb=1.0,  # Very low limit (1MB)
        )

        # With very low limit, should use chunked/hybrid
        assert result.strategy in ("chunked", "hybrid")


# =============================================================================
# Test Reason String
# =============================================================================


class TestReasonString:
    """Tests for human-readable reason string."""

    def test_reason_is_non_empty(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test reason is a non-empty string."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
        )

        assert isinstance(result.reason, str)
        assert len(result.reason) > 0


# =============================================================================
# Test Hybrid Configuration
# =============================================================================


class TestHybridConfiguration:
    """Tests for hybrid configuration generation."""

    def test_configure_hybrid_returns_config(
        self, coordinator: StreamingCoordinator
    ) -> None:
        """Test configure_hybrid returns HybridStreamingConfig."""
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        config = coordinator.configure_hybrid(
            n_data=100_000,
            n_params=10,
            available_memory_mb=8000.0,
        )

        assert isinstance(config, HybridStreamingConfig)

    def test_configure_hybrid_chunk_size_positive(
        self, coordinator: StreamingCoordinator
    ) -> None:
        """Test configured chunk size is positive."""
        config = coordinator.configure_hybrid(
            n_data=100_000,
            n_params=10,
            available_memory_mb=8000.0,
        )

        assert config.chunk_size > 0


# =============================================================================
# Test Immutability
# =============================================================================


class TestImmutability:
    """Tests for StreamingDecision immutability."""

    def test_streaming_decision_is_frozen(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test StreamingDecision cannot be modified."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=2,
        )

        with pytest.raises((AttributeError, TypeError)):
            result.strategy = "hybrid"  # type: ignore[misc]


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_data_point(self, coordinator: StreamingCoordinator) -> None:
        """Test with single data point."""
        x = jnp.array([1.0])
        y = jnp.array([2.0])

        result = coordinator.decide(
            xdata=x,
            ydata=y,
            n_params=1,
        )

        # Single point should use direct
        assert result.strategy == "direct"

    def test_many_parameters(
        self, coordinator: StreamingCoordinator, small_data
    ) -> None:
        """Test with many parameters."""
        x, y = small_data

        result = coordinator.decide(
            xdata=jnp.asarray(x),
            ydata=jnp.asarray(y),
            n_params=50,
        )

        # Should still return valid decision
        assert result.strategy in ("direct", "chunked", "hybrid", "auto_memory")

    def test_2d_xdata(self, coordinator: StreamingCoordinator) -> None:
        """Test with 2D xdata."""
        xy = jnp.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        z = jnp.array([3, 6, 9, 12, 15])

        result = coordinator.decide(
            xdata=xy,
            ydata=z,
            n_params=3,
        )

        assert result.strategy in ("direct", "chunked", "hybrid", "auto_memory")

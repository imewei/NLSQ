"""Tests for CMA-ES population batching and data streaming."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization import (
    CMAESConfig,
    CMAESOptimizer,
    auto_configure_cmaes_memory,
    estimate_cmaes_memory_gb,
)
from nlsq.global_optimization.cmaes_config import is_evosax_available

# Skip all tests if evosax is not available
pytestmark = pytest.mark.skipif(
    not is_evosax_available(),
    reason="evosax not installed - skipping CMA-ES batching tests",
)


class TestPopulationBatching:
    """Tests for population batching feature."""

    def test_population_batching_equivalency(self):
        """Test that batched evaluation produces same results as unbatched."""

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        bounds = ([-5, -5], [5, 5])

        # Run with standard (no batching)
        config_std = CMAESConfig(
            popsize=20,
            max_generations=10,
            population_batch_size=None,
            seed=42,
        )
        opt_std = CMAESOptimizer(config=config_std)
        res_std = opt_std.fit(model, x, y, bounds=bounds)

        # Run with small batching (batch size 4 < popsize 20)
        # This forces 5 batches per generation
        config_batch = CMAESConfig(
            popsize=20,
            max_generations=10,
            population_batch_size=4,
            seed=42,
        )
        opt_batch = CMAESOptimizer(config=config_batch)
        res_batch = opt_batch.fit(model, x, y, bounds=bounds)

        # Parameter results should be identical because seed is same
        # and batching is mathematically equivalent
        assert jnp.allclose(res_std["popt"], res_batch["popt"], rtol=1e-5)

        # Check diagnostics
        diag_std = res_std["cmaes_diagnostics"]
        diag_batch = res_batch["cmaes_diagnostics"]

        assert diag_std["total_generations"] == diag_batch["total_generations"]
        assert jnp.allclose(diag_std["best_fitness"], diag_batch["best_fitness"])

    def test_population_batching_with_bipop(self):
        """Test batching with BIPOP where population size changes."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 5, 20)
        y = 2.5 * jnp.exp(-0.5 * x)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        # BIPOP will vary popsize. Fixed batch size should handle it.
        config = CMAESConfig(
            popsize=None,  # auto
            max_generations=50,
            population_batch_size=5,
            restart_strategy="bipop",
            max_restarts=2,
            seed=100,
        )

        opt = CMAESOptimizer(config=config)
        result = opt.fit(model, x, y, bounds=bounds)

        assert result["popt"] is not None
        assert "cmaes_diagnostics" in result


class TestDataStreaming:
    """Tests for data streaming (chunked data processing) feature."""

    def test_data_streaming_equivalency(self):
        """Test that streaming produces same results as non-streaming."""

        def model(x, a, b):
            return a * x + b

        # Use larger dataset to trigger chunking
        x = jnp.linspace(0, 10, 5000)
        y = 2.0 * x + 1.0 + 0.01 * jnp.sin(x)  # Small noise
        bounds = ([-5, -5], [5, 5])

        # Run without data streaming
        config_std = CMAESConfig(
            popsize=10,
            max_generations=10,
            data_chunk_size=None,
            seed=42,
        )
        opt_std = CMAESOptimizer(config=config_std)
        res_std = opt_std.fit(model, x, y, bounds=bounds)

        # Run with data streaming (1024 points per chunk = ~5 chunks)
        config_stream = CMAESConfig(
            popsize=10,
            max_generations=10,
            data_chunk_size=1024,
            seed=42,
        )
        opt_stream = CMAESOptimizer(config=config_stream)
        res_stream = opt_stream.fit(model, x, y, bounds=bounds)

        # Results should be identical (same seed, mathematically equivalent)
        assert jnp.allclose(res_std["popt"], res_stream["popt"], rtol=1e-5)

        # Verify that both find similar quality solutions by computing actual SSE
        # (the diagnostics best_fitness may differ due to evosax internal tracking)
        popt_std = res_std["popt"]
        popt_stream = res_stream["popt"]
        sse_std = float(jnp.sum((y - model(x, *popt_std)) ** 2))
        sse_stream = float(jnp.sum((y - model(x, *popt_stream)) ** 2))
        assert np.isclose(sse_std, sse_stream, rtol=1e-5), (
            f"SSE mismatch: {sse_std} vs {sse_stream}"
        )

    def test_data_streaming_with_non_divisible_size(self):
        """Test streaming when data size is not divisible by chunk size."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # 2500 points with chunk_size=1024 -> 2 full chunks + 1 partial (452 points)
        x = jnp.linspace(0, 5, 2500)
        y = 2.5 * jnp.exp(-0.5 * x)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        config = CMAESConfig(
            popsize=8,
            max_generations=20,
            data_chunk_size=1024,
            seed=123,
        )

        opt = CMAESOptimizer(config=config)
        result = opt.fit(model, x, y, bounds=bounds)

        # Should find reasonable parameters
        popt = result["popt"]
        assert popt is not None
        assert len(popt) == 2
        # Check parameters are in reasonable range
        assert 1.0 < popt[0] < 5.0  # a should be ~2.5
        assert 0.1 < popt[1] < 1.5  # b should be ~0.5

    def test_data_streaming_with_sigma(self):
        """Test data streaming with weighted residuals (sigma)."""

        def model(x, a, b):
            return a * x + b

        rng = np.random.default_rng(42)
        x = jnp.linspace(0, 10, 3000)
        # Heteroscedastic noise
        sigma = 0.1 + 0.05 * x
        y = 2.0 * x + 1.0 + jnp.array(rng.normal(0, np.asarray(sigma)))
        bounds = ([-5, -5], [5, 5])

        # Without streaming
        config_std = CMAESConfig(
            popsize=8,
            max_generations=15,
            data_chunk_size=None,
            seed=42,
        )
        opt_std = CMAESOptimizer(config=config_std)
        res_std = opt_std.fit(model, x, y, sigma=sigma, bounds=bounds)

        # With streaming
        config_stream = CMAESConfig(
            popsize=8,
            max_generations=15,
            data_chunk_size=1024,
            seed=42,
        )
        opt_stream = CMAESOptimizer(config=config_stream)
        res_stream = opt_stream.fit(model, x, y, sigma=sigma, bounds=bounds)

        # Results should be identical
        assert jnp.allclose(res_std["popt"], res_stream["popt"], rtol=1e-5)

    def test_combined_population_and_data_batching(self):
        """Test using both population batching and data streaming together."""

        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        x = jnp.linspace(0, 5, 4000)
        y = 2.5 * jnp.exp(-0.5 * x) + 0.3
        bounds = ([0.1, 0.01, -1.0], [10.0, 2.0, 2.0])

        # Both batching strategies enabled
        config = CMAESConfig(
            popsize=16,
            max_generations=20,
            population_batch_size=4,  # 4 batches of 4
            data_chunk_size=1024,  # ~4 data chunks
            seed=456,
        )

        opt = CMAESOptimizer(config=config)
        result = opt.fit(model, x, y, bounds=bounds)

        popt = result["popt"]
        assert popt is not None
        assert len(popt) == 3
        # Reasonable parameter estimates
        assert 1.5 < popt[0] < 4.0  # a ~2.5
        assert 0.2 < popt[1] < 1.0  # b ~0.5
        assert -0.5 < popt[2] < 1.0  # c ~0.3

    def test_data_streaming_with_bipop(self):
        """Test data streaming with BIPOP restart strategy."""

        # Use exponential model - simpler landscape with unique global minimum
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 5, 3000)
        y = 2.5 * jnp.exp(-0.4 * x)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        config = CMAESConfig(
            popsize=None,  # auto
            max_generations=30,
            data_chunk_size=1024,
            restart_strategy="bipop",
            max_restarts=2,
            seed=789,
        )

        opt = CMAESOptimizer(config=config)
        result = opt.fit(model, x, y, bounds=bounds)

        popt = result["popt"]
        assert popt is not None
        # Should find parameters close to true values
        assert 1.5 < popt[0] < 4.0  # a ~2.5
        assert 0.1 < popt[1] < 1.0  # b ~0.4


class TestMemoryEstimation:
    """Tests for memory estimation helper functions."""

    def test_estimate_memory_no_batching(self):
        """Test memory estimation without batching."""
        # 10M points, popsize=16, no batching
        memory_gb = estimate_cmaes_memory_gb(10_000_000, popsize=16)
        # Expected: (10M * 16 * 3 * 8) / 1024^3 + (10M * 2 * 8) / 1024^3
        # = 3.576 GB + 0.149 GB ≈ 3.7 GB
        assert 3.5 < memory_gb < 4.0

    def test_estimate_memory_with_population_batching(self):
        """Test memory estimation with population batching."""
        memory_gb = estimate_cmaes_memory_gb(
            10_000_000, popsize=16, population_batch_size=4
        )
        # With pop_batch=4: (10M * 4 * 3 * 8) / 1024^3 + (10M * 2 * 8) / 1024^3
        # ≈ 0.894 GB + 0.149 GB ≈ 1.0 GB
        assert 0.9 < memory_gb < 1.2

    def test_estimate_memory_with_data_streaming(self):
        """Test memory estimation with data streaming."""
        memory_gb = estimate_cmaes_memory_gb(
            10_000_000, popsize=16, data_chunk_size=50000
        )
        # With data_chunk=50K: (50K * 16 * 3 * 8) / 1024^3 ≈ 0.018 GB
        # No data overhead since streaming doesn't load full dataset
        assert memory_gb < 0.05

    def test_estimate_memory_combined_batching(self):
        """Test memory estimation with both batching strategies."""
        memory_gb = estimate_cmaes_memory_gb(
            100_000_000, popsize=32, population_batch_size=4, data_chunk_size=50000
        )
        # Minimal memory: (50K * 4 * 3 * 8) / 1024^3 ≈ 0.0045 GB
        assert memory_gb < 0.01

    def test_auto_configure_small_dataset(self):
        """Test auto-configuration for small dataset (no batching needed)."""
        pop_batch, data_chunk = auto_configure_cmaes_memory(
            1_000_000, popsize=16, available_memory_gb=8.0
        )
        # 1M points fits in 8GB without batching
        assert pop_batch is None
        assert data_chunk is None

    def test_auto_configure_medium_dataset(self):
        """Test auto-configuration for medium dataset (population batching only)."""
        pop_batch, data_chunk = auto_configure_cmaes_memory(
            50_000_000, popsize=16, available_memory_gb=4.0
        )
        # 50M points needs population batching to fit in 4GB
        assert pop_batch is not None
        assert pop_batch < 16
        assert data_chunk is None

    def test_auto_configure_large_dataset(self):
        """Test auto-configuration for large dataset (both strategies needed)."""
        pop_batch, data_chunk = auto_configure_cmaes_memory(
            100_000_000, popsize=16, available_memory_gb=0.5
        )
        # 100M points with only 0.5GB needs both strategies
        assert pop_batch == 1
        assert data_chunk is not None
        assert data_chunk >= 1024  # Minimum chunk size


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_data_chunk_size_validation(self):
        """Test that data_chunk_size < 1024 raises error."""
        with pytest.raises(ValueError, match="data_chunk_size must be >= 1024"):
            CMAESConfig(data_chunk_size=512)

    def test_data_chunk_size_none_allowed(self):
        """Test that data_chunk_size=None is allowed."""
        config = CMAESConfig(data_chunk_size=None)
        assert config.data_chunk_size is None

    def test_data_chunk_size_valid(self):
        """Test that valid data_chunk_size is accepted."""
        config = CMAESConfig(data_chunk_size=2048)
        assert config.data_chunk_size == 2048

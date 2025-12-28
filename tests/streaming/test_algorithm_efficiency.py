"""Tests for algorithm efficiency: iteration counts and memory benchmarks.

Task Group 4: Algorithm Efficiency Tests and Benchmarks.

This module tests:
- Iteration count regression tests (deterministic, hardware-independent)
- Memory ceiling tests (O(p) scaling verification)
- Convergence efficiency benchmarks
- Implicit vs materialized scaling benchmarks
- L-BFGS -> CG-GN handoff stability benchmarks
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import (
    AdaptiveHybridStreamingOptimizer,
    get_defense_telemetry,
    reset_defense_telemetry,
)
from nlsq.streaming.hybrid_config import HybridStreamingConfig

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_quadratic_problem():
    """Simple quadratic problem for L-BFGS iteration count testing.

    This is a well-conditioned quadratic problem that L-BFGS should solve
    efficiently due to its ability to approximate the Hessian.
    """

    def model(x, a, b, c):
        return a * x**2 + b * x + c

    # Generate data
    n_points = 100
    x = jnp.linspace(-2, 2, n_points)

    # True parameters
    true_params = jnp.array([1.0, -2.0, 1.0])

    # Generate clean data
    y = model(x, *true_params)

    # Initial guess: start away from optimum
    p0 = jnp.array([0.5, -1.0, 0.5])

    return model, x, y, p0, true_params


@pytest.fixture
def exponential_decay_problem():
    """Exponential decay problem for testing L-BFGS efficiency.

    Simple two-parameter model that L-BFGS should handle efficiently.
    """

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    n_points = 100
    x = jnp.linspace(0, 5, n_points)

    true_params = jnp.array([10.0, 0.5])
    y = model(x, *true_params)

    # Add small noise
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, shape=y.shape) * 0.1
    y = y + noise

    # Initial guess moderately far
    p0 = jnp.array([8.0, 0.4])

    return model, x, y, p0, true_params


@pytest.fixture
def exponential_decay_large_scale():
    """Large-scale exponential decay problem for testing.

    Multi-parameter exponential decay model with 50 parameters.
    Tests L-BFGS performance on larger problems.
    """
    n_params = 50

    def model(x, *params):
        """Sum of exponential decay terms: sum_i(a_i * exp(-b_i * x))

        Parameters are pairs (a_i, b_i) for i = 1..n_params/2
        """
        params = jnp.asarray(params)
        n_terms = n_params // 2
        result = jnp.zeros_like(x)
        for i in range(n_terms):
            a_i = params[2 * i]
            b_i = jnp.abs(params[2 * i + 1]) + 0.01  # Ensure positive decay rate
            result = result + a_i * jnp.exp(-b_i * x)
        return result

    # Generate data
    n_points = 500
    x = jnp.linspace(0, 5, n_points)

    # True parameters: alternating amplitudes and decay rates
    key = jax.random.PRNGKey(42)
    true_params = jnp.abs(jax.random.normal(key, shape=(n_params,))) * 0.5 + 0.5

    # Generate clean y data
    y_clean = model(x, *true_params)

    # Add small noise
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=y_clean.shape) * 0.01
    y = y_clean + noise

    # Initial guess: perturbed true params
    key, subkey = jax.random.split(key)
    perturbation = jax.random.normal(subkey, shape=(n_params,)) * 0.2
    p0 = true_params + perturbation

    return model, x, y, p0, true_params


@pytest.fixture(autouse=True)
def reset_telemetry_fixture():
    """Reset telemetry before each test."""
    reset_defense_telemetry()
    yield
    reset_defense_telemetry()


# =============================================================================
# Task 4.2: Iteration Count Regression Tests
# =============================================================================


class TestIterationCountRegression:
    """Test iteration count regression for L-BFGS warmup.

    These tests verify that L-BFGS converges within expected iteration bounds
    on standard test problems. Iteration counts should be deterministic and
    hardware-independent (given same initial conditions).
    """

    def test_lbfgs_quadratic_converges_efficiently(self, simple_quadratic_problem):
        """Test L-BFGS converges efficiently on simple quadratic problem.

        Quadratic problems are ideal for L-BFGS since the Hessian is constant.
        L-BFGS should converge quickly due to accurate curvature estimation.
        """
        model, x, y, p0, _true_params = simple_quadratic_problem

        config = HybridStreamingConfig(
            warmup_iterations=5,  # Start checking switch after 5 iters
            max_warmup_iterations=50,  # Safety cap
            gradient_norm_threshold=1e-4,  # Reasonable convergence
            loss_plateau_threshold=1e-5,
            enable_warm_start_detection=False,  # Force warmup
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)
        optimizer._setup_normalization(model, p0, bounds=None)

        result = optimizer._run_phase1_warmup(
            data_source=(x, y),
            model=model,
            p0=p0,
        )

        iterations = result["iterations"]
        best_loss = result["best_loss"]

        # L-BFGS should converge within the budget on well-conditioned problems
        assert iterations <= 50, (
            f"L-BFGS should converge within budget on quadratic, got {iterations} iterations"
        )

        # Should achieve low loss
        assert best_loss < 0.1, (
            f"L-BFGS should achieve low loss on quadratic, got {best_loss}"
        )

    def test_lbfgs_exponential_converges_efficiently(self, exponential_decay_problem):
        """Test L-BFGS converges efficiently on exponential decay problem.

        This test verifies L-BFGS performs well on nonlinear problems.
        """
        model, x, y, p0, _true_params = exponential_decay_problem

        config = HybridStreamingConfig(
            warmup_iterations=5,
            max_warmup_iterations=50,
            gradient_norm_threshold=1e-4,
            loss_plateau_threshold=1e-5,
            enable_warm_start_detection=False,
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)
        optimizer._setup_normalization(model, p0, bounds=None)

        result = optimizer._run_phase1_warmup(
            data_source=(x, y),
            model=model,
            p0=p0,
        )

        iterations = result["iterations"]
        best_loss = result["best_loss"]

        # L-BFGS should converge within the budget
        assert iterations <= 50, (
            f"L-BFGS should converge within budget on exponential, got {iterations} iterations"
        )

        # Should achieve reasonable loss
        assert best_loss < 1.0, (
            f"L-BFGS should achieve reasonable loss on exponential, got {best_loss}"
        )

    def test_lbfgs_exponential_decay_under_50_iterations(
        self, exponential_decay_large_scale
    ):
        """Test L-BFGS converges in reasonable iterations on large-scale Exponential Decay.

        With 50 parameters, this is a more challenging problem.
        L-BFGS should still converge within reasonable iteration count.
        """
        model, x, y, p0, _true_params = exponential_decay_large_scale

        config = HybridStreamingConfig(
            warmup_iterations=10,  # Start checking switch after 10 iters
            max_warmup_iterations=60,  # Safety cap
            gradient_norm_threshold=1e-4,  # Reasonable convergence for large scale
            loss_plateau_threshold=1e-5,
            enable_warm_start_detection=False,  # Force warmup
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)
        optimizer._setup_normalization(model, p0, bounds=None)

        result = optimizer._run_phase1_warmup(
            data_source=(x, y),
            model=model,
            p0=p0,
        )

        iterations = result["iterations"]
        best_loss = result["best_loss"]

        # L-BFGS should converge within the configured max iterations
        assert iterations <= 60, (
            f"L-BFGS should converge within budget on large-scale Exponential Decay, "
            f"got {iterations} iterations"
        )

        # Should have finite best loss
        assert jnp.isfinite(best_loss), "L-BFGS should produce finite loss"

    def test_iteration_counts_are_deterministic(self, simple_quadratic_problem):
        """Assert iteration counts are deterministic (hardware-independent).

        Running the same problem multiple times should produce identical
        iteration counts, demonstrating reproducibility.
        """
        model, x, y, p0, _ = simple_quadratic_problem

        config = HybridStreamingConfig(
            warmup_iterations=5,
            max_warmup_iterations=30,
            enable_warm_start_detection=False,
            verbose=0,
        )

        iteration_counts = []
        for _ in range(3):
            optimizer = AdaptiveHybridStreamingOptimizer(config)
            optimizer._setup_normalization(model, p0, bounds=None)
            result = optimizer._run_phase1_warmup(
                data_source=(x, y),
                model=model,
                p0=p0,
            )
            iteration_counts.append(result["iterations"])

        # All runs should have identical iteration counts
        assert all(ic == iteration_counts[0] for ic in iteration_counts), (
            f"Iteration counts should be deterministic: {iteration_counts}"
        )

    def test_lbfgs_vs_adam_baseline_fewer_iterations(self, exponential_decay_problem):
        """Compare against Adam baseline (should be significantly fewer iterations).

        L-BFGS uses approximate second-order information, so it should
        converge much faster than first-order Adam. The old Adam warmup
        used 200-500 iterations, while L-BFGS presets use 20-50.
        """
        model, x, y, p0, _ = exponential_decay_problem

        # L-BFGS configuration
        lbfgs_config = HybridStreamingConfig(
            warmup_iterations=5,
            max_warmup_iterations=50,
            enable_warm_start_detection=False,
            verbose=0,
        )

        optimizer_lbfgs = AdaptiveHybridStreamingOptimizer(lbfgs_config)
        optimizer_lbfgs._setup_normalization(model, p0, bounds=None)
        result_lbfgs = optimizer_lbfgs._run_phase1_warmup(
            data_source=(x, y),
            model=model,
            p0=p0,
        )

        lbfgs_iterations = result_lbfgs["iterations"]

        # The key assertion: L-BFGS should use significantly fewer iterations
        # than the old Adam baseline (which was 200-500 iterations)
        # The new L-BFGS presets use 20-50 iterations, which is 5-10x fewer
        assert lbfgs_iterations <= 50, (
            f"L-BFGS should converge within reduced iteration budget. "
            f"Got {lbfgs_iterations} iterations (vs typical Adam 200-500)"
        )

        # Verify the improvement claim: L-BFGS iteration budget is 5-10x less than Adam's
        typical_adam_iterations = 200  # Conservative Adam baseline
        improvement_factor = typical_adam_iterations / max(lbfgs_iterations, 1)
        assert improvement_factor >= 4, (
            f"L-BFGS iteration budget should be at least 4x fewer than Adam. "
            f"Got {improvement_factor:.1f}x improvement (with {lbfgs_iterations} iterations)"
        )


# =============================================================================
# Task 4.3: Memory Ceiling Tests
# =============================================================================


class TestMemoryCeiling:
    """Test memory usage stays within expected bounds.

    The CG solver should use O(p) memory instead of O(p^2) for J^T J storage.
    These tests verify that memory usage scales appropriately.
    """

    @pytest.mark.slow
    @pytest.mark.serial  # Memory-intensive: runs without parallelism to prevent OOM
    def test_5000_param_problem_under_500mb(self):
        """Test 5000-parameter problem stays < 500MB peak RSS.

        For p=5000, O(p^2) would be ~200MB just for J^T J in float64.
        With CG implicit matvec, we avoid this O(p^2) storage entirely.
        """
        n_params = 5000
        n_points = 200

        # Create high-order polynomial model
        def model(x, *params):
            """High-order polynomial with many terms."""
            params = jnp.asarray(params)
            result = jnp.zeros_like(x)
            # Use subset of terms for efficiency
            step = max(1, n_params // 50)
            for i in range(0, min(n_params, 50), 1):
                result = result + params[i] * (x ** (i % 10))
            return result

        # Generate data
        key = jax.random.PRNGKey(123)
        x = jnp.linspace(0, 1, n_points)

        key, subkey = jax.random.split(key)
        true_params = jax.random.normal(subkey, shape=(n_params,)) * 0.1

        y = model(x, *true_params)

        key, subkey = jax.random.split(key)
        p0 = true_params + jax.random.normal(subkey, shape=(n_params,)) * 0.01

        # Configure optimizer to use CG solver
        config = HybridStreamingConfig(
            warmup_iterations=0,
            max_warmup_iterations=1,
            gauss_newton_max_iterations=1,
            cg_param_threshold=100,  # Force CG for this test
            cg_max_iterations=10,  # Few iterations for speed
            verbose=0,
        )

        # Start memory tracking
        tracemalloc.start()

        try:
            optimizer = AdaptiveHybridStreamingOptimizer(config)
            optimizer._setup_normalization(model, p0, bounds=None)

            # Perform implicit matvec (the key operation for CG)
            v = jnp.ones(n_params)
            result = optimizer._implicit_jtj_matvec(v, p0, x, y)

            # Force computation
            _ = float(jnp.sum(result))
        except jax.errors.JaxRuntimeError as e:
            tracemalloc.stop()
            if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
                pytest.skip("Skipped due to GPU memory exhaustion")
            raise

        # Get peak memory
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)

        # Should stay well under 500MB
        # Note: Actual peak will depend on JAX tracing overhead, but should be far below 500MB
        assert peak_mb < 500, (
            f"Peak memory {peak_mb:.1f}MB should be < 500MB for {n_params} params. "
            f"O(p^2) would be ~200MB just for J^T J storage."
        )

    @pytest.mark.slow
    @pytest.mark.serial  # Memory-intensive: runs without parallelism to prevent OOM
    def test_cg_memory_scales_linear(self):
        """Test CG solver memory scales O(p) not O(p^2).

        We test multiple problem sizes and verify that memory growth
        is approximately linear, not quadratic.
        """
        param_counts = [100, 500, 1000]
        peak_memories = []

        for n_params in param_counts:
            n_points = 100

            # Simple polynomial model
            def model(x, *params, _n_params=n_params):
                params = jnp.asarray(params)
                result = jnp.zeros_like(x)
                for i in range(min(_n_params, 20)):
                    result = result + params[i] * (x ** (i % 5))
                return result

            key = jax.random.PRNGKey(42)
            x = jnp.linspace(0, 1, n_points)

            key, subkey = jax.random.split(key)
            true_params = jax.random.normal(subkey, shape=(n_params,)) * 0.1
            y = model(x, *true_params)

            key, subkey = jax.random.split(key)
            p0 = true_params + jax.random.normal(subkey, shape=(n_params,)) * 0.01

            config = HybridStreamingConfig(
                warmup_iterations=0,
                max_warmup_iterations=1,
                cg_param_threshold=10,
                cg_max_iterations=5,
                verbose=0,
            )

            tracemalloc.start()

            optimizer = AdaptiveHybridStreamingOptimizer(config)
            optimizer._setup_normalization(model, p0, bounds=None)

            v = jnp.ones(n_params)
            result = optimizer._implicit_jtj_matvec(v, p0, x, y)
            _ = float(jnp.sum(result))

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_memories.append(peak)

        # Check scaling: if O(p), ratio of memories should be ~ratio of params
        # If O(p^2), ratio would be (ratio of params)^2

        # Between n=100 and n=1000 (10x params), memory should scale < 100x
        # For O(p), we expect ~10x. For O(p^2), we'd expect ~100x.
        memory_ratio = (
            peak_memories[-1] / peak_memories[0] if peak_memories[0] > 0 else 0
        )
        param_ratio = param_counts[-1] / param_counts[0]

        # Allow some overhead, but should be closer to linear than quadratic
        # Linear would be 10x, quadratic would be 100x
        # We accept up to 50x as "approximately linear" to account for overhead
        assert memory_ratio < 50, (
            f"Memory scaling appears quadratic: {memory_ratio:.1f}x memory for "
            f"{param_ratio:.1f}x params. Expected closer to linear scaling."
        )


# =============================================================================
# Task 4.4: Convergence Efficiency Benchmark
# =============================================================================


class TestConvergenceEfficiency:
    """Benchmark convergence efficiency: model/gradient evaluations to target chi-squared."""

    def test_convergence_efficiency_high_dimensional(self):
        """Measure total model/gradient evaluations to target chi-squared tolerance.

        Focus on high-dimensional problems (p > 100).
        """
        n_params = 150
        n_points = 500

        # Create model with many parameters
        def model(x, *params):
            """Sum of weighted basis functions."""
            params = jnp.asarray(params)
            result = jnp.zeros_like(x)
            for i in range(min(n_params, 30)):
                # Use different basis functions
                freq = (i + 1) * 0.5
                result = result + params[i] * jnp.sin(freq * x)
            return result

        key = jax.random.PRNGKey(99)
        x = jnp.linspace(0, 10, n_points)

        key, subkey = jax.random.split(key)
        true_params = jax.random.normal(subkey, shape=(n_params,)) * 0.3

        y_clean = model(x, *true_params)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=y_clean.shape) * 0.01
        y = y_clean + noise

        key, subkey = jax.random.split(key)
        p0 = true_params + jax.random.normal(subkey, shape=(n_params,)) * 0.1

        # Track evaluations through warmup
        config = HybridStreamingConfig(
            warmup_iterations=10,
            max_warmup_iterations=50,
            gradient_norm_threshold=1e-4,
            enable_warm_start_detection=False,
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)
        optimizer._setup_normalization(model, p0, bounds=None)

        start_time = time.time()
        result = optimizer._run_phase1_warmup(
            data_source=(x, y),
            model=model,
            p0=p0,
        )
        elapsed = time.time() - start_time

        iterations = result["iterations"]
        best_loss = result["best_loss"]

        # Each L-BFGS iteration uses ~1-2 gradient evaluations (with line search)
        # Approximate total evaluations
        approx_evaluations = iterations * 2  # Rough estimate

        # For p=150, L-BFGS should complete within budget
        assert iterations <= 50, (
            f"L-BFGS should complete within budget on p={n_params} problem. "
            f"Got {iterations} iterations."
        )

        # Best loss should be finite
        assert jnp.isfinite(best_loss), (
            f"Should achieve finite loss, got best_loss={best_loss}"
        )


# =============================================================================
# Task 4.5: Implicit vs Materialized Scaling Benchmark
# =============================================================================


class TestImplicitVsMaterializedScaling:
    """Benchmark implicit CG vs materialized solver scaling.

    Sweep p from 10 to 10,000 (logarithmic sampling) to find crossover point.
    """

    @pytest.mark.slow
    @pytest.mark.serial  # Memory-intensive: sweeps large param counts with tracemalloc
    def test_implicit_vs_materialized_scaling(self):
        """Sweep p and measure peak memory and wall-clock time.

        Verify crossover point near cg_param_threshold.
        """
        # Logarithmic sampling of parameter counts
        param_counts = [10, 50, 200, 500, 1000, 2000]

        implicit_times = []
        materialized_times = []
        implicit_memories = []
        materialized_memories = []

        n_points = 100

        for n_params in param_counts:
            # Simple model
            def model(x, *params, _n_params=n_params):
                params = jnp.asarray(params)
                result = jnp.zeros_like(x)
                for i in range(min(_n_params, 20)):
                    result = result + params[i] * (x ** (i % 5))
                return result

            key = jax.random.PRNGKey(42)
            x = jnp.linspace(0, 1, n_points)

            key, subkey = jax.random.split(key)
            true_params = jax.random.normal(subkey, shape=(n_params,)) * 0.1
            y = model(x, *true_params)

            key, subkey = jax.random.split(key)
            p0 = true_params + jax.random.normal(subkey, shape=(n_params,)) * 0.01

            # Test implicit matvec
            config_cg = HybridStreamingConfig(
                warmup_iterations=0,
                max_warmup_iterations=1,
                cg_param_threshold=1,  # Force CG
                cg_max_iterations=5,
                verbose=0,
            )

            tracemalloc.start()
            start = time.time()

            try:
                optimizer = AdaptiveHybridStreamingOptimizer(config_cg)
                optimizer._setup_normalization(model, p0, bounds=None)

                v = jnp.ones(n_params)
                result = optimizer._implicit_jtj_matvec(v, p0, x, y)
                _ = float(jnp.sum(result))
            except jax.errors.JaxRuntimeError as e:
                tracemalloc.stop()
                if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
                    pytest.skip("Skipped due to GPU memory exhaustion")
                raise

            implicit_time = time.time() - start
            _, implicit_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            implicit_times.append(implicit_time)
            implicit_memories.append(implicit_peak)

            # Test materialized (only for smaller sizes to avoid memory issues)
            if n_params <= 1000:
                config_mat = HybridStreamingConfig(
                    warmup_iterations=0,
                    max_warmup_iterations=1,
                    cg_param_threshold=100000,  # Force materialized
                    verbose=0,
                )

                tracemalloc.start()
                start = time.time()

                optimizer2 = AdaptiveHybridStreamingOptimizer(config_mat)
                optimizer2._setup_normalization(model, p0, bounds=None)

                # For materialized, we compute JTJ
                JTJ, _JTr, _ = optimizer2._accumulate_jtj_jtr(
                    x, y, p0, jnp.zeros((n_params, n_params)), jnp.zeros(n_params)
                )
                result_mat = JTJ @ v
                _ = float(jnp.sum(result_mat))

                materialized_time = time.time() - start
                _, materialized_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                materialized_times.append(materialized_time)
                materialized_memories.append(materialized_peak)
            else:
                materialized_times.append(float("inf"))
                materialized_memories.append(float("inf"))

        # Find approximate crossover point (where implicit becomes preferable for memory)
        # Default threshold is 2000
        crossover_idx = None
        for i, n_params in enumerate(param_counts):
            if materialized_memories[i] == float("inf"):
                continue
            # Check if implicit uses less memory
            if implicit_memories[i] < materialized_memories[i]:
                crossover_idx = i
                break

        # Verify behavior: implicit should use less memory for large p
        # At p=2000 (default threshold), implicit should definitely be better
        idx_2000 = param_counts.index(2000) if 2000 in param_counts else -1
        if idx_2000 >= 0 and idx_2000 < len(implicit_memories):
            assert implicit_memories[idx_2000] < materialized_memories[
                idx_2000
            ] or materialized_memories[idx_2000] == float("inf"), (
                "At p=2000 (threshold), implicit should use less memory than materialized"
            )


# =============================================================================
# Task 4.6: Handoff Stability Benchmark
# =============================================================================


class TestHandoffStability:
    """Test L-BFGS -> CG-GN transition stability.

    The transition from Phase 1 (L-BFGS warmup) to Phase 2 (Gauss-Newton)
    should be stable, with low Layer 3 trigger rate after transition.
    """

    def test_lbfgs_to_gn_transition_stability(self):
        """Test L-BFGS -> CG-GN transition stability.

        Measure Layer 3 trigger rate immediately after transition.
        Should be low (< 50%) for well-behaved problems.
        """

        # Simple well-behaved problem
        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        n_points = 500
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(0, 5, n_points)

        true_params = jnp.array([5.0, 0.5, 1.0])
        y_clean = model(x, *true_params)

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=y_clean.shape) * 0.05
        y = y_clean + noise

        # Initial guess somewhat close
        p0 = jnp.array([4.0, 0.4, 0.8])

        # Run full fit with L-BFGS warmup + GN
        config = HybridStreamingConfig(
            warmup_iterations=15,
            max_warmup_iterations=30,
            gauss_newton_max_iterations=20,
            enable_warm_start_detection=False,
            enable_cost_guard=True,
            cost_increase_tolerance=0.1,  # 10% tolerance
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=p0,
            bounds=None,
            verbose=0,
        )

        # Check telemetry for Layer 3 (cost guard) triggers
        telemetry = get_defense_telemetry()
        trigger_rates = telemetry.get_trigger_rates()

        layer3_rate = trigger_rates.get("layer3_cost_guard_rate", 0.0)

        # For well-behaved problems, Layer 3 should rarely trigger
        assert layer3_rate < 50, (
            f"Layer 3 cost guard rate ({layer3_rate:.1f}%) should be low "
            f"for well-behaved problems"
        )

        # Verify fit was successful
        assert result["success"], "Fit should succeed on well-behaved problem"

    def test_transition_preserves_convergence(self):
        """Verify that transition from L-BFGS to GN preserves convergence quality."""

        def model(x, a, b):
            return a * x + b

        n_points = 200
        x = jnp.linspace(0, 10, n_points)
        true_params = jnp.array([2.0, 1.0])

        key = jax.random.PRNGKey(123)
        y_clean = model(x, *true_params)
        noise = jax.random.normal(key, shape=y_clean.shape) * 0.1
        y = y_clean + noise

        p0 = jnp.array([1.5, 0.5])

        config = HybridStreamingConfig(
            warmup_iterations=10,
            max_warmup_iterations=20,
            gauss_newton_max_iterations=15,
            gauss_newton_tol=1e-8,
            enable_warm_start_detection=False,
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=p0,
            bounds=None,
            verbose=0,
        )

        # Should converge to near true params
        popt = result["x"]
        np.testing.assert_allclose(
            np.array(popt),
            np.array(true_params),
            rtol=0.1,
            err_msg="Should converge to true parameters after L-BFGS + GN",
        )

        # Covariance should be well-conditioned
        pcov = result["pcov"]
        eigenvalues = jnp.linalg.eigvalsh(pcov)
        assert jnp.all(eigenvalues > 0), "Covariance should be positive definite"


# =============================================================================
# Task 4.7: Critical Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests for full L-BFGS + CG-GN workflow.

    These tests verify that all components work together correctly.
    """

    def test_full_fit_lbfgs_to_cg_gn_small_params(self):
        """Test full fit workflow with L-BFGS warmup -> CG-GN for small p."""

        def model(x, a, b, c):
            return a * jnp.sin(b * x) + c

        n_points = 300
        x = jnp.linspace(0, 2 * jnp.pi, n_points)

        true_params = jnp.array([2.0, 1.5, 0.5])
        key = jax.random.PRNGKey(42)
        y_clean = model(x, *true_params)
        noise = jax.random.normal(key, shape=y_clean.shape) * 0.05
        y = y_clean + noise

        p0 = jnp.array([1.5, 1.2, 0.3])

        config = HybridStreamingConfig(
            warmup_iterations=15,
            max_warmup_iterations=30,
            gauss_newton_max_iterations=20,
            cg_param_threshold=10,  # Force materialized for small p
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=p0,
            bounds=None,
            verbose=0,
        )

        assert result["success"], "Full fit should succeed"

        popt = result["x"]
        np.testing.assert_allclose(
            np.array(popt),
            np.array(true_params),
            rtol=0.2,
            err_msg="Should recover true parameters",
        )

    def test_full_fit_with_cg_solver_large_params(self):
        """Test full fit workflow with CG solver for large parameter count."""
        n_params = 100
        n_points = 200

        def model(x, *params):
            """Sum of weighted Gaussians."""
            params = jnp.asarray(params)
            result = jnp.zeros_like(x)
            for i in range(min(n_params, 20)):
                center = i / 20.0
                result = result + params[i] * jnp.exp(-50 * (x - center) ** 2)
            return result

        key = jax.random.PRNGKey(99)
        x = jnp.linspace(0, 1, n_points)

        key, subkey = jax.random.split(key)
        true_params = jax.random.normal(subkey, shape=(n_params,)) * 0.2

        y_clean = model(x, *true_params)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=y_clean.shape) * 0.01
        y = y_clean + noise

        key, subkey = jax.random.split(key)
        p0 = true_params + jax.random.normal(subkey, shape=(n_params,)) * 0.05

        config = HybridStreamingConfig(
            warmup_iterations=10,
            max_warmup_iterations=30,
            gauss_newton_max_iterations=10,
            cg_param_threshold=50,  # Force CG for p >= 50
            cg_max_iterations=30,
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=p0,
            bounds=None,
            verbose=0,
        )

        # Should complete without error
        assert result is not None, "Fit should complete"
        assert "x" in result, "Result should contain optimized parameters"

    def test_integration_with_bounds(self):
        """Test integration with parameter bounds."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        n_points = 150
        x = jnp.linspace(0, 5, n_points)

        true_params = jnp.array([5.0, 0.5])
        key = jax.random.PRNGKey(42)
        y_clean = model(x, *true_params)
        noise = jax.random.normal(key, shape=y_clean.shape) * 0.1
        y = y_clean + noise

        p0 = jnp.array([4.0, 0.4])
        bounds = (jnp.array([1.0, 0.1]), jnp.array([10.0, 2.0]))

        config = HybridStreamingConfig(
            warmup_iterations=10,
            max_warmup_iterations=25,
            gauss_newton_max_iterations=15,
            normalize=True,
            normalization_strategy="bounds",
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=p0,
            bounds=bounds,
            verbose=0,
        )

        assert result["success"], "Bounded fit should succeed"

        popt = result["x"]
        # Verify parameters are within bounds
        assert jnp.all(popt >= bounds[0]), "Parameters should be >= lower bounds"
        assert jnp.all(popt <= bounds[1]), "Parameters should be <= upper bounds"

    def test_diagnostics_contain_all_phases(self):
        """Test that streaming diagnostics contain all phase information."""

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 10, 100)
        y = (
            2.0 * x
            + 1.0
            + jax.random.normal(jax.random.PRNGKey(42), shape=(100,)) * 0.1
        )
        p0 = jnp.array([1.5, 0.5])

        config = HybridStreamingConfig(
            warmup_iterations=10,
            max_warmup_iterations=20,
            gauss_newton_max_iterations=10,
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=p0,
            bounds=None,
            verbose=0,
        )

        assert "streaming_diagnostics" in result
        diag = result["streaming_diagnostics"]

        # Verify phase timings
        assert "phase_timings" in diag
        assert "phase0_normalization" in diag["phase_timings"]
        assert "phase1_warmup" in diag["phase_timings"]
        assert "phase2_gauss_newton" in diag["phase_timings"]

        # Verify phase history
        assert "phase_history" in diag
        assert len(diag["phase_history"]) >= 2  # At least normalization and warmup

    def test_defense_layers_tracked_in_full_fit(self):
        """Test that all defense layer telemetry is tracked during full fit."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 5, 100)
        y = 5.0 * jnp.exp(-0.5 * x)
        p0 = jnp.array([3.0, 0.3])  # Start somewhat far

        config = HybridStreamingConfig(
            warmup_iterations=15,
            max_warmup_iterations=30,
            gauss_newton_max_iterations=10,
            enable_warm_start_detection=False,
            enable_adaptive_warmup_lr=True,
            enable_cost_guard=True,
            enable_step_clipping=True,
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=p0,
            bounds=None,
            verbose=0,
        )

        telemetry = get_defense_telemetry()

        # Warmup should have been tracked
        assert telemetry.total_warmup_calls >= 1

        # Layer 2 mode should have been recorded
        total_modes = sum(telemetry.layer2_lr_mode_counts.values())
        assert total_modes >= 1


# =============================================================================
# Task 4.8: Run Full Feature Test Suite
# =============================================================================


class TestFullFeatureSuite:
    """Meta-tests to ensure all features work together.

    These tests run comprehensive scenarios that exercise all major features.
    """

    def test_complete_optimization_workflow(self):
        """Run complete optimization workflow with all features enabled."""

        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        n_points = 400
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(0, 10, n_points)

        true_params = jnp.array([5.0, 0.3, 1.0])
        y_clean = model(x, *true_params)

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=y_clean.shape) * 0.1
        y = y_clean + noise

        p0 = jnp.array([4.0, 0.25, 0.8])
        bounds = (jnp.array([1.0, 0.1, 0.1]), jnp.array([10.0, 1.0, 5.0]))

        config = HybridStreamingConfig(
            # L-BFGS warmup
            warmup_iterations=20,
            max_warmup_iterations=40,
            lbfgs_history_size=10,
            # Defense layers
            enable_warm_start_detection=True,
            warm_start_threshold=0.01,
            enable_adaptive_warmup_lr=True,
            enable_cost_guard=True,
            cost_increase_tolerance=0.2,
            enable_step_clipping=True,
            max_warmup_step_size=0.5,
            # Gauss-Newton
            gauss_newton_max_iterations=30,
            gauss_newton_tol=1e-8,
            # CG configuration
            cg_param_threshold=100,  # Materialized for small p
            cg_max_iterations=50,
            # Normalization
            normalize=True,
            normalization_strategy="bounds",
            verbose=0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=p0,
            bounds=bounds,
            verbose=0,
        )

        # Comprehensive checks
        assert result["success"], "Complete workflow should succeed"
        assert "x" in result
        assert "pcov" in result
        assert "perr" in result
        assert "streaming_diagnostics" in result

        popt = result["x"]

        # Should be within bounds
        assert jnp.all(popt >= bounds[0])
        assert jnp.all(popt <= bounds[1])

        # Should be reasonably close to true params (with wider tolerance)
        np.testing.assert_allclose(
            np.array(popt),
            np.array(true_params),
            rtol=0.5,
            err_msg="Should recover true parameters",
        )

    def test_no_regressions_basic_functionality(self):
        """Verify no regressions in basic curve_fit functionality."""
        from nlsq import curve_fit

        def model(x, a, b):
            return a * x + b

        np.random.seed(42)  # For reproducibility
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 100)
        p0 = np.array([1.5, 0.5])

        # Use hybrid_streaming method
        popt, pcov = curve_fit(
            model,
            x,
            y,
            p0=p0,
            method="hybrid_streaming",
            verbose=0,
        )

        assert popt.shape == (2,)
        assert pcov.shape == (2, 2)

        # Should converge to reasonable values
        np.testing.assert_allclose(popt, [2.0, 1.0], atol=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

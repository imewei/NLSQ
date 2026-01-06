"""Comprehensive tests for nlsq/core/profiler.py.

Tests for TRFProfiler and NullProfiler classes used for timing TRF algorithm operations.
Enterprise-level coverage with scientific computing focus.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.profiler import NullProfiler, TRFProfiler

if TYPE_CHECKING:
    from jax import Array


# =============================================================================
# TRFProfiler Tests
# =============================================================================


class TestTRFProfilerInitialization:
    """Test TRFProfiler initialization."""

    def test_initialization_empty_lists(self) -> None:
        """Verify all timing lists are initialized empty."""
        profiler = TRFProfiler()

        assert profiler.ftimes == []
        assert profiler.jtimes == []
        assert profiler.svd_times == []
        assert profiler.ctimes == []
        assert profiler.gtimes == []
        assert profiler.gtimes2 == []
        assert profiler.ptimes == []

    def test_initialization_conversion_times_empty(self) -> None:
        """Verify all conversion timing lists are initialized empty."""
        profiler = TRFProfiler()

        assert profiler.svd_ctimes == []
        assert profiler.g_ctimes == []
        assert profiler.c_ctimes == []
        assert profiler.p_ctimes == []

    def test_slots_defined(self) -> None:
        """Verify __slots__ prevents arbitrary attribute assignment."""
        profiler = TRFProfiler()

        with pytest.raises(AttributeError):
            profiler.arbitrary_attr = "should fail"


class TestTRFProfilerTimeOperation:
    """Test TRFProfiler.time_operation() method."""

    def test_time_operation_fun(self) -> None:
        """Test timing function evaluation."""
        profiler = TRFProfiler()
        arr = jnp.array([1.0, 2.0, 3.0])

        result = profiler.time_operation("fun", arr)

        assert len(profiler.ftimes) == 1
        assert profiler.ftimes[0] >= 0.0
        np.testing.assert_array_equal(result, arr)

    def test_time_operation_jac(self) -> None:
        """Test timing Jacobian evaluation."""
        profiler = TRFProfiler()
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = profiler.time_operation("jac", arr)

        assert len(profiler.jtimes) == 1
        assert profiler.jtimes[0] >= 0.0
        np.testing.assert_array_equal(result, arr)

    def test_time_operation_svd(self) -> None:
        """Test timing SVD computation."""
        profiler = TRFProfiler()
        arr = jnp.array([1.0, 2.0, 3.0])

        result = profiler.time_operation("svd", arr)

        assert len(profiler.svd_times) == 1
        assert profiler.svd_times[0] >= 0.0
        np.testing.assert_array_equal(result, arr)

    def test_time_operation_cost(self) -> None:
        """Test timing cost computation."""
        profiler = TRFProfiler()
        arr = jnp.array([0.5])

        result = profiler.time_operation("cost", arr)

        assert len(profiler.ctimes) == 1
        assert profiler.ctimes[0] >= 0.0
        np.testing.assert_array_equal(result, arr)

    def test_time_operation_grad(self) -> None:
        """Test timing gradient computation."""
        profiler = TRFProfiler()
        arr = jnp.array([1.0, 2.0])

        result = profiler.time_operation("grad", arr)

        assert len(profiler.gtimes) == 1
        assert profiler.gtimes[0] >= 0.0
        np.testing.assert_array_equal(result, arr)

    def test_time_operation_grad_norm(self) -> None:
        """Test timing gradient norm computation."""
        profiler = TRFProfiler()
        arr = jnp.array([1.0])

        result = profiler.time_operation("grad_norm", arr)

        assert len(profiler.gtimes2) == 1
        assert profiler.gtimes2[0] >= 0.0
        np.testing.assert_array_equal(result, arr)

    def test_time_operation_param_update(self) -> None:
        """Test timing parameter update."""
        profiler = TRFProfiler()
        arr = jnp.array([1.0, 2.0, 3.0])

        result = profiler.time_operation("param_update", arr)

        assert len(profiler.ptimes) == 1
        assert profiler.ptimes[0] >= 0.0
        np.testing.assert_array_equal(result, arr)

    def test_time_operation_unknown_operation(self) -> None:
        """Test timing with unknown operation - no recording."""
        profiler = TRFProfiler()
        arr = jnp.array([1.0])

        result = profiler.time_operation("unknown", arr)

        # Should still return result
        np.testing.assert_array_equal(result, arr)
        # But no timing recorded
        assert len(profiler.ftimes) == 0
        assert len(profiler.jtimes) == 0

    def test_time_operation_multiple_calls(self) -> None:
        """Test multiple timing calls accumulate."""
        profiler = TRFProfiler()

        for i in range(5):
            arr = jnp.array([float(i)])
            profiler.time_operation("fun", arr)

        assert len(profiler.ftimes) == 5
        assert all(t >= 0.0 for t in profiler.ftimes)

    def test_time_operation_preserves_array_shape(self) -> None:
        """Test that time_operation preserves array shape and dtype."""
        profiler = TRFProfiler()
        arr_1d = jnp.array([1.0, 2.0, 3.0])
        arr_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result_1d = profiler.time_operation("fun", arr_1d)
        result_2d = profiler.time_operation("jac", arr_2d)

        assert result_1d.shape == (3,)
        assert result_2d.shape == (2, 2)
        assert result_1d.dtype == arr_1d.dtype
        assert result_2d.dtype == arr_2d.dtype


class TestTRFProfilerTimeConversion:
    """Test TRFProfiler.time_conversion() method."""

    def test_time_conversion_svd_convert(self) -> None:
        """Test timing SVD conversion."""
        profiler = TRFProfiler()
        start = time.time()
        time.sleep(0.001)  # Small delay

        profiler.time_conversion("svd_convert", start)

        assert len(profiler.svd_ctimes) == 1
        assert profiler.svd_ctimes[0] >= 0.001

    def test_time_conversion_grad_convert(self) -> None:
        """Test timing gradient conversion."""
        profiler = TRFProfiler()
        start = time.time()

        profiler.time_conversion("grad_convert", start)

        assert len(profiler.g_ctimes) == 1
        assert profiler.g_ctimes[0] >= 0.0

    def test_time_conversion_cost_convert(self) -> None:
        """Test timing cost conversion."""
        profiler = TRFProfiler()
        start = time.time()

        profiler.time_conversion("cost_convert", start)

        assert len(profiler.c_ctimes) == 1
        assert profiler.c_ctimes[0] >= 0.0

    def test_time_conversion_param_convert(self) -> None:
        """Test timing parameter conversion."""
        profiler = TRFProfiler()
        start = time.time()

        profiler.time_conversion("param_convert", start)

        assert len(profiler.p_ctimes) == 1
        assert profiler.p_ctimes[0] >= 0.0

    def test_time_conversion_unknown_operation(self) -> None:
        """Test conversion timing with unknown operation - no recording."""
        profiler = TRFProfiler()
        start = time.time()

        profiler.time_conversion("unknown", start)

        # No timing recorded
        assert len(profiler.svd_ctimes) == 0
        assert len(profiler.g_ctimes) == 0
        assert len(profiler.c_ctimes) == 0
        assert len(profiler.p_ctimes) == 0

    def test_time_conversion_multiple_calls(self) -> None:
        """Test multiple conversion timing calls accumulate."""
        profiler = TRFProfiler()

        for _ in range(3):
            profiler.time_conversion("svd_convert", time.time())

        assert len(profiler.svd_ctimes) == 3


class TestTRFProfilerGetTimingData:
    """Test TRFProfiler.get_timing_data() method."""

    def test_get_timing_data_empty(self) -> None:
        """Test get_timing_data returns empty lists initially."""
        profiler = TRFProfiler()
        data = profiler.get_timing_data()

        expected_keys = [
            "ftimes",
            "jtimes",
            "svd_times",
            "ctimes",
            "gtimes",
            "gtimes2",
            "ptimes",
            "svd_ctimes",
            "g_ctimes",
            "c_ctimes",
            "p_ctimes",
        ]

        assert set(data.keys()) == set(expected_keys)
        for key in expected_keys:
            assert data[key] == []

    def test_get_timing_data_with_operations(self) -> None:
        """Test get_timing_data returns recorded timings."""
        profiler = TRFProfiler()

        # Record some operations
        profiler.time_operation("fun", jnp.array([1.0]))
        profiler.time_operation("jac", jnp.array([[1.0]]))
        profiler.time_operation("svd", jnp.array([1.0]))
        profiler.time_conversion("svd_convert", time.time())

        data = profiler.get_timing_data()

        assert len(data["ftimes"]) == 1
        assert len(data["jtimes"]) == 1
        assert len(data["svd_times"]) == 1
        assert len(data["svd_ctimes"]) == 1

    def test_get_timing_data_returns_actual_lists(self) -> None:
        """Test that get_timing_data returns actual list references."""
        profiler = TRFProfiler()
        profiler.time_operation("fun", jnp.array([1.0]))

        data = profiler.get_timing_data()

        # Verify it's the actual list (same object)
        assert data["ftimes"] is profiler.ftimes


# =============================================================================
# NullProfiler Tests
# =============================================================================


class TestNullProfilerInitialization:
    """Test NullProfiler initialization."""

    def test_initialization_no_state(self) -> None:
        """Verify NullProfiler has no timing state."""
        profiler = NullProfiler()

        # NullProfiler should not have timing attributes
        assert not hasattr(profiler, "ftimes")
        assert not hasattr(profiler, "jtimes")

    def test_slots_defined(self) -> None:
        """Verify __slots__ prevents arbitrary attribute assignment."""
        profiler = NullProfiler()

        with pytest.raises(AttributeError):
            profiler.arbitrary_attr = "should fail"


class TestNullProfilerTimeOperation:
    """Test NullProfiler.time_operation() method."""

    def test_time_operation_returns_input_unchanged(self) -> None:
        """Test that time_operation returns input unchanged."""
        profiler = NullProfiler()
        arr = jnp.array([1.0, 2.0, 3.0])

        result = profiler.time_operation("fun", arr)

        # Should be the same object (no copy)
        assert result is arr

    def test_time_operation_all_operations(self) -> None:
        """Test that all operations return input unchanged."""
        profiler = NullProfiler()
        arr = jnp.array([1.0])

        operations = [
            "fun",
            "jac",
            "svd",
            "cost",
            "grad",
            "grad_norm",
            "param_update",
            "unknown",
        ]

        for op in operations:
            result = profiler.time_operation(op, arr)
            assert result is arr

    def test_time_operation_preserves_non_jax_types(self) -> None:
        """Test that time_operation works with non-JAX types."""
        profiler = NullProfiler()

        # Python int
        result_int = profiler.time_operation("fun", 42)
        assert result_int == 42

        # Python list
        result_list = profiler.time_operation("fun", [1, 2, 3])
        assert result_list == [1, 2, 3]


class TestNullProfilerTimeConversion:
    """Test NullProfiler.time_conversion() method."""

    def test_time_conversion_no_op(self) -> None:
        """Test that time_conversion does nothing."""
        profiler = NullProfiler()

        # Should not raise and return None
        result = profiler.time_conversion("svd_convert", time.time())

        assert result is None

    def test_time_conversion_all_operations(self) -> None:
        """Test that all conversion operations are no-ops."""
        profiler = NullProfiler()
        start = time.time()

        operations = [
            "svd_convert",
            "grad_convert",
            "cost_convert",
            "param_convert",
            "unknown",
        ]

        for op in operations:
            result = profiler.time_conversion(op, start)
            assert result is None


class TestNullProfilerGetTimingData:
    """Test NullProfiler.get_timing_data() method."""

    def test_get_timing_data_returns_empty_dict(self) -> None:
        """Test get_timing_data returns dict with all empty lists."""
        profiler = NullProfiler()
        data = profiler.get_timing_data()

        expected_keys = [
            "ftimes",
            "jtimes",
            "svd_times",
            "ctimes",
            "gtimes",
            "gtimes2",
            "ptimes",
            "svd_ctimes",
            "g_ctimes",
            "c_ctimes",
            "p_ctimes",
        ]

        assert set(data.keys()) == set(expected_keys)
        for key in expected_keys:
            assert data[key] == []

    def test_get_timing_data_after_operations(self) -> None:
        """Test get_timing_data returns empty even after 'operations'."""
        profiler = NullProfiler()

        # "Record" some operations
        profiler.time_operation("fun", jnp.array([1.0]))
        profiler.time_conversion("svd_convert", time.time())

        data = profiler.get_timing_data()

        # Still all empty
        for key in data:
            assert data[key] == []


# =============================================================================
# Interface Compatibility Tests
# =============================================================================


class TestProfilerInterfaceCompatibility:
    """Test that TRFProfiler and NullProfiler have compatible interfaces."""

    @pytest.mark.parametrize("profiler_cls", [TRFProfiler, NullProfiler])
    def test_time_operation_interface(self, profiler_cls) -> None:
        """Test both profilers have time_operation method."""
        profiler = profiler_cls()
        arr = jnp.array([1.0])

        result = profiler.time_operation("fun", arr)

        np.testing.assert_array_equal(result, arr)

    @pytest.mark.parametrize("profiler_cls", [TRFProfiler, NullProfiler])
    def test_time_conversion_interface(self, profiler_cls) -> None:
        """Test both profilers have time_conversion method."""
        profiler = profiler_cls()

        # Should not raise
        profiler.time_conversion("svd_convert", time.time())

    @pytest.mark.parametrize("profiler_cls", [TRFProfiler, NullProfiler])
    def test_get_timing_data_interface(self, profiler_cls) -> None:
        """Test both profilers return same structure from get_timing_data."""
        profiler = profiler_cls()
        data = profiler.get_timing_data()

        expected_keys = {
            "ftimes",
            "jtimes",
            "svd_times",
            "ctimes",
            "gtimes",
            "gtimes2",
            "ptimes",
            "svd_ctimes",
            "g_ctimes",
            "c_ctimes",
            "p_ctimes",
        }

        assert set(data.keys()) == expected_keys


# =============================================================================
# JAX Integration Tests (Scientific Computing)
# =============================================================================


class TestJAXIntegration:
    """Test profiler with real JAX operations."""

    def test_time_operation_with_jit_result(self) -> None:
        """Test timing a JIT-compiled operation."""
        import jax

        profiler = TRFProfiler()

        @jax.jit
        def compute(x):
            return jnp.dot(x, x)

        x = jnp.array([1.0, 2.0, 3.0])
        jit_result = compute(x)

        result = profiler.time_operation("cost", jit_result)

        assert len(profiler.ctimes) == 1
        np.testing.assert_allclose(result, 14.0)

    def test_time_operation_with_vmap_result(self) -> None:
        """Test timing a vmap operation."""
        import jax

        profiler = TRFProfiler()

        def f(x):
            return jnp.sum(x**2)

        batched_f = jax.vmap(f)
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        vmap_result = batched_f(x)

        result = profiler.time_operation("fun", vmap_result)

        assert len(profiler.ftimes) == 1
        np.testing.assert_allclose(result, [5.0, 25.0])

    def test_time_operation_with_grad_result(self) -> None:
        """Test timing gradient computation."""
        import jax

        profiler = TRFProfiler()

        def loss(x):
            return jnp.sum(x**2)

        grad_fn = jax.grad(loss)
        x = jnp.array([1.0, 2.0, 3.0])
        grad_result = grad_fn(x)

        result = profiler.time_operation("grad", grad_result)

        assert len(profiler.gtimes) == 1
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0])

    def test_time_operation_with_svd_result(self) -> None:
        """Test timing SVD computation."""
        profiler = TRFProfiler()

        A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _U, s, _Vt = jnp.linalg.svd(A, full_matrices=False)

        profiler.time_operation("svd", s)

        assert len(profiler.svd_times) == 1
        assert profiler.svd_times[0] >= 0.0

    def test_timing_realistic_workflow(self) -> None:
        """Test profiler with realistic TRF-like workflow."""
        import jax

        profiler = TRFProfiler()

        # Simulate TRF iteration timing
        def model(x, params):
            return params[0] * jnp.exp(-params[1] * x)

        x_data = jnp.linspace(0, 5, 100)
        y_data = 2.5 * jnp.exp(-1.3 * x_data) + 0.1 * jax.random.normal(
            jax.random.PRNGKey(0), (100,)
        )
        params = jnp.array([2.0, 1.0])

        # Time function evaluation
        f = model(x_data, params)
        profiler.time_operation("fun", f)

        # Time cost computation
        residuals = y_data - f
        cost = jnp.sum(residuals**2)
        profiler.time_operation("cost", cost)

        # Time Jacobian
        jac_fn = jax.jacfwd(lambda p: model(x_data, p))
        J = jac_fn(params)
        profiler.time_operation("jac", J)

        # Time gradient
        grad = J.T @ residuals
        profiler.time_operation("grad", grad)

        # Verify all timings recorded
        data = profiler.get_timing_data()
        assert len(data["ftimes"]) == 1
        assert len(data["ctimes"]) == 1
        assert len(data["jtimes"]) == 1
        assert len(data["gtimes"]) == 1


# =============================================================================
# Performance Tests
# =============================================================================


class TestProfilerPerformance:
    """Test profiler performance characteristics."""

    def test_null_profiler_zero_overhead(self) -> None:
        """Test that NullProfiler has minimal overhead.

        Uses absolute threshold instead of relative comparison because:
        - Direct array access (`_ = arr`) takes ~70μs for 1000 iterations
        - Method calls inherently add overhead regardless of implementation
        - CI environments have variable timing characteristics
        """
        null_profiler = NullProfiler()
        arr = jnp.array([1.0] * 1000)

        # Time through NullProfiler
        start = time.time()
        for _ in range(1000):
            _ = null_profiler.time_operation("fun", arr)
        profiled_time = time.time() - start

        # NullProfiler should complete 1000 calls in under 1 second
        # (generous threshold for CI environments with variable load)
        assert profiled_time < 1.0, (
            f"NullProfiler took {profiled_time:.3f}s for 1000 calls, expected < 1.0s"
        )

    def test_trf_profiler_accumulation_efficiency(self) -> None:
        """Test that TRFProfiler efficiently accumulates timings."""
        profiler = TRFProfiler()
        arr = jnp.array([1.0])

        # Time 1000 operations
        start = time.time()
        for _ in range(1000):
            profiler.time_operation("fun", arr)
        elapsed = time.time() - start

        assert len(profiler.ftimes) == 1000
        # Should complete in reasonable time (< 5s)
        assert elapsed < 5.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array_timing(self) -> None:
        """Test timing with empty array."""
        profiler = TRFProfiler()
        empty_arr = jnp.array([])

        result = profiler.time_operation("fun", empty_arr)

        assert len(profiler.ftimes) == 1
        assert result.shape == (0,)

    def test_scalar_timing(self) -> None:
        """Test timing with scalar."""
        profiler = TRFProfiler()
        scalar = jnp.array(1.0)

        result = profiler.time_operation("cost", scalar)

        assert len(profiler.ctimes) == 1
        assert result.shape == ()

    def test_large_array_timing(self) -> None:
        """Test timing with large array."""
        profiler = TRFProfiler()
        large_arr = jnp.ones(10000)

        result = profiler.time_operation("fun", large_arr)

        assert len(profiler.ftimes) == 1
        assert result.shape == (10000,)

    def test_multidimensional_array_timing(self) -> None:
        """Test timing with multidimensional array."""
        profiler = TRFProfiler()
        arr_3d = jnp.ones((10, 20, 30))

        result = profiler.time_operation("jac", arr_3d)

        assert len(profiler.jtimes) == 1
        assert result.shape == (10, 20, 30)

    def test_negative_elapsed_time_handling(self) -> None:
        """Test conversion timing with negative elapsed time (clock adjustment)."""
        profiler = TRFProfiler()
        # Simulate future time (negative elapsed)
        future_time = time.time() + 1.0

        profiler.time_conversion("svd_convert", future_time)

        # Should record negative time (unusual but not an error)
        assert len(profiler.svd_ctimes) == 1
        assert profiler.svd_ctimes[0] < 0


__all__ = [
    "TestEdgeCases",
    "TestJAXIntegration",
    "TestNullProfilerGetTimingData",
    "TestNullProfilerInitialization",
    "TestNullProfilerTimeConversion",
    "TestNullProfilerTimeOperation",
    "TestProfilerInterfaceCompatibility",
    "TestProfilerPerformance",
    "TestTRFProfilerGetTimingData",
    "TestTRFProfilerInitialization",
    "TestTRFProfilerTimeConversion",
    "TestTRFProfilerTimeOperation",
]

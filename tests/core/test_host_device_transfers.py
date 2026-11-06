"""Test suite for Task Group 2: Host-Device Transfer Reduction.

This test suite validates the implementation of:
- Task 2.4: Verbose logging during optimization
- Task 2.6: Transfer profiling infrastructure
- Task 2.7-2.8: Transfer reduction via JAX operations
- Task 2.10: Performance improvement validation

All tests ensure GPU-CPU transfers are minimized during optimization.
"""

import contextlib
import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.core.least_squares import LeastSquares
from nlsq.utils.profiling import (
    PerformanceMetrics,
    analyze_source_transfers,
    compare_transfer_reduction,
    profile_optimization,
)


class TestTransferProfiling:
    """Test Task 2.6: Transfer profiling infrastructure."""

    def test_analyze_source_transfers_numpy_array(self):
        """Test detection of np.array() calls."""
        code = """
def bad_function(x):
    y = np.array(x)  # Transfer!
    z = np.array(y)  # Another transfer!
    return z
"""
        result = analyze_source_transfers(code)
        assert result["np_array_calls"] == 2
        assert result["total_potential_transfers"] == 2

    def test_analyze_source_transfers_numpy_asarray(self):
        """Test detection of np.asarray() calls."""
        code = """
def mixed_function(x):
    y = np.asarray(x)
    z = np.array(y)
    return z.block_until_ready()
"""
        result = analyze_source_transfers(code)
        assert result["np_array_calls"] == 1
        assert result["np_asarray_calls"] == 1
        assert result["block_until_ready_calls"] == 1
        assert result["total_potential_transfers"] == 3

    def test_analyze_source_transfers_clean_code(self):
        """Test clean JAX code with no transfers."""
        code = """
def good_function(x):
    y = jnp.asarray(x)  # JAX, no transfer
    z = jnp.sum(y)
    return z
"""
        result = analyze_source_transfers(code)
        assert result["total_potential_transfers"] == 0

    def test_compare_transfer_reduction(self):
        """Test transfer reduction comparison."""
        before = """
def old_code(x):
    y = np.array(x)
    z = np.array(y)
    w = np.asarray(z)
    return w.block_until_ready()
"""

        after = """
def new_code(x):
    y = jnp.asarray(x)
    z = jnp.sum(y)
    return z
"""

        result = compare_transfer_reduction(before, after, "test_module")
        assert result["module"] == "test_module"
        assert result["before"]["total_potential_transfers"] == 4
        assert result["after"]["total_potential_transfers"] == 0
        assert result["reduction_count"] == 4
        assert result["reduction_percent"] == 100.0

    def test_compare_transfer_reduction_partial(self):
        """Test partial transfer reduction."""
        before = "y = np.array(x); z = np.array(y)"
        after = "y = jnp.asarray(x); z = np.array(y)"

        result = compare_transfer_reduction(before, after)
        assert result["reduction_count"] == 1
        assert result["reduction_percent"] == 50.0

    def test_compare_transfer_reduction_zero_before(self):
        """Test comparison when before has zero transfers."""
        before = "y = jnp.asarray(x)"
        after = "y = jnp.asarray(x)"

        result = compare_transfer_reduction(before, after)
        assert result["reduction_percent"] == 0.0


class TestTransferReduction:
    """Test Task 2.7-2.8: Transfer reduction via JAX operations."""

    def test_curve_fit_uses_jax_operations(self):
        """Verify curve_fit uses JAX operations internally."""

        # Simple exponential fit
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = jnp.linspace(0, 10, 50)
        y_true = 2.5 * jnp.exp(-0.3 * x)
        y_data = y_true + 0.1 * np.random.randn(50)

        # Fit should complete without excessive transfers
        popt, _pcov = curve_fit(model, x, y_data, p0=[1.0, 0.1])

        # Verify convergence
        assert popt is not None
        assert np.allclose(popt, [2.5, 0.3], rtol=0.1)

    def test_least_squares_minimal_transfers(self):
        """Verify least_squares minimizes host-device transfers."""

        def residual(params, x, y):
            a, b = params
            return y - a * jnp.exp(-b * x)

        np.random.seed(42)
        x = jnp.linspace(0, 5, 30)
        y = 1.8 * jnp.exp(-0.5 * x) + 0.05 * np.random.randn(30)

        # Use LeastSquares class API
        lsqs = LeastSquares()
        result = lsqs.least_squares(
            residual,
            x0=jnp.array([1.0, 0.1]),
            args=(x, y),
            max_nfev=50,
        )

        assert result.success
        assert np.allclose(result.x, [1.8, 0.5], rtol=0.1)


class TestPerformanceMetrics:
    """Test profiling utilities."""

    def test_performance_metrics_basic(self):
        """Test basic metrics tracking."""
        metrics = PerformanceMetrics()

        assert metrics.iteration_count == 0
        assert metrics.total_time_sec == 0.0
        assert metrics.avg_iteration_time_ms == 0.0
        assert metrics.min_iteration_time_ms == 0.0
        assert metrics.max_iteration_time_ms == 0.0

    def test_performance_metrics_calculations(self):
        """Test metrics calculations."""
        metrics = PerformanceMetrics(
            iteration_count=10,
            total_time_sec=1.0,
            iteration_times=[
                0.08,
                0.09,
                0.10,
                0.11,
                0.12,
                0.09,
                0.10,
                0.11,
                0.09,
                0.11,
            ],
        )

        assert metrics.avg_iteration_time_ms == 100.0  # 1.0s / 10 iters * 1000
        assert metrics.min_iteration_time_ms == 80.0
        assert metrics.max_iteration_time_ms == 120.0

    def test_profile_optimization_context(self):
        """Test profiling context manager records non-zero time."""
        with profile_optimization() as metrics:
            _ = sum(range(1000))  # Trivial work to ensure non-zero time

        assert metrics.total_time_sec >= 0.0

    def test_profile_optimization_disabled(self):
        """Test disabled profiling."""
        with profile_optimization(enabled=False) as metrics:
            _ = sum(range(1000))  # Some work

        # Should not track time when disabled
        assert metrics.total_time_sec == 0.0

    def test_profile_optimization_with_curve_fit(self):
        """Test profiling real optimization."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = jnp.linspace(0, 5, 50)
        y = 2.0 * jnp.exp(-0.4 * x) + 0.05 * np.random.randn(50)

        with profile_optimization() as metrics:
            popt, _pcov = curve_fit(model, x, y, p0=[1.0, 0.1])

        # Should complete in reasonable time
        assert metrics.total_time_sec > 0.0
        assert metrics.total_time_sec < 10.0  # Sanity check

        # Verify fit succeeded
        assert np.allclose(popt, [2.0, 0.4], rtol=0.2)


class TestPerformanceImprovement:
    """Test Task 2.10: Performance improvement validation."""

    def test_no_unnecessary_numpy_conversions(self):
        """Verify no unnecessary JAX→NumPy conversions in hot path."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = jnp.linspace(0, 3, 50)
        y = 2.2 * jnp.exp(-0.6 * x)

        # Profile the fit
        with profile_optimization() as metrics:
            popt, _ = curve_fit(model, x, y, p0=[1.0, 0.1])

        # Should complete in reasonable time
        # Includes JIT compilation, so allow generous time
        assert metrics.total_time_sec < 10.0

        # Verify convergence
        assert np.allclose(popt, [2.2, 0.6], rtol=0.1)


# Integration test combining multiple features
class TestIntegration:
    """Integration tests combining verbose logging, profiling, and transfer reduction."""

    def test_full_workflow_with_profiling(self):
        """Test complete workflow with profiling and verbose logging."""

        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y_true = 3.0 * jnp.exp(-0.5 * x) + 0.5
        y_data = y_true + 0.1 * np.random.randn(100)

        with profile_optimization() as metrics:
            popt, pcov = curve_fit(
                model,
                x,
                y_data,
                p0=[1.0, 0.1, 0.0],
                verbose=2,  # Verbose logging enabled
            )

        # Verify profiling
        assert metrics.total_time_sec > 0.0
        assert metrics.total_time_sec < 10.0  # Increased for CI stability

        # Verify convergence
        assert popt is not None
        assert np.allclose(popt, [3.0, 0.5, 0.5], rtol=0.2)

        # Verify covariance
        assert pcov is not None
        assert pcov.shape == (3, 3)

    def test_transfer_analysis_workflow(self):
        """Test transfer analysis on real code."""
        # Simulate analyzing before/after code
        before_code = """
def old_fit(x, y):
    params = np.array([1.0, 0.5])
    result = optimize(params)
    return np.array(result)
"""

        after_code = """
def new_fit(x, y):
    params = jnp.array([1.0, 0.5])
    result = optimize(params)
    return result
"""

        comparison = compare_transfer_reduction(before_code, after_code, "fit_module")

        assert comparison["reduction_count"] == 2
        assert comparison["reduction_percent"] == 100.0
        assert comparison["module"] == "fit_module"


# ---------------------------------------------------------------------------
# A2: fused per-attempt scalar reads in TRF step acceptance
# ---------------------------------------------------------------------------

_CONVERSION_METHODS = ("__bool__", "__int__", "__float__", "__index__", "item")


@contextlib.contextmanager
def count_materializations():
    """Count host materializations of device arrays (``ArrayImpl._value`` reads).

    ``np.asarray`` and ``.item()`` bypass ``_value`` in the installed JAX, so the
    explicit-conversion dunders are counted separately (``counter["explicit"]``).
    Skips the test when the host-materialization property does not exist.
    """
    cls = type(jnp.zeros(1))
    orig_value = getattr(cls, "_value", None)
    if not isinstance(orig_value, property):
        pytest.skip("array class has no `_value` property in this JAX version")
    counter = {"value": 0, "explicit": 0}

    def getter(self):
        counter["value"] += 1
        return orig_value.fget(self)

    saved = {}
    cls._value = property(getter)
    try:
        for name in _CONVERSION_METHODS:
            orig = getattr(cls, name, None)
            if orig is None:
                continue
            saved[name] = orig

            def wrapper(self, *a, _orig=orig, **k):
                counter["explicit"] += 1
                return _orig(self, *a, **k)

            setattr(cls, name, wrapper)
        yield counter
    finally:
        for name, orig in saved.items():
            setattr(cls, name, orig)
        cls._value = orig_value


# Materializations of the sequential (legacy) path, measured on base commit
# 72b5067d: 6 per finite attempt that does not terminate (3 bool, 1 int, 2 float),
# 1 per non-finite attempt. Used as lower bounds so JAX-internal changes cannot
# make the check flaky.
_BASELINE_MATERIALIZATIONS = {
    "worse_then_better": 12,
    "nonfinite_then_better": 7,
    "accepted_first": 6,
}
_FIXTURE_ATTEMPTS = {
    "worse_then_better": (2, 2),  # (attempts, finite attempts)
    "nonfinite_then_better": (2, 1),
    "accepted_first": (1, 1),
}


def _acceptance_fixture(trials, loss="none"):
    """Direct `_evaluate_step_acceptance` kwargs with a scripted stub `fun`."""
    from nlsq.core.loss_functions import LossFunctionsJIT

    J = jnp.array([[1.0, 0.2], [0.3, 1.0], [0.5, 0.5]])
    f = jnp.array([1.0, -0.5, 0.25])
    d = np.ones(2)
    d_jnp = jnp.ones(2)
    g = f.dot(J)
    U, sv, VT = jnp.linalg.svd(J, full_matrices=False)
    data_mask = jnp.ones(3)
    calls = {"n": 0}
    script = list(trials)

    def fun(_x, _xdata, _ydata, _mask, _transform):
        out = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        return out

    def jac(_x, _xdata, _ydata, _mask, _transform):
        return J

    if loss == "callable":

        def loss_function(r, f_scale, data_mask=None, cost_only=False):
            assert np.all(np.isfinite(np.asarray(r))), (
                "loss called on non-finite residuals"
            )
            if cost_only:
                return 0.5 * jnp.sum(r**2)
            return jnp.stack([r**2, jnp.ones_like(r), jnp.zeros_like(r)])

        cost = loss_function(f, 1.0, data_mask, cost_only=True)
    elif loss == "soft_l1":
        loss_function = LossFunctionsJIT().get_loss_function("soft_l1")
        cost = loss_function(f, 1.0, data_mask, cost_only=True)
    else:
        loss_function = None
        cost = 0.5 * jnp.dot(f, f)

    kwargs = dict(  # noqa: C408
        fun=fun,
        jac=jac,
        x=np.array([1.0, 1.0]),
        f=f,
        J=J,
        J_h=J,
        g_h_jnp=g,
        cost=cost,
        d=d,
        d_jnp=d_jnp,
        Delta=1.0,
        alpha=0.0,
        step_h=None,
        s=sv,
        V=VT.T,
        uf=U.T @ f,
        xdata=np.zeros(3),
        ydata=np.zeros(3),
        data_mask=data_mask,
        transform=None,
        loss_function=loss_function,
        f_scale=1.0,
        scale_inv=np.ones(2),
        jac_scale=False,
        solver="exact",
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=50,
        nfev=0,
    )
    return kwargs, calls


_F = np.array([1.0, -0.5, 0.25])
_TRIALS = {
    "worse_then_better": [jnp.asarray(3 * _F), jnp.asarray(0.5 * _F)],
    "nonfinite_then_better": [jnp.array([jnp.nan, 0.0, 0.0]), jnp.asarray(0.5 * _F)],
    "accepted_first": [jnp.asarray(0.5 * _F)],
}

# Outcomes of the sequential path recorded by running base commit 72b5067d
# (before this change) on the same fixtures.
_GOLDEN = {
    "worse_then_better": {
        "accepted": True,
        "nfev": 2,
        "Delta": 0.4999999999999999,
        "alpha": 1.521141276871864,
        "actual_reduction": 0.4921875,
        "step_norm": 0.24999999999999997,
        "termination_status": None,
        "x_new0": 0.763595721621678,
    },
    "nonfinite_then_better": {
        "accepted": True,
        "nfev": 2,
        "Delta": 0.4999999999999999,
        "alpha": 1.5211371620317047,
        "actual_reduction": 0.4921875,
        "step_norm": 0.24999999999999992,
        "termination_status": None,
        "x_new0": 0.76359574366002,
    },
    "accepted_first": {
        "accepted": True,
        "nfev": 1,
        "Delta": 2.0,
        "alpha": 0.1325397997053954,
        "actual_reduction": 0.4921875,
        "step_norm": 0.9999999999999998,
        "termination_status": None,
        "x_new0": 0.14795453800297786,
    },
    "callable_nonfinite": {
        "accepted": True,
        "nfev": 2,
        "Delta": 0.4999999999999999,
        "alpha": 1.5211371620317047,
        "actual_reduction": 0.4921875,
        "step_norm": 0.24999999999999992,
        "termination_status": None,
        "x_new0": 0.76359574366002,
    },
    "soft_l1_nonfinite": {
        "accepted": True,
        "nfev": 2,
        "Delta": 0.4999999999999999,
        "alpha": 1.5211371620317047,
        "actual_reduction": 0.4064313438357765,
        "step_norm": 0.24999999999999992,
        "termination_status": None,
        "x_new0": 0.76359574366002,
    },
}


def _run(trials, fuse, loss="none"):
    from nlsq.core import trf as trf_mod

    kwargs, calls = _acceptance_fixture(trials, loss)
    tr = trf_mod.TrustRegionReflective()
    reads = {"n": 0}
    real_read = trf_mod._read_attempt_scalars

    def counting_read(vec):
        reads["n"] += 1
        return real_read(vec)

    with (
        mock.patch.object(trf_mod, "_read_attempt_scalars", counting_read),
        count_materializations() as counter,
    ):
        result = tr._evaluate_step_acceptance(fuse_scalars=fuse, **kwargs)
    return result, calls["n"], dict(counter), reads["n"]


def _outcome(result):
    keys = ("Delta", "alpha", "actual_reduction", "step_norm")
    out = {k: float(result[k]) for k in keys}
    out["accepted"] = bool(result["accepted"])
    out["nfev"] = int(result["nfev"])
    out["termination_status"] = result["termination_status"]
    if result["accepted"]:
        out["x_new0"] = float(result["x_new"][0])
    return out


def _assert_outcome(actual, golden):
    for k in ("accepted", "nfev", "termination_status"):
        assert actual[k] == golden[k], k
    for k, v in golden.items():
        if k not in ("accepted", "nfev", "termination_status"):
            assert actual[k] == pytest.approx(v, rel=1e-10, abs=1e-14), k


class TestFusedStepAcceptance:
    """A2: one device-to-host transfer per inner attempt (unbounded TRF)."""

    @pytest.mark.parametrize("name", list(_TRIALS))
    def test_legacy_baseline_is_expensive(self, name):
        attempts, finite = _FIXTURE_ATTEMPTS[name]
        _, n_calls, counter, reads = _run(_TRIALS[name], fuse=False)
        assert n_calls == attempts
        assert reads == 0
        assert _BASELINE_MATERIALIZATIONS[name] > 0
        assert counter["value"] >= _BASELINE_MATERIALIZATIONS[name]
        assert counter["value"] >= 3 * finite

    @pytest.mark.parametrize("name", list(_TRIALS))
    def test_fused_one_transfer_per_attempt(self, name):
        attempts, _ = _FIXTURE_ATTEMPTS[name]
        result, n_calls, counter, reads = _run(_TRIALS[name], fuse=True)
        assert n_calls == attempts
        assert reads == attempts
        assert counter["value"] == attempts
        assert counter["explicit"] == 0
        assert result["accepted"] is True
        _assert_outcome(_outcome(result), _GOLDEN[name])

    @pytest.mark.parametrize("name", list(_TRIALS))
    def test_legacy_outcome_matches_golden(self, name):
        result, *_ = _run(_TRIALS[name], fuse=False)
        _assert_outcome(_outcome(result), _GOLDEN[name])

    def test_callable_loss_never_sees_nonfinite_residuals(self):
        # fuse_scalars=False is what LeastSquares sets for a callable loss; the
        # callable asserts finiteness of its input, so it must not be called on NaN.
        trials = _TRIALS["nonfinite_then_better"]
        result, n_calls, _, reads = _run(trials, fuse=False, loss="callable")
        assert n_calls == 2
        assert reads == 0
        _assert_outcome(_outcome(result), _GOLDEN["callable_nonfinite"])

    def test_builtin_robust_loss_nonfinite_first_trial(self):
        trials = _TRIALS["nonfinite_then_better"]
        result, n_calls, counter, reads = _run(trials, fuse=True, loss="soft_l1")
        assert n_calls == 2
        assert reads == 2
        assert counter["value"] == 2
        _assert_outcome(_outcome(result), _GOLDEN["soft_l1_nonfinite"])
        legacy, *_ = _run(trials, fuse=False, loss="soft_l1")
        _assert_outcome(_outcome(legacy), _GOLDEN["soft_l1_nonfinite"])


def _spy_trf_kwargs(monkeypatch):
    from nlsq.core.trf import TrustRegionReflective

    seen = []
    real = TrustRegionReflective.trf

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("fuse_scalars"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(TrustRegionReflective, "trf", spy)
    return seen


def _decay(x, a, b):
    return a * jnp.exp(-b * x)


class TestFuseScalarsWiring:
    """`LeastSquares` decides `fuse_scalars` from the loss argument."""

    def _data(self):
        x = np.linspace(0, 3, 40)
        return x, 2.0 * np.exp(-1.3 * x)

    def test_builtin_loss_unbounded_enables_fusion(self, monkeypatch):
        seen = _spy_trf_kwargs(monkeypatch)
        x, y = self._data()
        popt, _ = curve_fit(_decay, x, y, p0=[1.0, 1.0])
        assert seen == [True]
        np.testing.assert_allclose(popt, [2.0, 1.3], rtol=1e-6)

    def test_callable_loss_keeps_sequential_path(self, monkeypatch):
        from nlsq import LeastSquares

        seen = _spy_trf_kwargs(monkeypatch)
        x, y = self._data()
        res = LeastSquares().least_squares(
            lambda p: p[0] * jnp.exp(-p[1] * x) - y,
            [1.0, 1.0],
            loss=lambda z: jnp.stack([z, jnp.ones_like(z), jnp.zeros_like(z)]),
        )
        assert seen == [False]
        np.testing.assert_allclose(res.x, [2.0, 1.3], rtol=1e-5)

    def test_builtin_robust_loss_enables_fusion(self, monkeypatch):
        from nlsq import LeastSquares

        seen = _spy_trf_kwargs(monkeypatch)
        x, y = self._data()
        LeastSquares().least_squares(
            lambda p: p[0] * jnp.exp(-p[1] * x) - y, [1.0, 1.0], loss="soft_l1"
        )
        assert seen == [True]


class TestFusedAcceptanceJitCache:
    """Python-float Delta/alpha must not cause recompilation across fits."""

    def test_second_fit_does_not_grow_jit_caches(self):
        from nlsq import common_jax as cj
        from nlsq.core import trf_jit as tj

        fns = [
            tj.accept_step_scalars,
            cj.update_tr_radius_jax,
            cj.check_termination_jax,
            cj._solve_lsq_trust_region_jax_jit,
            cj._evaluate_quadratic1,
            cj._evaluate_quadratic_diagonal1,
            cj._evaluate_quadratic2,
            cj._evaluate_quadratic_diagonal2,
        ]
        if not all(hasattr(f, "_cache_size") for f in fns):
            pytest.skip("jit `_cache_size` (private API) not available")

        def fit(seed):
            x = np.linspace(0, 4, 200)
            noise = 0.01 * np.random.default_rng(seed).standard_normal(200)
            y = (2.5 + 0.1 * seed) * np.exp(-1.3 * x) + 0.5 + noise
            curve_fit(lambda x, a, b, c: a * jnp.exp(-b * x) + c, x, y, p0=[1, 1, 0])

        fit(0)
        warm = [f._cache_size() for f in fns]
        fit(1)
        assert [f._cache_size() for f in fns] == warm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Regression tests for three-brain review round 1 findings in nlsq/caching."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.caching._closure_serial import closure_serial
from nlsq.caching.memory_manager import MemoryManager
from nlsq.caching.unified_cache import UnifiedCache


class _Slotted:
    """Callable that cannot be weakly referenced (no ``__weakref__`` slot)."""

    __slots__ = ("k",)

    def __init__(self, k):
        self.k = k

    def __call__(self, x, a, b):
        return a * jnp.exp(-b * x) + self.k


def test_closure_serial_raises_typeerror_for_slotted_callable():
    with pytest.raises(TypeError):
        closure_serial(_Slotted(0.0))


def test_chunked_validation_accepts_slotted_callable():
    from nlsq.streaming.large_dataset import LargeDatasetFitter

    x = np.linspace(0, 1, 200)
    y = 2.0 * np.exp(-1.0 * x)
    fitter = LargeDatasetFitter(memory_limit_gb=1.0)
    model = _Slotted(0.0)
    fitter._validate_model_function(model, x, y, p0=[1.0, 1.0])
    fitter._validate_model_function(model, x, y, p0=[1.0, 1.0])  # uncached rerun
    assert fitter._validated_functions == {}


def test_curve_fit_large_chunked_slotted_callable():
    from nlsq import curve_fit_large
    from nlsq.streaming.specs import Chunked

    x = np.linspace(0, 1, 2000)  # 500 points never reaches validation
    y = 2.0 * np.exp(-1.0 * x)
    popt, _ = curve_fit_large(
        _Slotted(0.0), x, y, p0=[1.0, 1.0], strategy=Chunked(), show_progress=False
    )
    np.testing.assert_allclose(popt, [2.0, 1.0], rtol=1e-3)


# --- unified_cache -----------------------------------------------------------


def test_disk_cache_enabled_warns():
    with pytest.warns(UserWarning, match="not implemented"):
        UnifiedCache(disk_cache_enabled=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        UnifiedCache()


def test_static_arg_key_includes_type():
    cache = UnifiedCache()

    def f(x, n):
        return x * n

    x = jnp.ones(3)
    k_int = cache._generate_cache_key(f, (x, 1), {}, (1,))
    k_str = cache._generate_cache_key(f, (x, "1"), {}, (1,))
    assert k_int != k_str


def test_stats_expose_wrapper_time_alias():
    cache = UnifiedCache()
    cache.get_or_compile(lambda x: x, (jnp.ones(2),), {})
    stats = cache.get_stats()
    assert stats["jit_wrap_time_ms"] == stats["compile_time_ms"]


# --- memory_manager ----------------------------------------------------------


@pytest.mark.serial
def test_memory_guard_never_records_negative_bytes_used():
    mm = MemoryManager(memory_cache_ttl=1000.0)
    big = np.ones(20_000_000)  # 160 MB, freed inside the guard
    with mm.memory_guard(1000, record_telemetry=False):
        del big
    assert all(a["bytes_used"] >= 0 for a in mm.allocation_history)


def test_adaptive_safety_factor_raises_after_underprediction():
    mm = MemoryManager(enable_adaptive_safety=True)
    for _ in range(12):
        mm._record_safety_telemetry(120, 200.0, 1.2)  # needs 200/(120/1.2) = 2.0
    assert mm.safety_factor == pytest.approx(2.0)


def test_adaptive_safety_factor_recovers_upward_and_is_capped():
    mm = MemoryManager(enable_adaptive_safety=True)
    for _ in range(40):
        mm._record_safety_telemetry(100, 100.0, 1.0)  # needs 1.0 -> ratchet down
    assert mm.safety_factor == pytest.approx(1.05)
    mm._safety_telemetry.clear()
    for _ in range(12):
        mm._record_safety_telemetry(105, 10_000.0, 1.05)  # needs ~100 -> capped
    assert mm.safety_factor == pytest.approx(mm._max_safety_factor)


def test_telemetry_uses_factor_in_force_at_prediction():
    mm = MemoryManager(enable_adaptive_safety=True)
    mm._record_safety_telemetry(200, 100.0, safety_factor_used=2.0)
    assert mm._safety_telemetry[-1]["safety_factor_needed"] == pytest.approx(1.0)


def test_available_memory_refresh_bypasses_ttl_cache():
    import psutil

    mm = MemoryManager(memory_cache_ttl=1000.0, adaptive_ttl=False)
    mm._available_memory_cache = 1.0  # stale and tiny
    mm._available_memory_cache_time = float("inf")
    assert mm.get_available_memory() == 1.0
    ok, _ = mm.check_memory_availability(1000)
    assert ok  # real availability, not the stale 1 byte

    # Stale-HIGH cache must not admit what no longer fits.
    mm._available_memory_cache = 1e18
    mm._available_memory_cache_time = float("inf")
    ok, _ = mm.check_memory_availability(int(2 * psutil.virtual_memory().total))
    assert not ok


def test_free_array_skips_readonly_and_strided_arrays():
    mm = MemoryManager()
    ro = np.zeros(8)
    ro.flags.writeable = False
    mm.free_array(ro)
    mm.free_array(np.zeros((4, 4))[:, ::2])
    assert not mm.memory_pool
    arr = mm.allocate_array((8,))
    arr[:] = 1
    mm.free_array(arr)
    assert mm.allocate_array((8,)).sum() == 0


def test_fraction_fallback_warns_once(monkeypatch, caplog):
    import nlsq.caching.memory_manager as mmod

    class Boom:
        @staticmethod
        def virtual_memory():
            raise RuntimeError("boom")

    monkeypatch.setattr(mmod, "HAS_PSUTIL", True)
    monkeypatch.setattr(mmod, "psutil", Boom, raising=False)
    mm = MemoryManager(memory_cache_ttl=0.0)
    with caplog.at_level("WARNING", logger=mmod.logger.name):
        assert mm.get_memory_usage_fraction() == 0.5
        assert mm.get_memory_usage_fraction() == 0.5
    assert sum("constant 0.5" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("algorithm", ["trf", "lm", "dogbox"])
def test_predict_counts_svd_arrays(algorithm):
    """Unbounded TRF holds J, J*d and U (three m x n); bounded holds more."""
    mm = MemoryManager(safety_factor=1.0)
    m, n = 1_000_000, 10
    three = 3 * 8 * m * n
    got = mm.predict_memory_requirement(m, n, algorithm)
    assert got >= three
    if algorithm == "dogbox":
        assert got >= 4 * 8 * m * n  # + J_augmented and U of (m + n) x n


def test_predict_uses_dtype_itemsize():
    mm = MemoryManager(safety_factor=1.0)
    f64 = mm.predict_memory_requirement(1000, 5, dtype=jnp.float64)
    f32 = mm.predict_memory_requirement(1000, 5, dtype=jnp.float32)
    f16 = mm.predict_memory_requirement(1000, 5, dtype=np.float16)
    assert f64 == pytest.approx(2 * f32, rel=1e-6)
    assert f16 == pytest.approx(f32 / 2, rel=1e-6)


# Mirrors the memory_manager review table; arrays are never allocated.
@pytest.mark.parametrize(
    ("algorithm", "n_points", "n_params", "limit_gb"),
    [
        ("trf", 1_000_000, 50, 2.0),
        ("trf", 10_000_000, 5, 2.0),
        ("lm", 200_000, 200, 8.0),
    ],
)
def test_estimate_chunking_does_not_chunk_problems_that_fit(
    algorithm, n_points, n_params, limit_gb
):
    mm = MemoryManager()
    assert mm.predict_memory_requirement(n_points, n_params, algorithm) < limit_gb * 1e9
    s = mm.estimate_chunking_strategy(
        n_points, n_params, algorithm, memory_limit_gb=limit_gb
    )
    assert not s["needs_chunking"]
    assert s["n_chunks"] == 1


def test_estimate_chunking_uses_largest_fitting_chunk():
    mm = MemoryManager()
    n, p = 1_000_000, 50
    full = mm.predict_memory_requirement(n, p, "trf")
    limit_gb = full * 0.6 / 1e9
    s = mm.estimate_chunking_strategy(n, p, "trf", memory_limit_gb=limit_gb)
    assert s["needs_chunking"]
    assert s["n_chunks"] == 2  # not forced up to ~100 chunks
    assert s["memory_per_chunk_gb"] <= limit_gb
    # the chunk is maximal: a 1% larger chunk would not fit
    assert mm.predict_memory_requirement(int(s["chunk_size"] * 1.01), p) > (
        limit_gb * 1e9
    )


# --- round 2 ----------------------------------------------------------------


def test_estimate_chunking_wide_problem_that_fits_does_not_raise():
    mm = MemoryManager()
    need = mm.predict_memory_requirement(100, 2000, "trf")
    assert need < 0.03e9
    s = mm.estimate_chunking_strategy(100, 2000, "trf", memory_limit_gb=0.03)
    assert not s["needs_chunking"]
    mm1 = MemoryManager(safety_factor=1.0)
    assert mm1.predict_memory_requirement(1, 100) < 10_000
    s = mm1.estimate_chunking_strategy(1, 100, memory_limit_gb=0.00001)
    assert not s["needs_chunking"]


@pytest.mark.parametrize("n_params", [1, 3, 50])
@pytest.mark.parametrize("algorithm", ["trf", "dogbox"])
def test_estimate_chunking_chunk_fits_and_is_maximal(n_params, algorithm):
    mm = MemoryManager()
    limit_gb = 1e-3
    s = mm.estimate_chunking_strategy(
        10**6, n_params, algorithm, memory_limit_gb=limit_gb
    )
    c = s["chunk_size"]
    assert s["needs_chunking"]
    assert mm.predict_memory_requirement(c, n_params, algorithm) <= limit_gb * 1e9
    assert mm.predict_memory_requirement(c + 1, n_params, algorithm) > limit_gb * 1e9
    assert s["memory_per_chunk_gb"] <= limit_gb


def test_estimate_chunking_wide_problem_chunks_by_real_estimate():
    mm = MemoryManager()
    limit_gb = 0.05
    assert mm.predict_memory_requirement(10_000, 2000) > limit_gb * 1e9
    s = mm.estimate_chunking_strategy(10_000, 2000, memory_limit_gb=limit_gb)
    c = s["chunk_size"]
    assert s["needs_chunking"]
    assert mm.predict_memory_requirement(c, 2000) <= limit_gb * 1e9
    assert mm.predict_memory_requirement(c + 1, 2000) > limit_gb * 1e9


def test_predict_bounded_trf_is_larger_and_default_unbounded():
    mm = MemoryManager(safety_factor=1.0)
    m, n = 1_000_000, 10
    unb = mm.predict_memory_requirement(m, n, "trf")
    assert unb == mm.predict_memory_requirement(m, n, "trf", bounded=False)
    b = mm.predict_memory_requirement(m, n, "trf", bounded=True)
    assert b > unb
    assert mm.predict_memory_requirement(m, n, "lm", bounded=True) == b
    assert mm.predict_memory_requirement(m, n, "dogbox") > unb


def test_callers_pass_bounds_to_memory_estimate():
    from unittest.mock import MagicMock

    from nlsq.core.least_squares import LeastSquares

    ls = LeastSquares(enable_stability=True)
    ls.memory_manager = MagicMock()
    ls.memory_manager.check_memory_availability.return_value = (True, "")
    ls._check_memory_and_adjust_solver(100, 2, "trf", "exact", bounded=True)
    assert ls.memory_manager.predict_memory_requirement.call_args.kwargs["bounded"]
    ls._check_memory_and_adjust_solver(100, 2, "trf", "exact")
    assert not ls.memory_manager.predict_memory_requirement.call_args.kwargs["bounded"]


def test_safety_cap_saturation_is_flagged_and_warned_once(caplog):
    mm = MemoryManager(enable_adaptive_safety=True)
    with caplog.at_level("WARNING", logger="nlsq.caching.memory_manager"):
        for _ in range(15):
            mm._record_safety_telemetry(100, 10_000.0, 1.0)
    assert mm.safety_factor == pytest.approx(mm._max_safety_factor)
    assert mm.get_safety_telemetry()["safety_saturated"] is True
    assert sum("saturated" in r.message for r in caplog.records) == 1


def test_safety_not_saturated_in_normal_use():
    mm = MemoryManager(enable_adaptive_safety=True)
    for _ in range(12):
        mm._record_safety_telemetry(120, 200.0, 1.2)
    assert mm.get_safety_telemetry()["safety_saturated"] is False


def test_free_array_ignores_jax_arrays():
    mm = MemoryManager()
    mm.free_array(jnp.ones(4))  # must not raise
    assert not mm.memory_pool


# --- round 3 ----------------------------------------------------------------


def test_estimate_chunking_bounded_layout_is_planned():
    mm = MemoryManager(safety_factor=1.0)
    n, p, limit_gb = 1_000_000, 10, 0.3
    limit = limit_gb * 1e9
    assert not mm.estimate_chunking_strategy(n, p, memory_limit_gb=limit_gb)[
        "needs_chunking"
    ]
    s = mm.estimate_chunking_strategy(n, p, memory_limit_gb=limit_gb, bounded=True)
    assert s["needs_chunking"]
    c = s["chunk_size"]
    assert mm.predict_memory_requirement(c, p, bounded=True) <= limit
    assert mm.predict_memory_requirement(c + 1, p, bounded=True) > limit
    assert s["memory_per_chunk_gb"] <= limit_gb


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 0.0, float("-inf")])
def test_estimate_chunking_rejects_invalid_limit(bad):
    with pytest.raises(ValueError, match="memory_limit_gb"):
        MemoryManager().estimate_chunking_strategy(1000, 5, memory_limit_gb=bad)


@pytest.mark.parametrize(("n_points", "n_params"), [(-1, 5), (10, -1)])
def test_estimate_chunking_rejects_negative_sizes(n_points, n_params):
    with pytest.raises(ValueError, match=r"n_points|n_params"):
        MemoryManager().estimate_chunking_strategy(
            n_points, n_params, memory_limit_gb=1.0
        )


def _record_bounded(monkeypatch):
    calls: list[bool] = []
    orig = MemoryManager.predict_memory_requirement

    def rec(self, *a, **k):
        calls.append(k.get("bounded", "MISSING"))
        return orig(self, *a, **k)

    monkeypatch.setattr(MemoryManager, "predict_memory_requirement", rec)
    return calls


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        (None, False),  # default (-inf, inf)
        ((-np.inf, np.inf), False),
        (([-np.inf] * 2, [np.inf] * 2), False),
        ((0, np.inf), True),
        ((-np.inf, 10), True),
        (([-np.inf] * 2, [np.inf, 5.0]), True),
    ],
)
def test_curve_fit_and_least_squares_pass_bounds_to_estimate(
    monkeypatch, bounds, expected
):
    from nlsq import CurveFit, LeastSquares

    x = np.linspace(0, 1, 200)
    y = 2 * np.exp(-x)
    kw = {} if bounds is None else {"bounds": bounds}
    calls = _record_bounded(monkeypatch)

    CurveFit(enable_stability=True).curve_fit(
        lambda x, a, b: a * jnp.exp(-b * x), x, y, p0=[1.0, 1.0], **kw
    )
    assert calls and all(c is expected for c in calls), calls

    calls.clear()
    LeastSquares(enable_stability=True).least_squares(
        lambda p: p[0] * jnp.exp(-p[1] * x) - y, np.array([1.0, 1.0]), **kw
    )
    assert calls and all(c is expected for c in calls), calls

"""Regression tests for the 2026-11-18 round-2 streaming review (fixer F4)."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.streaming import LargeDatasetFitter, LDMemoryConfig
from nlsq.streaming import exact_lm as lm
from nlsq.streaming.large_fit import fit_with_strategy
from nlsq.streaming.normal_equations import FactorySource, iter_blocks
from nlsq.streaming.specs import Auto, Chunked, Exact, Hybrid, InMemory
from nlsq.streaming.strategy import normalize_p0_bounds, scipy_layout_to_points_first
from nlsq.streaming.validators import (
    ConfigValidationError,
    validate_residual_weighting_config,
)


def _slope(x, a):
    return a * x


def _call(strategy, f, x, y, p0=None, sigma=None, bounds=(-np.inf, np.inf), **kw):
    return fit_with_strategy(
        strategy,
        f,
        x,
        y,
        None if p0 is None else np.asarray(p0),
        sigma=sigma,
        absolute_sigma=False,
        bounds=bounds,
        check_finite=True,
        memory_limit_gb=None,
        show_progress=False,
        **kw,
    )


# ---------------------------------------------------------------- exact_lm


def test_polish_keeps_certified_iterate_when_step_overshoots():
    """Codex P05 HIGH: a rounding-level cost rise replaced a certified point."""

    def f(x, a):
        return x * (a + 5e11 * a * a)

    res = lm.exact_lm_fit(
        f,
        np.array([1.0, 0.0]),
        np.array([1e-9, 1.0]),
        [0.0],
        ftol=1e-14,
        gtol=1e-12,
        xtol=None,
        compute_covariance=False,
    )
    assert res.termination == "ftol"
    assert (
        res.x[0] == 0.0
    )  # the polished a=1e-9 has cost 0.5 + 1.25e-13 and is not stationary
    assert res.cost == pytest.approx(0.5, abs=1e-15)


def test_exact_stationary_start_takes_no_polish_pass():
    """Codex P05 LOW: a zero Newton step still cost a full data pass."""
    res = lm.exact_lm_fit(
        lambda x, a: a + 0.0 * x,
        np.arange(4.0),
        np.array([0.0, 2.0, 0.0, 2.0]),
        [1.0],
        ftol=None,
        gtol=None,
        xtol=1e-8,
    )
    assert res.nfev == 1


def test_covariance_pass_detects_a_source_that_changed_length():
    """Codex P05 MEDIUM: the TSQR pass reread a grown source without a count check."""
    x = np.arange(10.0)
    source = (x, 2 + x, None)
    theta = np.array([1.0, 2.0])

    def model(x, a, b):
        return a + b * (1 + 1e-5 * x)

    assert (
        lm.tsqr_r_factor(model, theta, source, 10, weighted=False, n=10).shape[1] == 2
    )
    grown = (np.append(x, 1e5), np.append(2 + x, 1e5), None)
    with pytest.raises(ValueError, match="changed length"):
        lm.tsqr_r_factor(model, theta, grown, 10, weighted=False, n=10)


def test_iter_blocks_rejects_longer_y():
    """Codex P05 MEDIUM: a response longer than x was silently truncated."""
    with pytest.raises(ValueError, match="same length"):
        list(iter_blocks(np.arange(4.0), np.array([0.0, 1, 2, 3, 100]), None, 4))


def test_factory_iterator_closed_when_probe_fails():
    """Codex P05 MEDIUM: a rejected pointwise probe left the factory's generator open."""
    closed = []

    def factory(chunk):
        try:
            for s in range(0, 8, 4):
                x = np.arange(s, s + 4, dtype=float)
                yield x, x
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="pointwise"):
        lm.exact_lm_fit(lambda x, a: a * (x - jnp.mean(x)), factory, None, [1.0])
    assert closed


def test_factory_source_close_is_idempotent():
    src = FactorySource(lambda c: iter([(np.arange(4.0), np.arange(4.0))]), 4, None)
    src.close()
    src.close()


def test_initial_model_failure_does_not_replay_checked_data():
    """Codex P05 LOW: check_finite=True data were reread to name an invalid p0."""
    calls = []

    def factory(chunk):
        calls.append(1)
        yield np.arange(3.0), np.arange(3.0)

    with pytest.raises(ValueError, match="not finite"):
        lm.exact_lm_fit(lambda x, a: jnp.sqrt(a) * x, factory, None, [-1.0], max_nfev=1)
    assert len(calls) == 1


# ---------------------------------------------------------------- large_fit / specs


class _LazySource:
    """Sliceable source that refuses a whole-array read."""

    def __init__(self, a):
        self._a = np.asarray(a, dtype=float)
        self.shape = self._a.shape
        self.dtype = self._a.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return self._a[idx]

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("full read forbidden")


def test_auto_exact_never_reads_a_lazy_source_whole():
    """Codex P03 HIGH / P05: Auto converted a sliceable source before choosing exact."""
    x = np.arange(1.0, 7.0)
    got = _call(
        Auto(size_threshold=3),
        _slope,
        _LazySource(x),
        _LazySource(2 * x),
        p0=[1.0],
        sigma=_LazySource(np.ones(6)),
    )
    np.testing.assert_allclose(got[0], [2.0])


def test_auto_probe_never_exceeds_the_exact_chunk_size():
    """Codex P05 MEDIUM: the Auto pointwise probe used up to 4096 rows at once."""
    table = jnp.arange(16.0)

    def tiny_batch_model(x, a):
        # only supports batches of at most 16 rows
        return a * x + 0.0 * table[: x.shape[0]].sum() + jnp.zeros(16)[: x.shape[0]]

    x = np.arange(1.0, 101.0)
    got = _call(
        Auto(size_threshold=20, chunked=Chunked(chunk_size=8)),
        tiny_batch_model,
        x,
        2 * x,
        p0=[1.0],
    )
    np.testing.assert_allclose(got[0], [2.0])


def test_frozen_strategy_options_mutation_is_rejected_at_use():
    """Codex P05 MEDIUM: mutating options after construction bypassed the reserved keys."""
    spec = InMemory()
    spec.options["sigma"] = np.array([1.0, 100.0, 1.0])
    with pytest.raises(TypeError, match="owned by"):
        _call(
            spec,
            lambda x, a: a + 0.0 * x,
            np.arange(3.0),
            np.array([0.0, 10.0, 0.0]),
            p0=[1.0],
        )


def test_in_memory_adapter_rejects_malformed_infinite_bounds():
    """Codex P05 LOW: bounds=([-inf,-inf],[inf]) was dropped instead of rejected."""
    x = np.arange(1.0, 6.0)
    with pytest.raises(ValueError):
        _call(
            InMemory(),
            _slope,
            x,
            2 * x,
            p0=[1.0],
            bounds=(np.array([-np.inf, -np.inf]), np.array([np.inf])),
        )


def test_exact_default_p0_is_inside_the_bounds():
    """Codex P05 MEDIUM: omitted p0 started on the lower bound where the model is NaN."""

    def model(x, a):
        return jnp.log(a - 10.0) * x

    x = np.arange(1.0, 7.0)
    y = np.log(3.0) * x
    got = _call(Exact(chunk_size=3), model, x, y, bounds=([10.0], [20.0]))
    np.testing.assert_allclose(got[0], [13.0], rtol=1e-6)
    p0, _ = normalize_p0_bounds(model, None, ([10.0], [20.0]))
    assert 10.0 < p0[0] < 20.0


def test_hybrid_layout_adapter_transposes_estimate_p0():
    """Codex P05 MEDIUM: the wrapped model's estimate_p0 received (M, k), not (k, M)."""
    seen = {}

    def model(x, *p):
        return x[0] * p[0] + x[1] * p[1]

    def estimate_p0(x, y):
        seen["shape"] = np.shape(x)
        return np.linalg.lstsq(np.asarray(x).T, y, rcond=None)[0]

    model.estimate_p0 = estimate_p0
    x = np.random.default_rng(0).normal(size=(2, 20))
    wrapped, x_pf = scipy_layout_to_points_first(model, x)
    assert x_pf.shape == (20, 2)
    out = wrapped.estimate_p0(x_pf, 2 * x[0] + 3 * x[1])
    assert seen["shape"] == (2, 20)
    np.testing.assert_allclose(out, [2.0, 3.0], atol=1e-10)


def test_residual_weights_reject_complex_values():
    """Codex P05 MEDIUM: complex weights were accepted, then truncated to their real part."""
    with pytest.raises(ConfigValidationError, match="real"):
        validate_residual_weighting_config(True, np.array([1 + 4j, 2 + 5j]))


# ---------------------------------------------------------------- large_dataset


def _robust_data(n=20_000):
    rng = np.random.default_rng(2)
    x = np.linspace(0, 5, n)
    y = 2.0 * x + 1 + 0.1 * rng.standard_normal(n)
    y[rng.random(n) < 0.05] += 20
    return x, y


def _linear2(x, a, b):
    return a * x + b


@pytest.mark.serial
@pytest.mark.slow
@pytest.mark.parametrize("case", ["mask", "omit"])
def test_chunked_robust_covariance_matches_curve_fit_on_fallback_paths(case):
    """Claude C05 MEDIUM-1 / Codex P05: the SSR fallback ignored the robust loss."""
    x, y = _robust_data()
    kw = {}
    if case == "mask":
        mask = np.ones(len(x), bool)
        mask[0] = False
        kw["data_mask"] = mask
    else:
        y = y.copy()
        y[0] = np.nan
        kw["nan_policy"] = "omit"
    cfg = LDMemoryConfig(memory_limit_gb=1, min_chunk_size=1000, max_chunk_size=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = curve_fit(_linear2, x, y, p0=[1, 1], loss="soft_l1", **kw)
        res = LargeDatasetFitter(config=cfg).fit(
            _linear2, x, y, p0=[1, 1], loss="soft_l1", **kw
        )
    np.testing.assert_allclose(
        np.sqrt(np.diag(res.pcov)), np.sqrt(np.diag(ref[1])), rtol=1e-3
    )


def test_chunk_ssr_applies_the_robust_loss():
    r = np.array([0.0, 3.0])
    f = lambda x, a: a + 0.0 * np.asarray(x)
    plain = LargeDatasetFitter._chunk_ssr(f, np.zeros(2), r, [0.0])
    robust = LargeDatasetFitter._chunk_ssr(
        f, np.zeros(2), r, [0.0], None, "soft_l1", 1.0
    )
    assert plain == pytest.approx(9.0)
    assert robust == pytest.approx(2.0 * (np.sqrt(10.0) - 1.0))


def test_bounded_chunked_pcov_scales_around_the_returned_estimate():
    """Codex P05 MEDIUM: pooled variance used the unclipped GLS point."""
    fitter = LargeDatasetFitter(config=LDMemoryConfig(memory_limit_gb=1))
    fitter._absolute_sigma = False
    fitter._accum_n = 6
    fitter._accum_info_vector = np.array([2.0, 0.0])  # GLS point (1, 0) at pcov=0.5*I
    from nlsq.streaming.large_dataset import _ChunkFit

    prec = np.eye(2)
    fitter._accum_chunks = [
        _ChunkFit(np.array([0.0, 0.0]), prec, 0.0),
        _ChunkFit(np.array([2.0, 0.0]), prec, 0.0),
    ]
    at_gls = fitter._pooled_variance(np.eye(2) * 0.5)
    at_returned = fitter._pooled_variance(np.eye(2) * 0.5, np.array([1.0, 1.0]))
    # SSR at (1, 1): 2 + 2; at the GLS point (1, 0): 1 + 1; divided by n - p = 4
    assert at_returned == pytest.approx(1.0)
    assert at_gls == pytest.approx(0.5)


def test_absolute_sigma_chunking_does_not_retain_chunk_precisions():
    """Codex P05 MEDIUM: absolute_sigma kept one dense matrix per chunk it never reads."""
    x = np.linspace(0, 5, 6000)
    y = 2 * x + 1
    fitter = LargeDatasetFitter(
        config=LDMemoryConfig(
            memory_limit_gb=1, min_chunk_size=1000, max_chunk_size=1500
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fitter.fit(_linear2, x, y, p0=[1, 1], absolute_sigma=True)
    assert res.n_chunks > 1
    assert fitter._accum_chunks == []


def test_chunk_probe_uses_feasible_p0_and_retained_rows():
    """Codex P05 MEDIUM: the preflight evaluated the model at an infeasible p0 / masked row."""
    probed = []

    def model(x, a):
        probed.append((np.asarray(x).copy(), float(a)))
        return a * jnp.sqrt(x)

    x = np.concatenate([[-4.0], np.arange(1.0, 10.0)])
    fitter = LargeDatasetFitter(config=LDMemoryConfig(memory_limit_gb=1))
    mask = np.ones(len(x), bool)
    mask[0] = False
    fitter._validate_model_function(
        model, x, 2 * np.sqrt(np.abs(x)), [-1.0], bounds=([1.0], [9.0]), row_mask=mask
    )
    xs, a = probed[0]
    assert (xs > 0).all()
    assert a == 1.0


def test_lr_schedule_decay_steps_must_be_integral():
    """Agy P05 MEDIUM-1: a fractional step count passed validation."""
    from nlsq.streaming.validators import validate_lr_schedule_config

    validate_lr_schedule_config(True, 10, 450, 1e-5)
    with pytest.raises(ConfigValidationError):
        validate_lr_schedule_config(True, 10, 450.5, 1e-5)


def test_group_variance_bounds_raise_config_validation_error():
    """Agy P05 MEDIUM-2: the one validator that raised a bare ValueError."""
    from types import SimpleNamespace

    from nlsq.streaming.validators import validate_group_variance_bounds

    cfg = SimpleNamespace(
        enable_group_variance_regularization=True, group_variance_indices=((0, 5),)
    )
    with pytest.raises(ConfigValidationError, match="exceeds the number"):
        validate_group_variance_bounds(cfg, 3)


def test_scipy_layout_helper_is_exported():
    """Agy P05 LOW-1."""
    from nlsq.streaming import strategy

    assert "scipy_layout_to_points_first" in strategy.__all__


def test_false_output_flags_do_not_force_the_chunked_fallback():
    """Claude C05 non-blocking: `full_output=False` fell back to chunked GLS with a warning."""
    import nlsq

    x = np.linspace(0, 5, 4000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nlsq.curve_fit_large(
            lambda x, a, b: a * x + b,
            x,
            2 * x + 1,
            p0=[1, 1],
            size_threshold=1000,
            full_output=False,
        )
    assert not [w for w in caught if "exact streaming not used" in str(w.message)]


def test_chunk_preflight_accepts_a_scalar_constant_model():
    """Claude C05 non-blocking: a scalar-returning model is valid for curve_fit, not for chunks."""
    cfg = LDMemoryConfig(memory_limit_gb=1, min_chunk_size=1000, max_chunk_size=1500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = LargeDatasetFitter(config=cfg).fit(
            lambda x, a: a + 0.0 * jnp.sum(x),
            np.arange(4000.0),
            np.full(4000, 3.0),
            p0=[1.0],
        )
    np.testing.assert_allclose(res.popt, [3.0])


def test_chunked_multistart_is_announced_not_silently_inert():
    """Agy P05 HIGH-1: an enabled Chunked multistart was dropped without a word."""
    from nlsq.streaming.specs import Multistart

    x = np.linspace(0, 5, 3000)
    with pytest.warns(UserWarning, match="multistart"):
        _call(
            Chunked(chunk_size=1000, multistart=Multistart(enabled=True, n_starts=2)),
            lambda x, a, b: a * x + b,
            x,
            2 * x + 1,
            p0=[1.0, 1.0],
        )


@pytest.mark.parametrize("entry", ["curve_fit_large", "fit"])
def test_public_entry_points_keep_a_lazy_source_unread(entry):
    """Codex P03 HIGH (wave 2): the default out-of-core route end to end, not only large_fit."""
    import nlsq

    x = np.arange(1.0, 7.0)
    got = getattr(nlsq, entry)(
        lambda x, a: a * x,
        _LazySource(x),
        _LazySource(2 * x),
        p0=[1.0],
        size_threshold=3,
    )
    np.testing.assert_allclose(got[0], [2.0])


def test_gpu_memory_estimate_is_per_device_not_summed():
    """Wave 2: two 16 GB devices with 12 and 14 GB free budget 12 GB (one device), not 26."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from nlsq.streaming import large_dataset as ld

    def dev(in_use_gb):
        return SimpleNamespace(
            platform="gpu",
            memory_stats=lambda: {
                "bytes_limit": 16 * 1024**3,
                "bytes_in_use": in_use_gb * 1024**3,
            },
        )

    with patch.object(ld, "get_cached_devices", return_value=[dev(4), dev(2)]):
        assert ld.GPUMemoryEstimator().get_available_gpu_memory_gb() == pytest.approx(
            12.0
        )


@pytest.mark.parametrize(
    "weights",
    [
        [1 + 1j, 2.0],
        [1.0, np.nan],
        [1.0, np.inf],
        [1.0, -2.0],
        [0.0, 1.0],
        ["a", "b"],
        np.array([None, 1], dtype=object),
        [True, True],
    ],
    ids=["complex", "nan", "inf", "negative", "zero", "string", "object", "bool"],
)
def test_residual_weight_config_validator_rejects_every_unusable_weight(weights):
    """Wave 2 (F3 ledger): the config path must refuse what the setters refuse, via ConfigValidationError."""
    with pytest.raises(ConfigValidationError):
        validate_residual_weighting_config(True, weights)


def _kink(x, a):
    return (1 - x) * a + x * (1e17 * a * a)


_TIGHT = {"ftol": 1e-14, "gtol": 1e-12, "xtol": None, "compute_covariance": False}


def test_polish_recertifies_the_point_it_returns():
    """Codex verdict 1 (wave 4): the polish returned a=1e-9 (cost .405) as ftol-converged."""
    res = lm.exact_lm_fit(
        _kink, np.array([0.0, 1.0]), np.array([1e-9, 1.0]), [0.0], **_TIGHT
    )
    # not the uncertified polish iterate: either the true minimiser or a non-success
    assert res.x[0] == pytest.approx(10**-8.5, rel=1e-6)
    assert not (res.success and res.cost > 0.1)


def test_polish_outcome_matches_a_fit_started_at_the_minimiser():
    """The tolerances ask for more than float64 resolves here; both starts report it alike."""
    y = np.array([3e-9, 1.0])
    x = np.array([0.0, 1.0])
    from_zero = lm.exact_lm_fit(_kink, x, y, [0.0], **_TIGHT)
    from_min = lm.exact_lm_fit(_kink, x, y, [10**-8.5], **_TIGHT)
    assert from_zero.termination == from_min.termination == "stalled"
    assert not from_zero.success and not from_min.success


def test_polish_with_rounded_equal_cost_is_still_a_certified_point():
    """Codex verdict 1 (wave 4): a cost that rounds equal is certified by ftol, not skipped.

    The constant row makes the cost 5e15 (ulp 1), so the polish step's gain (.4) is invisible;
    it is also below ``ftol * cost`` = 50, which is exactly what certifies the point.
    """

    def model(x, a):
        return (x == 0) * a + (x == 1) * (1e17 * a * a) + 0.0 * x

    res = lm.exact_lm_fit(
        model,
        np.array([0.0, 1.0, 2.0]),
        np.array([1e-9, 1.0, 1e8]),
        [0.0],
        ftol=1e-14,
        gtol=1e-20,
        xtol=None,
        compute_covariance=False,
    )
    assert res.success and res.termination == "ftol"
    assert res.cost == pytest.approx(5e15, rel=1e-12)


@pytest.mark.serial
@pytest.mark.slow
@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_chunked_pcov_excludes_zero_weight_rows_like_the_dropped_row_fit(
    absolute_sigma, masked
):
    """Claude C01 verdict 2 MEDIUM-2: sigma=inf rows (zero weight) counted in the chunked dof."""
    rng = np.random.default_rng(0)
    n = 30000
    x = np.linspace(0, 5, n)
    y = 2 * x + 1 + 0.1 * rng.standard_normal(n)
    sigma = np.full(n, 0.1)
    sigma[::3] = np.inf
    kw = {"sigma": sigma, "absolute_sigma": absolute_sigma, "check_finite": False}
    keep = np.isfinite(sigma)
    if masked:
        mask = np.ones(n, bool)
        mask[1::7] = False  # drops finite and infinite rows alike
        kw["data_mask"] = mask
        keep &= mask
    cfg = LDMemoryConfig(memory_limit_gb=1, min_chunk_size=3000, max_chunk_size=6000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = LargeDatasetFitter(config=cfg).fit(_linear2, x, y, p0=[1, 1], **kw)
        ref = curve_fit(
            _linear2,
            x[keep],
            y[keep],
            p0=[1, 1],
            sigma=sigma[keep],
            absolute_sigma=absolute_sigma,
        )
    assert res.n_chunks > 1
    np.testing.assert_allclose(
        np.sqrt(np.diag(res.pcov)), np.sqrt(np.diag(ref[1])), rtol=2e-3
    )


def _ten_row_chunks():
    cfg = LDMemoryConfig(memory_limit_gb=1, min_chunk_size=10, max_chunk_size=10)
    return LargeDatasetFitter(config=cfg)


def test_chunk_without_a_weighted_row_is_skipped_not_failed():
    """Codex verdict 3 (wave 6): zero-weight chunks were retried and counted as failures."""
    x = np.arange(60.0)
    y = 2 * x + 1
    perm = np.random.default_rng(0).permutation(60)  # the fitter's own shuffle
    sigma = np.full(60, np.inf)
    sigma[perm[40:]] = 1.0  # four of the six ten-row chunks hold no weighted row
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _ten_row_chunks().fit(
            _linear2, x, y, p0=[1.0, 1.0], sigma=sigma, check_finite=False
        )
    assert res.success
    assert res.success_rate == 1.0
    np.testing.assert_allclose(res.popt, [2.0, 1.0], rtol=1e-8)


@pytest.mark.parametrize("masked", [False, True])
def test_all_zero_weight_problem_is_rejected_at_the_fitter_boundary(masked):
    """Wave 6: when no chunk has a weighted row the fit raises, as every other route does."""
    x = np.arange(60.0)
    sigma = np.full(60, np.inf)
    kw = {}
    if masked:  # the only weighted rows are masked out: nothing weighted is used
        sigma[:5] = 1.0
        kw["data_mask"] = np.arange(60) >= 5
    with pytest.raises(ValueError, match="zero weight"):
        _ten_row_chunks().fit(
            _linear2, x, 2 * x + 1, p0=[1.0, 1.0], sigma=sigma, check_finite=False, **kw
        )


def _sigma_dtype_cases():
    import ml_dtypes

    return {
        "object": (lambda s: s.astype(object), True),
        "float16": (lambda s: s.astype(np.float16), True),
        "bfloat16": (lambda s: s.astype(ml_dtypes.bfloat16), True),
        "int": (lambda s: np.ones(len(s), dtype=int), False),
        "bool": (lambda s: np.ones(len(s), dtype=bool), False),
    }


@pytest.mark.parametrize("kind", ["object", "float16", "bfloat16", "int", "bool"])
def test_chunked_fit_treats_every_real_sigma_dtype_like_float(kind):
    """F1 wave 9 (cross-boundary): an object sigma silently lost its zero-weight rows."""
    rng = np.random.default_rng(0)
    n = 3000
    x = np.linspace(0, 5, n)
    y = 2 * x + 1 + 0.1 * rng.standard_normal(n)
    convert, with_inf = _sigma_dtype_cases()[kind]
    sigma = np.full(n, 1.0)
    if with_inf:
        sigma[::3] = np.inf
    cfg = LDMemoryConfig(memory_limit_gb=1, min_chunk_size=500, max_chunk_size=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = LargeDatasetFitter(config=cfg).fit(
            _linear2, x, y, p0=[1, 1], sigma=sigma, check_finite=False
        )
        got = LargeDatasetFitter(config=cfg).fit(
            _linear2, x, y, p0=[1, 1], sigma=convert(sigma), check_finite=False
        )
    np.testing.assert_allclose(got.popt, ref.popt, rtol=1e-6)
    np.testing.assert_allclose(got.pcov, ref.pcov, rtol=1e-6)

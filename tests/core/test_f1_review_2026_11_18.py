"""Regression tests for the 2026-11-18 three-brain round, fixer F1.

One test per fixed finding (minpack / problem / covariance / workflow / __init__ /
common_* / device / callbacks / result).
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq import common_jax as cj
from nlsq import common_scipy as cs
from nlsq import curve_fit
from nlsq.core.covariance import pcov_from_normal
from nlsq.core.problem import PrepareContext, prepare_problem


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


def _decay_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4.0, n)
    y = 2.5 * np.exp(-1.3 * x) + 0.001 * rng.standard_normal(n)
    return x, y


# memory_limit_gb that makes the selector pick the chunked / streaming strategy
# for 2000 points (see MemoryBudgetSelector)
_CHUNKED = 1e-4
_STREAMING = 1e-6


# ----------------------------------------------------------------- minpack (C01)


@pytest.mark.slow
@pytest.mark.parametrize("limit", [_CHUNKED, _STREAMING])
def test_auto_route_warns_for_dropped_diagnostics(limit):
    """C01 M1: callback/compute_diagnostics are not silently dropped."""
    x, y = _decay_data()
    with pytest.warns(UserWarning, match="compute_diagnostics"):
        curve_fit(
            _exp,
            x,
            y,
            p0=[2.0, 1.0],
            method="auto",
            memory_limit_gb=limit,
            compute_diagnostics=True,
        )


def test_auto_chunked_resolves_p0_auto():
    """C01 M2: p0='auto' on the chunked route of method='auto'."""
    x, y = _decay_data()
    popt, _ = curve_fit(_exp, x, y, p0="auto", method="auto", memory_limit_gb=_CHUNKED)
    np.testing.assert_allclose(popt, [2.5, 1.3], rtol=1e-2)


@pytest.mark.slow
def test_auto_streaming_accepts_gtol():
    """C01 L1: gtol is not a TypeError on the streaming route of method='auto'."""
    x, y = _decay_data()
    popt, _ = curve_fit(
        _exp,
        x,
        y,
        p0=[2.0, 1.0],
        method="auto",
        memory_limit_gb=_STREAMING,
        gtol=1e-10,
    )
    np.testing.assert_allclose(popt, [2.5, 1.3], rtol=1e-2)


def test_auto_chunked_applies_sigma():
    """C01 L2: the chunked route weights by sigma instead of raising."""
    x, y = _decay_data()
    sigma = np.where(x < 2.0, 0.01, 1.0)
    res = curve_fit(
        _exp,
        x,
        y,
        p0=[2.0, 1.0],
        sigma=sigma,
        method="auto",
        memory_limit_gb=_CHUNKED,
    )
    np.testing.assert_allclose(res.popt, [2.5, 1.3], rtol=1e-2)
    np.testing.assert_array_equal(res["sigma"], sigma)


@pytest.mark.slow
def test_auto_chunked_applies_data_mask():
    """C01 L2: the chunked route excludes masked rows (they hold garbage here)."""
    x, y = _decay_data()
    mask = x < 3.0
    y = np.where(mask, y, 1e3)
    res = curve_fit(
        _exp,
        x,
        y,
        p0=[2.0, 1.0],
        data_mask=mask,
        method="auto",
        memory_limit_gb=_CHUNKED,
    )
    np.testing.assert_allclose(res.popt, [2.5, 1.3], rtol=2e-2)
    assert len(res["ydata"]) == int(mask.sum())


@pytest.mark.slow
def test_global_config_forwards_check_finite():
    """C01 L3 / codex P01 #5: fit(workflow=GlobalOptimizationConfig) honours check_finite."""
    from nlsq.global_optimization import GlobalOptimizationConfig

    def shape_only(x, a):
        return jnp.full(x.shape, a)

    x = np.array([np.nan, 2.0])
    y = np.array([2.0, 4.0])
    res = nlsq.fit(
        shape_only,
        x,
        y,
        p0=[1.0],
        bounds=(0.0, 4.0),
        check_finite=False,
        workflow=GlobalOptimizationConfig(n_starts=1),
    )
    np.testing.assert_allclose(res.popt, [3.0], atol=1e-6)


# ---------------------------------------------------------------- minpack (codex)


@pytest.mark.slow
def test_float32_sigma_is_widened():
    """codex P01 #3: 1/sigma is computed in float64 for a float32 sigma."""

    def const(x, a):
        return jnp.full(x.shape, a)

    x = np.array([1.0, 2.0])
    y = np.array([2.0, 3.0])
    sigma = np.full(2, 1e-39, dtype=np.float32)  # 1/sigma overflows in float32
    popt, pcov = curve_fit(const, x, y, p0=[2.0], sigma=sigma, absolute_sigma=True)
    np.testing.assert_allclose(popt, [2.5], atol=1e-6)
    assert np.isfinite(pcov).all()


def test_return_eval_scalar_model():
    """codex P01 #4: return_eval with a model that returns a scalar."""

    def const(x, a):
        return a

    popt, _pcov, feval = curve_fit(
        const,
        np.array([1.0]),
        np.array([2.0]),
        p0=[1.0],
        absolute_sigma=True,
        return_eval=True,
    )
    assert feval.shape == (1,)
    np.testing.assert_allclose(feval, popt)


def test_empty_method_name_is_rejected():
    """codex P01 low 2: method='' is an unknown method, not the default."""
    with pytest.raises(ValueError):
        curve_fit(
            lambda x, a: a * x,
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            p0=[1.0],
            method="",
        )


def test_max_nfev_integer_is_exact():
    """codex P01 low 1: a huge int max_nfev is neither rounded nor an OverflowError."""
    from nlsq.core._option_checks import validate_max_nfev

    assert validate_max_nfev(2**53 + 1, allow_fractional=True) == 2**53 + 1
    assert validate_max_nfev(10**400, allow_fractional=True) == 10**400
    assert validate_max_nfev(5.5, allow_fractional=True) == 5.5


@pytest.mark.slow
def test_overflow_check_ignores_padded_rows():
    """agy P01 #1: f(EPS) at padded rows must not trigger the clipping warning."""
    from nlsq.core.minpack import CurveFit

    def inv(x, a):
        return a / x

    x = jnp.linspace(1.0, 10.0, 50)
    y = inv(x, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", nlsq.OptimizeWarning)
        CurveFit(flength=100, enable_overflow_check=True).curve_fit(inv, x, y, p0=[1.0])


@pytest.mark.slow
def test_multistart_honours_flength():
    """agy P01 #2: flength is a CurveFit option, not an 'unknown keyword'."""
    x, y = _decay_data(20)
    y = 2.5 * np.exp(-1.3 * x)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        curve_fit(_exp, x, y, p0=[1.0, 1.0], multistart=True, n_starts=2, flength=50)
    assert not [i for i in w if "flength" in str(i.message)]


@pytest.mark.slow
def test_auto_global_cmaes_without_refinement_honours_compute_covariance():
    """agy P01 #4: no inf placeholder when compute_covariance=False and no refinement."""
    pytest.importorskip("evosax")
    from nlsq.global_optimization import CMAESConfig

    x = np.linspace(0, 4, 200)
    y = 2.5 * np.exp(-1.3 * x)
    res = nlsq.fit(
        _exp,
        x,
        y,
        p0=[2.0, 1.0],
        bounds=([0, 0], [5, 5]),
        workflow="auto_global",
        method="cmaes",
        cmaes_config=CMAESConfig(refine_with_nlsq=False),
        compute_covariance=False,
    )
    assert res.pcov is None


def test_curve_fit_method_docstring_is_the_full_one():
    """agy P01 #5: the parameter docs are the docstring, not a dead string literal."""
    from nlsq.core.minpack import CurveFit

    doc = CurveFit.curve_fit.__doc__
    assert "xdata : array_like" in doc and "Raises" in doc
    assert "compute_covariance" in doc


# ------------------------------------------------------------------ problem (codex)


def test_omit_mask_must_have_original_length():
    """codex P01 #1: a data_mask the length of the omit-shortened data is rejected."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, np.nan, 12.0])
    with pytest.raises(ValueError, match="Data mask"):
        curve_fit(
            lambda x, a: a * x,
            x,
            y,
            p0=[1.0],
            nan_policy="omit",
            data_mask=[True, False],
        )
    popt, _ = curve_fit(
        lambda x, a: a * x,
        x,
        y,
        p0=[1.0],
        nan_policy="omit",
        data_mask=[True, True, False],
    )
    np.testing.assert_allclose(popt, [2.0], atol=1e-6)


def test_all_false_data_mask_is_rejected():
    """codex P01 #2: a mask that drops every row is an error, not a 'fit'."""
    with pytest.raises(ValueError, match="excludes every"):
        curve_fit(
            lambda x, a: a * x,
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            p0=[1.0],
            data_mask=[False, False],
        )


def test_prepare_problem_accepts_scipy_bounds_with_stability():
    """agy P01 #3: scipy Bounds reaches algorithm auto-selection as (lb, ub)."""
    from scipy.optimize import Bounds

    x = np.linspace(0.0, 1.0, 20)
    prepared = prepare_problem(
        lambda x, a, b: a + b * x,
        x,
        2.0 + 3.0 * x,
        p0=np.array([0.5, 0.5]),
        bounds=Bounds([0.0, 0.0], [10.0, 10.0]),
        ctx=PrepareContext(enable_stability=True),
    )
    assert prepared.method in ("trf", "dogbox", "lm")


# --------------------------------------------------------------- covariance (codex)


@pytest.mark.filterwarnings(
    "ignore:normal-equation Hessian is ill-conditioned:UserWarning"
)
def test_absolute_sigma_covariance_does_not_overflow_intermediate():
    """codex P01 #6: diag(2e-150, 2e160) normal matrix, representable inverse."""
    H = np.diag([2e-150, 2e160])
    pcov, fired = pcov_from_normal(H, 0.0, 4, 2, absolute_sigma=True)
    assert not fired
    np.testing.assert_allclose(np.diag(pcov), [5e149, 5e-161], rtol=1e-12)


# ----------------------------------------------------------------- workflow (C03/codex)


def test_checkpoint_directories_in_the_same_second_differ(tmp_path):
    """C03 L1 / codex P03 low 4."""
    from nlsq.core.workflow import create_checkpoint_directory

    a = create_checkpoint_directory(tmp_path)
    b = create_checkpoint_directory(tmp_path)
    assert a != b
    assert len({a, b}) == 2


def test_selector_validates_counts_with_a_supplied_budget():
    """codex P03 #5."""
    from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector

    with pytest.raises(ValueError):
        MemoryBudgetSelector().select(
            0, 0, budget=MemoryBudget(1.0, 1.0, 0.0, 0.0, 0.0)
        )


# ------------------------------------------------------------------ __init__ (C10/codex)


def test_fit_forwards_chunk_factory_to_explicit_strategy():
    """C10 #1 / codex P03 #2: a chunk factory is not turned into a 0-D array."""
    from nlsq.streaming import Exact

    def factory(chunk):
        yield np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])

    popt, _ = nlsq.fit(
        lambda x, a: a * x, factory, None, p0=[1.0], strategy=Exact(chunk_size=4)
    )
    np.testing.assert_allclose(popt, [2.0], rtol=1e-8)


def test_fit_does_not_materialize_sliceable_sources():
    """C10 #1: an array-like with shape/len/slicing reaches curve_fit_large unread."""

    class Source:
        shape = (3,)

        def __init__(self, a):
            self._a = a
            self.reads = 0

        def __len__(self):
            return 3

        def __getitem__(self, k):
            self.reads += 1
            return self._a[k]

        def __array__(self, dtype=None, copy=None):
            raise AssertionError("the whole source was converted")

    seen = {}

    def spy(f, xdata, ydata, **kw):
        seen["x"], seen["y"] = xdata, ydata
        return "ok"

    sx, sy = Source(np.arange(3.0)), Source(2.0 * np.arange(3.0))
    with mock.patch.object(nlsq, "curve_fit_large", spy):
        assert nlsq.fit(lambda x, a: a * x, sx, sy, p0=[1.0], size_threshold=2) == "ok"
    assert seen["x"] is sx and seen["y"] is sy


def test_fit_method_auto_forwards_memory_limit_below_threshold():
    """codex P03 #3."""
    seen = {}

    def spy(f, x, y, **kw):
        seen.update(kw)
        return "ok"

    with mock.patch.object(nlsq, "curve_fit", spy):
        nlsq.fit(
            lambda x, a: a * x,
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            p0=[1.0],
            method="auto",
            memory_limit_gb=0.01,
        )
    assert seen["memory_limit_gb"] == 0.01


@pytest.mark.slow
@pytest.mark.serial
def test_debug_import_survives_missing_stderr():
    """codex P03 #6: NLSQ_DEBUG=1 with sys.stderr is None must not abort the import."""
    code = (
        "import sys\n"
        "sys.stderr = None\n"
        "try:\n"
        "    import nlsq\n"
        "    print('imported')\n"
        "except BaseException as e:\n"
        "    print('FAILED', type(e).__name__, e)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "NLSQ_DEBUG": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.stdout.strip() == "imported", r.stdout[-500:]


# ----------------------------------------------------------- common_scipy / common_jax


def test_intersect_trust_region_tangent_line():
    """codex P03 low 2: tangent at the boundary is a double root, not nan."""
    t = cs.intersect_trust_region(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0)
    assert t == (0.0, 0.0)


@pytest.mark.parametrize("impl", ["numpy", "jax"])
def test_intersect_trust_region_large_magnitudes(impl):
    """codex P03 #6: a * c overflowed near 1e100."""
    if impl == "numpy":
        lo, hi = cs.intersect_trust_region(np.array([0.0]), np.array([1e100]), 1e100)
    else:
        lo, hi = cj.intersect_trust_region_jax(
            jnp.array([0.0]), jnp.array([1e100]), 1e100
        )
    np.testing.assert_allclose([lo, hi], [-1.0, 1.0], rtol=1e-12)


def test_step_size_to_bound_reports_no_hit_without_a_bound():
    """codex P03 low 3."""
    x, s = np.array([0.0]), np.array([1.0])
    lb, ub = np.array([-np.inf]), np.array([np.inf])
    step, hits = cs.step_size_to_bound(x, s, lb, ub)
    assert np.isinf(step) and hits.tolist() == [0]
    step, hits = cj.step_size_to_bound_jax(
        jnp.array(x), jnp.array(s), jnp.array(lb), jnp.array(ub)
    )
    assert np.isinf(step) and np.asarray(hits).tolist() == [0]


def test_solve_trust_region_2d_boundary_minimizer_at_infinity():
    """codex P03 #7: the minimizer (0, -Delta) is the t -> inf point."""
    p, newton = cs.solve_trust_region_2d(np.eye(2), np.array([0.0, 2.0]), 1.0)
    np.testing.assert_allclose(p, [0.0, -1.0], atol=1e-12)
    assert not newton


def test_solve_lsq_trust_region_rank_deficient_gauss_newton():
    """codex P03 #8: the feasible minimum-norm Gauss-Newton step is taken."""
    p, alpha, n_iter = cs.solve_lsq_trust_region(
        2, 2, np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.eye(2), 10.0
    )
    np.testing.assert_allclose(p, [-1.0, 0.0], atol=1e-12)
    assert alpha == 0.0 and n_iter == 0


@pytest.mark.parametrize("radius", [1.0, jnp.array(1.0, dtype=jnp.float64)])
def test_solve_lsq_trust_region_jax_float32_under_x64(radius):
    """codex P03 low 1 + wave 3: float32 SVD factors compile when x64 is enabled,
    for a weak Python radius and for a float64 JAX scalar radius."""
    assert jax.config.jax_enable_x64
    f32 = jnp.float32
    p, alpha, _ = cj.solve_lsq_trust_region_jax(
        1, 1, 3 * jnp.ones(1, f32), jnp.ones(1, f32), jnp.eye(1, dtype=f32), radius
    )
    assert p.dtype == f32 and alpha.dtype == f32
    np.testing.assert_allclose(np.asarray(p), [-1.0], rtol=1e-6)


# ------------------------------------------------------------------ device / callbacks


def test_gpu_availability_without_nvidia_hardware_never_queries_jax_devices():
    """The check runs on `import nlsq`: without nvidia-smi it must return before
    jax.devices(), or every CPU-only import would initialize the JAX backend."""
    from nlsq import device

    with (
        mock.patch.object(device, "_env_flag", return_value=False),
        mock.patch.object(device.sys, "platform", "linux"),
        mock.patch.object(device, "get_gpu_info", return_value=(None, None)),
        mock.patch.object(device, "get_system_cuda_version", return_value=(None, None)),
        mock.patch("jax.devices", side_effect=AssertionError("backend init")),
    ):
        assert device.check_gpu_availability(warn=False) is False


def test_callback_chain_closes_every_callback_despite_a_failure():
    """codex P03 low 8."""
    from nlsq.callbacks import CallbackChain

    class Bad:
        def close(self):
            raise OSError("flush failure")

    class Good:
        closed = False

        def close(self):
            self.closed = True

    good = Good()
    chain = CallbackChain(Bad(), good)
    with pytest.raises(OSError, match="flush failure"):
        chain.close()
    assert good.closed
    chain.callbacks = []  # keep __del__ quiet


# ---------------------------------------------------------------------------- result


@pytest.mark.slow
def test_result_methods_broadcast_a_scalar_model_output():
    """codex P03 #4: predictions / bands / intervals cover every observation."""

    def const(x, a):
        return a

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    res = curve_fit(const, x, y, p0=[1.0])
    assert res.predictions.shape == (3,)
    lo, hi = res.confidence_band()
    assert lo.shape == (3,) and hi.shape == (3,)
    assert res.prediction_interval().shape == (3, 2)


def test_prediction_interval_points_first_multifeature():
    """codex P03 #5: noise term sized by the points axis, not the feature count."""
    rng = np.random.default_rng(1)
    X = rng.uniform(1.0, 2.0, size=(10, 2))
    y = 3.0 * X[:, 0] + 0.01 * rng.standard_normal(10)
    # build the result directly in the hybrid path's points-first layout
    from nlsq.result import CurveFitResult

    res = CurveFitResult(
        {
            "x": np.array([3.0]),
            "popt": np.array([3.0]),
            "pcov": np.array([[1e-4]]),
            "model": lambda x, a: a * x[:, 0],
            "xdata": X,
            "ydata": y,
            "fun": 3.0 * X[:, 0] - y,
            "sigma": np.full(10, 0.1),
            "absolute_sigma": True,
            "success": True,
            "cost": 0.0,
        }
    )
    out = res.prediction_interval(np.ones((4, 2)))
    assert out.shape == (4, 2) and np.isfinite(out).all()


# ------------------------------------------------------------------------- agy P03


def test_cluster_detect_falls_back_when_pbs_nodefile_is_unusable(monkeypatch):
    """agy P03 #1: a broken PBS_NODEFILE does not skip local GPU detection."""
    from nlsq.core.workflow import ClusterDetector

    monkeypatch.setenv("PBS_NODEFILE", "/nonexistent/path")
    local = mock.Mock(total_gpus=2)
    with mock.patch.object(ClusterDetector, "detect_local_gpus", return_value=local):
        assert ClusterDetector().detect() is local


def test_rocm_device_is_an_accelerator():
    """agy P03 #2."""
    from nlsq.device import _is_accelerator_device

    assert _is_accelerator_device(mock.Mock(platform="rocm"))
    assert _is_accelerator_device(mock.Mock(platform="cuda"))
    assert not _is_accelerator_device(mock.Mock(platform="cpu"))


def test_termination_ftol_and_xtol_is_exported():
    """agy P03 #3 (the STATUS_DESCRIPTIONS part is stale: no such table exists)."""
    from nlsq import constants

    assert "TERMINATION_FTOL_AND_XTOL" in constants.__all__


def test_curve_fit_result_is_importable_from_the_top_level():
    """agy P03 #4."""
    from nlsq import CurveFitResult
    from nlsq.result import CurveFitResult as Original

    assert CurveFitResult is Original and "CurveFitResult" in nlsq.__all__


# ----------------------------------------------------------------------- wave 3


@pytest.mark.slow
def test_auto_chunked_return_eval_covers_all_caller_rows_with_data_mask():
    """wave 3 (1): return_eval on the chunked route has one value per caller row."""
    x = np.linspace(1.0, 2.0, 2000)
    y = 2.0 * x
    mask = np.arange(2000) < 1500

    def f(x, a):
        return a * x

    std = curve_fit(f, x, y, p0=[1.0], data_mask=mask, return_eval=True)
    chunked = curve_fit(
        f,
        x,
        y,
        p0=[1.0],
        data_mask=mask,
        return_eval=True,
        method="auto",
        memory_limit_gb=_CHUNKED,
    )
    assert len(std[2]) == len(chunked[2]) == 2000
    np.testing.assert_allclose(chunked[2], std[2], rtol=1e-6)


@pytest.mark.slow
def test_auto_streaming_gtol_zero_means_disabled():
    """wave 3 (2): gtol=0 works on the streaming route as on the standard one."""
    x = np.linspace(1.0, 2.0, 2000)
    y = 2.0 * x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(
            lambda x, a: a * x,
            x,
            y,
            p0=[1.0],
            gtol=0,
            method="auto",
            memory_limit_gb=_STREAMING,
        )
    np.testing.assert_allclose(popt, [2.0], rtol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("masked", "sigma_nan"),
    [(True, True), (False, True), (True, False)],
    ids=["mask+sigma-nan", "sigma-nan", "mask"],
)
def test_auto_chunked_return_eval_matches_trf_rows(masked, sigma_nan):
    """wave 5: chunked return_eval covers the same rows as the TRF route
    (nan_policy='omit' drops x/y/sigma NaN rows; data_mask does not shrink it)."""
    x = np.linspace(1.0, 5.0, 2000)
    y = 2.0 * x
    sigma = np.ones(2000)
    if sigma_nan:
        sigma[200] = np.nan
    kw = {"p0": [1.0], "sigma": sigma, "nan_policy": "omit", "return_eval": True}
    if masked:
        kw["data_mask"] = x < 3.0

    def f(x, a):
        return a * x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trf = curve_fit(f, x, y, **kw)
        chunked = curve_fit(f, x, y, method="auto", memory_limit_gb=_CHUNKED, **kw)
    assert len(chunked[2]) == len(trf[2]) == (1999 if sigma_nan else 2000)
    np.testing.assert_allclose(chunked[2], trf[2], rtol=1e-6)

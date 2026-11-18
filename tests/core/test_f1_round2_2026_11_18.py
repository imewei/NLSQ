"""Regression tests for fixer F1, three-brain round 2026-11-18 #2 (core API, results)."""

from __future__ import annotations

import os
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq import CurveFit, curve_fit, fit
from nlsq.global_optimization import CMAESConfig


def _lin(x, a):
    return a * x


# ------------------------------------------------------------------ problem.py


class _Spy(Exception):
    pass


def test_auto_p0_estimator_sees_only_rows_used_by_fit(monkeypatch):
    """codex P01 HIGH: p0='auto' ignores data_mask / omitted (NaN sigma) rows."""
    import nlsq.precision.parameter_estimation as pe

    seen = []
    real = pe.estimate_initial_parameters

    def spy(f, x, y, p0=None):
        seen.append((np.size(x), np.size(y)))
        return real(f, x, y, p0)

    monkeypatch.setattr(pe, "estimate_initial_parameters", spy)

    def g(x, a, mu, s):
        return a * jnp.exp(-((x - mu) ** 2) / (2 * s**2))

    x = np.linspace(-5, 5, 101)
    y = 3 * np.exp(-(x**2) / 2)
    x2, y2 = np.r_[x, 100, 100.1], np.r_[y, 500, 500]
    mask = np.r_[np.ones(101, bool), False, False]
    curve_fit(g, x2, y2, p0="auto", data_mask=mask)
    sigma = np.r_[np.ones(101), np.nan, np.nan]
    curve_fit(g, x2, y2, p0="auto", sigma=sigma, nan_policy="omit")
    assert seen == [(101, 101), (101, 101)]


def test_rows_used_by_fit_helper():
    from nlsq.core.problem import rows_used_by_fit

    x, y = np.arange(5.0), np.arange(5.0)
    xs, ys = rows_used_by_fit(x, y, None, [True, False, True, True, True], "raise")
    np.testing.assert_array_equal(xs, [0, 2, 3, 4])
    sig = np.array([1, 1, np.nan, 1, 1.0])
    xs, ys = rows_used_by_fit(x, y, sig, None, "omit")
    np.testing.assert_array_equal(ys, [0, 1, 3, 4])
    # misaligned mask: returned untouched, the fit's own validation reports it
    xs, _ = rows_used_by_fit(x, y, None, [True, False], "raise")
    assert xs is x


def test_complex_inputs_are_rejected_not_truncated():
    """codex P01 MEDIUM: complex sigma/x/y were cast to real with a warning."""
    x = np.arange(3.0)
    with pytest.raises(ValueError, match="real-valued"):
        curve_fit(
            lambda x, a: a + 0 * x,
            x,
            np.array([0, 10, 0.0]),
            p0=[1.0],
            sigma=np.array([1 + 100j, 1, 1]),
        )
    with pytest.raises(ValueError, match="real-valued"):
        curve_fit(_lin, np.arange(4) + 1j, 2 * np.arange(4.0), p0=[1.0])
    with pytest.raises(ValueError, match="real-valued"):
        curve_fit(_lin, np.arange(4.0), 2 * np.arange(4) + 1j, p0=[1.0])


def test_auto_p0_with_extra_keyword_arguments():
    """codex P01 LOW / P06: **kwargs adds no positional fit parameter."""

    def model(x, a, b, **kw):
        return a * x + b

    x = np.arange(5.0)
    res = curve_fit(model, x, 3 * x + 4, p0="auto")
    np.testing.assert_allclose(res.popt, [3.0, 4.0], rtol=1e-6)


def test_auto_p0_still_rejects_var_positional():
    def model(x, *p):
        return p[0] * x

    with pytest.raises(ValueError, match=r"\*args"):
        curve_fit(model, np.arange(5.0), np.arange(5.0), p0="auto")


# ------------------------------------------------------------------ _option_checks


def test_complex_tolerance_is_rejected():
    """codex P01 LOW: a complex gtol was projected onto the real axis."""
    from nlsq import LeastSquares

    with pytest.raises(ValueError, match="real"):
        LeastSquares().least_squares(
            lambda p: p - 1.0, np.array([0.0]), gtol=np.complex128(1e-8 + 2j)
        )
    with pytest.raises(ValueError, match="real"):
        LeastSquares().least_squares(
            lambda p: p - 1.0, np.array([0.0]), gtol=np.array(1e-8 + 2j)
        )


# ------------------------------------------------------------------ minpack


def test_maxfev_alias_validated_on_cmaes_route():
    """agy P01 #1: an invalid maxfev was swallowed by the CMA-ES refinement."""
    x = np.linspace(0, 1, 10)
    for bad in ("invalid", True, -1):
        with pytest.raises(ValueError, match="max_nfev"):
            curve_fit(
                _lin, x, 2 * x, p0=[1.0], bounds=(0, 5), method="cmaes", maxfev=bad
            )


@pytest.mark.parametrize("sigma", [0.0, np.nan, np.inf, [2.0]])
def test_cmaes_rejects_bad_scalar_and_misaligned_sigma(sigma):
    """codex P01 MEDIUM: scalar 0/NaN/inf and a length-1 sigma entered CMA-ES."""
    x = np.linspace(0, 1, 8)
    cfg = CMAESConfig(max_generations=2, refine_with_nlsq=False)
    with pytest.raises(ValueError):
        curve_fit(
            _lin,
            x,
            2 * x,
            p0=[1.0],
            bounds=(0, 4),
            sigma=sigma,
            method="cmaes",
            cmaes_config=cfg,
        )


def test_cmaes_return_eval_broadcasts_scalar_model():
    """codex P01 MEDIUM + claude C01 LOW-2: scalar output, CMA-ES route."""
    x = np.arange(6.0)
    y = np.array([1, 2, 3, 2, 1, 3.0])
    cfg = CMAESConfig(max_generations=2, refine_with_nlsq=False)
    _, _, feval = curve_fit(
        lambda x, a: a,
        x,
        y,
        p0=[1.0],
        bounds=([0], [4]),
        method="cmaes",
        cmaes_config=cfg,
        return_eval=True,
    )
    assert feval.shape == (6,)


def test_global_cmaes_workflow_formats_output_flags():
    """codex P01 MEDIUM: fit(workflow='auto_global', method='cmaes', return_eval)."""
    x = np.linspace(0, 1, 8)
    cfg = CMAESConfig(max_generations=2, refine_with_nlsq=False)
    out = fit(
        _lin,
        x,
        2 * x,
        p0=[1.0],
        bounds=([0.0], [4.0]),
        workflow="auto_global",
        method="cmaes",
        cmaes_config=cfg,
        return_eval=True,
    )
    assert isinstance(out, tuple) and len(out) == 3
    assert out[2].shape == (8,)


def test_multistart_with_fallback_runs_the_fallback_handler():
    """codex P01 MEDIUM: multistart returned before the fallback orchestrator."""
    x = np.linspace(0, 5, 40)
    res = curve_fit(
        lambda x, a, b: a * x + b,
        x,
        2 * x + 1,
        p0=[1.0, 1.0],
        bounds=([0, 0], [5, 5]),
        multistart=True,
        n_starts=2,
        fallback=True,
    )
    assert type(res).__name__ == "FallbackResult"
    np.testing.assert_allclose(res.popt, [2.0, 1.0], rtol=1e-5)


def test_auto_memory_routing_counts_predictor_features(monkeypatch):
    """codex P01 MEDIUM: the selector used its default of one feature."""
    from nlsq.core.workflow import MemoryBudgetSelector

    seen = {}

    def spy(self, **kw):
        seen.update(kw)
        raise _Spy

    monkeypatch.setattr(MemoryBudgetSelector, "select", spy)
    x = np.ones((128, 2000))
    with pytest.raises(_Spy):
        curve_fit(
            lambda x, a: a * x[0],
            x,
            np.ones(2000),
            p0=[1.0],
            method="auto",
            memory_limit_gb=0.001,
        )
    assert seen["n_features"] == 128


def test_chunked_full_output_whitens_dense_covariance():
    """codex P01 MEDIUM: fvec was residual / covariance matrix, shape (n, n)."""
    x = np.arange(6.0)
    y = 2 * x + np.sin(x)
    i = np.arange(6)
    sigma = 0.5 ** np.abs(i[:, None] - i[None, :])
    out = curve_fit(
        _lin,
        x,
        y,
        p0=[1.0],
        sigma=sigma,
        method="auto",
        memory_limit_gb=8e-7,
        full_output=True,
    )
    ref = curve_fit(_lin, x, y, p0=[1.0], sigma=sigma, full_output=True)
    assert out[2]["fvec"].shape == (6,)
    np.testing.assert_allclose(out[2]["fvec"], ref[2]["fvec"], atol=1e-3)


def test_chunked_does_not_validate_covariance_rows_masked_out():
    """codex P01 MEDIUM: unused x_eval preparation rejected NaN masked rows."""
    x = np.arange(6.0)
    sigma = np.eye(6)
    sigma[0, :] = np.nan
    sigma[:, 0] = np.nan
    res = curve_fit(
        _lin,
        x,
        2 * x,
        p0=[1.0],
        sigma=sigma,
        data_mask=x > 0,
        nan_policy="omit",
        method="auto",
        memory_limit_gb=8e-7,
    )
    np.testing.assert_allclose(res.popt, [2.0], rtol=1e-6)


def test_square_predictor_masks_with_auto_bounds_and_ld_workflow():
    """codex P01 MEDIUM: a (3, 3) predictor array was 'ambiguous' for masking."""
    from nlsq.streaming.large_dataset import LDMemoryConfig

    x = np.arange(9.0).reshape(3, 3) + 1
    y = 2 * x[0]
    mask = np.array([True, False, True])

    def model(x, a):
        return a * x[0]

    r1 = curve_fit(model, x, y, p0=[1.0], data_mask=mask, auto_bounds=True)
    r2 = fit(model, x, y, p0=[1.0], data_mask=mask, workflow=LDMemoryConfig())
    for r in (r1, r2):
        np.testing.assert_allclose(r.popt, [2.0], rtol=1e-6)


def test_hybrid_workflow_rejects_options_it_cannot_honour():
    """codex P01 MEDIUM: compute_covariance=False / full_output were ignored."""
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    x = np.linspace(0, 1, 50)
    for kw in ({"compute_covariance": False}, {"full_output": True}):
        with pytest.raises(ValueError, match="hybrid"):
            fit(_lin, x, 2 * x, p0=[1.0], workflow=HybridStreamingConfig(), **kw)


@pytest.mark.slow
def test_hybrid_workflow_survives_gtol_zero_and_none():
    """agy P01 #3: gtol=0/None crashed on gauss_newton_tol validation."""
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    x = np.linspace(0, 1, 200)
    y = 2 * x + 1
    for gtol in (0, None):
        res = fit(
            lambda x, a, b: a * x + b,
            x[:, None].T,
            y,
            p0=[1.0, 1.0],
            workflow=HybridStreamingConfig(),
            gtol=gtol,
        )
        np.testing.assert_allclose(res.popt, [2.0, 1.0], rtol=1e-3)


@pytest.mark.slow
def test_auto_streaming_route_gtol_zero_still_converges():
    """claude C01 LOW-1: gtol=0 left the hybrid engine no stopping test."""
    x = np.linspace(0, 4, 2000)
    y = 3 * np.exp(-0.8 * x) + 0.5

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    with pytest.warns(UserWarning, match="gtol"):
        res = curve_fit(
            model, x, y, p0=[2.0, 1.0, 0.0], method="auto", memory_limit_gb=1e-6, gtol=0
        )
    assert res.success
    np.testing.assert_allclose(res.popt, [3.0, 0.8, 0.5], rtol=1e-3)


def test_multistart_streaming_default_p0_is_inside_the_bounds(monkeypatch):
    """agy P01 #4: p0=ones lay outside the (finite) bounds."""
    import nlsq.streaming.strategy as strategy_mod
    from nlsq.core.minpack import _fit_global_multistart

    seen = {}

    class FakeStrategy:
        def fit(self, f, x, y, p0, **kw):
            seen["p0"] = np.asarray(p0)
            raise _Spy

    monkeypatch.setattr(
        strategy_mod, "build_large_strategy", lambda *a, **k: FakeStrategy()
    )
    x = np.linspace(0, 1, 20)
    with pytest.raises(_Spy):
        _fit_global_multistart(
            lambda x, a, b: a * x + b,
            x,
            2 * x,
            None,
            None,
            False,
            True,
            ([10.0, 10.0], [20.0, 20.0]),
            "streaming",
            2,
        )
    assert np.all(seen["p0"] >= 10.0) and np.all(seen["p0"] <= 20.0)


def test_auto_cmaes_forwards_memory_limit_to_budget(monkeypatch):
    """codex P03 MEDIUM: the auto CMA-ES budget ignored memory_limit_gb."""
    from nlsq.core.workflow import MemoryBudget

    seen = {}
    real = MemoryBudget.compute.__func__

    def spy(cls, *a, **kw):
        seen.update(kw)
        raise _Spy

    monkeypatch.setattr(MemoryBudget, "compute", classmethod(spy))
    x = np.linspace(0, 1, 50)
    with pytest.raises(_Spy):
        curve_fit(
            _lin,
            x,
            2 * x,
            p0=[1.0],
            bounds=([0.0], [1e6]),
            method="auto",
            memory_limit_gb=0.125,
        )
    assert seen["memory_limit_gb"] == 0.125
    assert real is not None


def test_curvefit_enable_stability_reaches_the_inner_solver():
    """codex P01 MEDIUM: CurveFit(enable_stability=True) built an unguarded solver."""
    x = np.arange(6.0)

    def jac(x, a):
        j = jnp.asarray(x).reshape(-1, 1) * jnp.ones_like(a)
        return jnp.where((jnp.arange(len(x)) == 0)[:, None] & (a == 1.0), jnp.nan, j)

    assert CurveFit(enable_stability=True).ls.enable_stability
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CurveFit(enable_stability=True).curve_fit(
            _lin, x, 2 * x, p0=[1.0], jac=jac
        )
    np.testing.assert_allclose(res.popt, [2.0], rtol=1e-6)


def test_explicit_hybrid_method_warns_on_compute_diagnostics():
    """codex P01 LOW: compute_diagnostics was consumed without a word."""
    x = np.linspace(0, 1, 50)
    with pytest.warns(UserWarning, match="compute_diagnostics"):
        curve_fit(
            _lin,
            x,
            2 * x,
            p0=[1.0],
            method="hybrid_streaming",
            compute_diagnostics=True,
        )


def test_curvefit_p0_docstring_distinguishes_none_from_auto():
    """codex P01 LOW / claude C01 LOW-3: p0=None does not inspect the data."""
    doc = CurveFit.curve_fit.__doc__
    assert "If 'auto', initial" in doc and "If None (the SciPy default)" in doc
    assert "from jaxfit" not in doc


def test_fit_with_chunk_factory_defaults_to_exact_strategy():
    """claude C03 LOW-1: fit(factory) without strategy= could only fail."""

    def factory(chunk):
        yield np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])

    popt, _ = nlsq.fit(lambda x, a: a * x, factory, None, p0=[1.0])
    np.testing.assert_allclose(popt, [2.0], rtol=1e-8)


# ------------------------------------------------------------------ config / device


def test_configure_for_large_datasets_is_scoped_to_memory_context():
    """codex P03 MEDIUM: process-wide large-dataset settings escaped a memory_context."""
    from nlsq.config import (
        JAXConfig,
        LargeDatasetConfig,
        MemoryConfig,
        configure_for_large_datasets,
        get_large_dataset_config,
        memory_context,
    )

    before = JAXConfig.get_large_dataset_config()
    try:
        custom = LargeDatasetConfig(enable_automatic_solver_selection=False)
        JAXConfig.set_large_dataset_config(custom)
        with memory_context(MemoryConfig(memory_limit_gb=4.0)):
            configure_for_large_datasets(2.0)
            assert get_large_dataset_config().enable_automatic_solver_selection
        assert not get_large_dataset_config().enable_automatic_solver_selection
    finally:
        JAXConfig.set_large_dataset_config(before)


def test_configure_for_large_datasets_keeps_other_large_dataset_fields():
    from nlsq.config import (
        JAXConfig,
        LargeDatasetConfig,
        configure_for_large_datasets,
        get_large_dataset_config,
    )

    before = JAXConfig.get_large_dataset_config()
    try:
        custom = LargeDatasetConfig(
            enable_automatic_solver_selection=False,
            solver_selection_thresholds={"direct": 1, "iterative": 2, "chunked": 3},
        )
        JAXConfig.set_large_dataset_config(custom)
        configure_for_large_datasets(2.0)
        got = get_large_dataset_config()
        assert got.enable_automatic_solver_selection
        assert got.solver_selection_thresholds == {
            "direct": 1,
            "iterative": 2,
            "chunked": 3,
        }
    finally:
        JAXConfig.set_large_dataset_config(before)
        JAXConfig.set_memory_config(JAXConfig.get_memory_config())


def test_gpu_advisory_goes_to_stderr_and_names_a_compatible_package(capsys):
    """codex P03 MEDIUM + LOW: stdout stays clean; no cuda11 package suggested."""
    from nlsq.device import _print_gpu_warning

    _print_gpu_warning("RTX 4090", 8.9, "11.8", 11)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "GPU AVAILABLE BUT NOT USED" in captured.err
    assert "cuda11" not in captured.err.replace("jax-cuda", "")
    _print_gpu_warning("RTX 4090", 8.9, "12.6", 12)
    assert 'pip install "jax[cuda12-local]"' in capsys.readouterr().err


def test_workflow_budgets_a_single_gpu_device(monkeypatch):
    """codex P03 MEDIUM: free memory of all GPUs was pooled for a one-device solver."""
    import nlsq.streaming.large_dataset as ld
    from nlsq.core.workflow import MemoryBudget

    class Dev:
        platform = "gpu"

        def memory_stats(self):
            return {"bytes_limit": 8 * 1024**3, "bytes_in_use": 0}

    monkeypatch.setattr(ld, "get_cached_devices", lambda: [Dev(), Dev(), Dev(), Dev()])
    assert MemoryBudget._detect_available_memory(use_gpu=True) == pytest.approx(8.0)


def test_create_checkpoint_directory_returns_absolute_path(tmp_path, monkeypatch):
    """codex P03 LOW / agy P03 #2."""
    from nlsq.core.workflow import create_checkpoint_directory

    monkeypatch.chdir(tmp_path)
    path = create_checkpoint_directory("relative")
    assert os.path.isabs(path) and os.path.isdir(path)


def test_memory_budget_selector_logs_under_module_logger(caplog):
    """agy P03 #4: select() used a local 'nlsq' logger shadowing the module one."""
    from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector

    budget = MemoryBudget(1.0, 0.5, 0.1, 0.1, 0.1, fixed_gb=9.0)
    with caplog.at_level("WARNING"):
        MemoryBudgetSelector().select(10, 10, budget=budget)
    assert any(r.name == "nlsq.core.workflow" for r in caplog.records)


# ------------------------------------------------------------------ callbacks / types


def test_callback_with_positional_only_and_keyword_only_arguments():
    """codex P03 MEDIUM: neither all-keyword nor all-positional binds."""
    from nlsq.callbacks import call_user_callback

    def cb(iteration, /, *, cost, params, info):
        return (iteration, cost, params, info)

    assert call_user_callback(cb, 1, 2.0, "p", {"i": 1}) == (1, 2.0, "p", {"i": 1})


def test_iteration_logger_closes_owned_file_when_footer_write_fails(tmp_path):
    """codex P03 LOW: a failed footer write leaked the logger's own file."""
    from nlsq.callbacks import IterationLogger

    logger = IterationLogger(filename=str(tmp_path / "log.txt"))
    logger._open_file()
    logger._start_time = 1.0
    handle = logger._file

    def boom(msg):
        raise OSError("disk full")

    logger._write = boom  # type: ignore[method-assign]
    with pytest.raises(OSError, match="disk full"):
        logger.close()
    assert handle.closed and logger._file is None


def test_bounds_alias_accepts_mixed_scalar_and_vector():
    """codex P03 LOW: BoundsTuple rejected (0, [2, 2])."""
    import typing

    from nlsq.types import BoundsTuple

    lower, upper = typing.get_args(BoundsTuple)
    assert float in typing.get_args(lower) and float in typing.get_args(upper)


# ------------------------------------------------------------------ result


def test_prediction_interval_keeps_points_first_layout_for_square_hybrid_data():
    """codex P03 MEDIUM: a square (n, n) training array flipped the layout."""
    x = np.array([[1.0, 2.0], [2.0, 4.0]])
    y = np.array([2.1, 4.0])
    res = curve_fit(lambda x, a: a, x, y, p0=[1.0], method="hybrid_streaming")
    q = np.zeros((3, 2))
    assert res.prediction_interval(q).shape == (3, 2)
    assert res.confidence_band(q)[0].shape == (3,)


def test_prediction_interval_ignores_zero_weight_rows_for_noise():
    """codex P03 + verdict1: sigma=inf rows are neither noise nor degrees of freedom."""
    f = lambda x, a: a + 0 * x
    x = np.arange(4.0)
    y = np.array([1, 2, 1, 100.0])
    res = curve_fit(
        f, x, y, p0=[1.0], sigma=np.array([1, 1, 1, np.inf]), check_finite=False
    )
    ref = curve_fit(f, x[:3], y[:3], p0=[1.0], sigma=np.ones(3))
    np.testing.assert_allclose(res.popt, ref.popt)
    np.testing.assert_allclose(res.pcov, ref.pcov, rtol=1e-10)
    q = np.array([1.0])
    np.testing.assert_allclose(
        res.prediction_interval(q), ref.prediction_interval(q), rtol=1e-8
    )
    np.testing.assert_allclose(
        res.confidence_band(q)[0], ref.confidence_band(q)[0], rtol=1e-8
    )


def test_chunked_scalar_return_eval_matches_standard_route_with_mask_and_omit():
    """verdict1: the chunked route broadcast over the pre-omit mask length (6 vs 5)."""
    x = np.arange(6.0)
    y = np.array([1, 1, np.nan, 1, 1, 1.0])
    mask = np.array([False, True, True, True, True, True])
    kw = {"p0": [0.8], "data_mask": mask, "nan_policy": "omit", "return_eval": True}
    std = curve_fit(lambda x, a: a, x, y, **kw)[2]
    chunked = curve_fit(
        lambda x, a: a, x, y, method="auto", memory_limit_gb=8e-7, **kw
    )[2]
    assert std.shape == chunked.shape == (5,)


def test_chunked_route_rejects_complex_data_like_the_standard_route():
    """verdict1: the chunked route cast complex y to float (ComplexWarning)."""
    x = np.arange(6.0)
    for kw in ({}, {"method": "auto", "memory_limit_gb": 8e-7}):
        with pytest.raises(ValueError, match="real-valued"):
            curve_fit(_lin, x, 2 * x + 1j * x, p0=[1.0], **kw)


def test_curve_fit_result_dir_lists_statistics_and_methods():
    """agy P03 #1."""
    res = curve_fit(_lin, np.arange(5.0) + 1, 2 * np.arange(5.0) + 2, p0=[1.0])
    names = dir(res)
    assert {"r_squared", "summary", "popt", "x"} <= set(names)


def test_summary_without_data_reports_na(capsys):
    """agy P03 #3: summary() crashed for a result without ydata/model."""
    from nlsq.result.curve_fit_result import CurveFitResult

    CurveFitResult(x=[1.0, 2.0], success=True, message="ok", cost=0.01).summary()
    assert "N/A" in capsys.readouterr().out


class _LazySource:
    """Shape/len/slicing like an h5py dataset; a full ``np.asarray`` read is forbidden."""

    ndim = 1

    def __init__(self, a):
        self._a = np.asarray(a, float)
        self.shape, self.dtype = self._a.shape, self._a.dtype

    def __len__(self):
        return len(self._a)

    def __getitem__(self, i):
        return self._a[i]

    def __array__(self, *a, **k):
        raise RuntimeError("full read forbidden")


def test_fit_default_route_streams_a_sliceable_source_without_full_read():
    """codex P03 HIGH: fit() above size_threshold must not materialise out-of-core data."""
    x = np.arange(1, 7.0)
    popt, _ = fit(
        lambda x, a: a * x,
        _LazySource(x),
        _LazySource(2 * x),
        p0=[1.0],
        size_threshold=3,
    )[:2]
    np.testing.assert_allclose(popt, [2.0], rtol=1e-8)


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_statistics_ignore_zero_weight_rows_like_the_dropped_row_fit(absolute_sigma):
    """team-lead wave 4: R2/adjR2/RMSE/MAE/AIC/BIC/dof use the effective sample size."""

    def f(x, a, b):
        return a + b * x

    x = np.arange(5.0)
    y = np.array([1.0, 2.2, 2.9, 4.1, 500.0])
    kw = {"p0": [0.0, 1.0], "absolute_sigma": absolute_sigma}
    res = curve_fit(
        f, x, y, sigma=np.array([1, 1, 1, 1, np.inf]), check_finite=False, **kw
    )
    ref = curve_fit(f, x[:4], y[:4], sigma=np.ones(4), **kw)
    np.testing.assert_allclose(res.popt, ref.popt, rtol=1e-8)
    np.testing.assert_allclose(res.pcov, ref.pcov, rtol=1e-8)
    for name in ("r_squared", "adj_r_squared", "rmse", "mae", "aic", "bic"):
        np.testing.assert_allclose(getattr(res, name), getattr(ref, name), rtol=1e-8)
    np.testing.assert_allclose(
        res.confidence_intervals(0.9), ref.confidence_intervals(0.9), rtol=1e-8
    )


@pytest.mark.parametrize("masked", [False, True])
def test_flength_padding_keeps_the_zero_weight_dof_correction(masked):
    """claude C01 verdict2 BLOCKING-1: padded data hid sigma=inf rows from the dof."""

    def f(x, a, b):
        return a + b * x

    rng = np.random.default_rng(0)
    n = 40
    x = np.linspace(0, 1, n)
    y = 1 + 2 * x + 0.05 * rng.normal(size=n)
    sigma = np.ones(n)
    sigma[::5] = np.inf
    mask = np.ones(n, bool)
    if masked:
        mask[[1, 7]] = False
    kw = {"p0": [0.0, 1.0], "check_finite": False, "absolute_sigma": False}
    if masked:
        kw["data_mask"] = mask
    res = CurveFit(flength=64).curve_fit(f, x, y, sigma=sigma, **kw)
    keep = np.isfinite(sigma) & mask
    ref = CurveFit().curve_fit(
        f, x[keep], y[keep], sigma=np.ones(keep.sum()), p0=[0.0, 1.0]
    )
    np.testing.assert_allclose(res.popt, ref.popt, rtol=1e-6)
    np.testing.assert_allclose(res.pcov, ref.pcov, rtol=1e-6)
    np.testing.assert_allclose(res.aic, ref.aic, rtol=1e-8)
    np.testing.assert_allclose(res.r_squared, ref.r_squared, rtol=1e-6)


def test_all_zero_weight_rows_are_rejected_not_ranked_optimal():
    """codex verdict2: sigma=inf everywhere gave success, AIC=2, BIC=-inf."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="every row has zero weight"):
        curve_fit(
            _lin,
            x,
            2 * x,
            p0=[1.0],
            sigma=np.full(4, np.inf),
            check_finite=False,
            absolute_sigma=True,
        )
    # same contract as an all-False data_mask
    with pytest.raises(ValueError, match="nothing to fit"):
        curve_fit(_lin, x, 2 * x, p0=[1.0], data_mask=np.zeros(4, bool))
    # a mask that leaves only zero-weight rows is the same situation
    with pytest.raises(ValueError, match="every row has zero weight"):
        curve_fit(
            _lin,
            x,
            2 * x,
            p0=[1.0],
            sigma=np.array([np.inf, np.inf, 1.0, 1.0]),
            data_mask=np.array([True, True, False, False]),
            check_finite=False,
        )


def test_cmaes_masked_scalar_return_eval_covers_all_caller_rows_points_first():
    """codex verdict2: the formatter used the feature count of a points-first x."""
    x = np.array([[1.0, 2], [2, 4], [3, 6], [4, 8]])
    y = np.array([1.0, 2, 3, 4])
    cfg = CMAESConfig(max_generations=2, refine_with_nlsq=False)
    _, _, feval = curve_fit(
        lambda x, a: a,
        x,
        y,
        p0=[1.0],
        bounds=([0.0], [5.0]),
        method="cmaes",
        cmaes_config=cfg,
        data_mask=np.array([True, False, True, True]),
        return_eval=True,
    )
    assert feval.shape == (4,)


def _all_zero_weight_cases():
    cfg = CMAESConfig(max_generations=2, refine_with_nlsq=False)
    return {
        "cmaes": lambda f, x, y, s: curve_fit(
            f,
            x,
            y,
            p0=[1.0],
            bounds=([0.0], [5.0]),
            sigma=s,
            method="cmaes",
            cmaes_config=cfg,
            check_finite=False,
        ),
        "multistart": lambda f, x, y, s: curve_fit(
            f,
            x,
            y,
            p0=[1.0],
            sigma=s,
            multistart=True,
            n_starts=2,
            check_finite=False,
        ),
        "fallback": lambda f, x, y, s: curve_fit(
            f, x, y, p0=[1.0], sigma=s, fallback=True, check_finite=False
        ),
        "auto_chunked": lambda f, x, y, s: curve_fit(
            f,
            x,
            y,
            p0=[1.0],
            sigma=s,
            method="auto",
            memory_limit_gb=8e-7,
            check_finite=False,
        ),
        "hybrid_method": lambda f, x, y, s: curve_fit(
            f,
            x,
            y,
            p0=[1.0],
            sigma=s,
            method="hybrid_streaming",
            check_finite=False,
        ),
        "curve_fit_large": lambda f, x, y, s: nlsq.curve_fit_large(
            f, x, y, p0=[1.0], sigma=s, check_finite=False
        ),
        "fit_global_cmaes": lambda f, x, y, s: fit(
            f,
            x,
            y,
            p0=[1.0],
            bounds=([0.0], [5.0]),
            sigma=s,
            workflow="auto_global",
            method="cmaes",
            cmaes_config=cfg,
            check_finite=False,
        ),
        "fit_hybrid_workflow": lambda f, x, y, s: fit(
            f,
            x,
            y,
            p0=[1.0],
            sigma=s,
            check_finite=False,
            workflow=__import__(
                "nlsq.streaming.hybrid_config", fromlist=["HybridStreamingConfig"]
            ).HybridStreamingConfig(),
        ),
        "curvefit_class": lambda f, x, y, s: CurveFit().curve_fit(
            f, x, y, p0=[1.0], sigma=s, check_finite=False
        ),
    }


@pytest.mark.parametrize("route", list(_all_zero_weight_cases()))
def test_all_zero_weight_is_rejected_on_every_route(route):
    """team-lead wave 6: one contract for the same input on every route."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    run = _all_zero_weight_cases()[route]
    with pytest.raises(ValueError, match="every row has zero weight"):
        run(_lin, x, 2 * x, np.full(4, np.inf))
    # a data_mask that leaves only zero-weight rows is the same situation
    if route in ("cmaes", "multistart", "curvefit_class", "curve_fit_large"):
        sigma = np.array([np.inf, np.inf, 1.0, 1.0])
        mask = np.array([True, True, False, False])
        with pytest.raises(ValueError, match="every row has zero weight"):
            if route == "curve_fit_large":
                nlsq.curve_fit_large(
                    _lin, x, 2 * x, p0=[1.0], sigma=sigma, data_mask=mask
                )
            elif route == "curvefit_class":
                CurveFit().curve_fit(
                    _lin,
                    x,
                    2 * x,
                    p0=[1.0],
                    sigma=sigma,
                    data_mask=mask,
                    check_finite=False,
                )
            else:
                kw = (
                    {"bounds": ([0.0], [5.0]), "method": "cmaes"}
                    if route == "cmaes"
                    else {"multistart": True, "n_starts": 2}
                )
                curve_fit(
                    _lin,
                    x,
                    2 * x,
                    p0=[1.0],
                    sigma=sigma,
                    data_mask=mask,
                    check_finite=False,
                    **kw,
                )


@pytest.mark.parametrize(
    "sigma", [float("inf"), np.float64("inf"), np.array(np.inf)], ids=str
)
@pytest.mark.parametrize("route", ["standard", "class", "multistart", "auto_chunked"])
def test_scalar_infinite_sigma_is_rejected(route, sigma):
    """codex verdict3 (a): a scalar sigma=inf zero-weights every observation."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    kw = {
        "standard": {},
        "multistart": {"multistart": True, "n_starts": 2},
        "auto_chunked": {"method": "auto", "memory_limit_gb": 8e-7},
    }
    with pytest.raises(ValueError, match="every row has zero weight"):
        if route == "class":
            CurveFit().curve_fit(
                _lin, x, 2 * x, p0=[1.0], sigma=sigma, check_finite=False
            )
        else:
            curve_fit(
                _lin, x, 2 * x, p0=[1.0], sigma=sigma, check_finite=False, **kw[route]
            )


def test_all_infinite_jax_sigma_is_rejected_on_cmaes_and_standard_routes():
    """codex verdict3 (b): an in-memory JAX sigma vector was not inspected."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cfg = CMAESConfig(max_generations=2, refine_with_nlsq=False)
    sigma = jnp.full(4, jnp.inf)
    with pytest.raises(ValueError, match="every row has zero weight"):
        curve_fit(
            _lin,
            x,
            2 * x,
            p0=[1.0],
            bounds=([0.0], [5.0]),
            sigma=sigma,
            method="cmaes",
            cmaes_config=cfg,
            check_finite=False,
        )
    with pytest.raises(ValueError, match="every row has zero weight"):
        curve_fit(_lin, x, 2 * x, p0=[1.0], sigma=sigma, check_finite=False)


def test_reject_all_zero_weight_never_reads_out_of_core_sources():
    from nlsq.core.problem import reject_all_zero_weight

    reject_all_zero_weight(_LazySource(np.full(4, np.inf)))  # raises on __array__
    reject_all_zero_weight(None)
    reject_all_zero_weight(2.0)
    reject_all_zero_weight(np.array([np.inf, 1.0]))
    reject_all_zero_weight(float("inf"), data_mask=np.zeros(3, bool))


@pytest.mark.parametrize(
    "sigma",
    [
        jnp.full((4,), 1 + 1j),
        jnp.full((4,), complex(np.inf, 1)),
        np.full(4, complex(np.inf, 1)),
        [complex(np.inf, 1)] * 4,
        np.complex128(complex(np.inf, 1)),
        complex(np.inf, 1),
        1 + 1j,
    ],
    ids=[
        "jax",
        "jax_inf_imag",
        "numpy",
        "list",
        "np_scalar",
        "scalar",
        "scalar_finite",
    ],
)
def test_complex_sigma_reports_real_input_error_without_casting(sigma):
    """codex verdict4: the preflight cast complex sigma to float before the rejection."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a ComplexWarning would fail the test
        for kw in ({}, {"method": "auto", "memory_limit_gb": 8e-7}):
            with pytest.raises(ValueError, match="real-valued"):
                curve_fit(_lin, x, 2 * x, p0=[1.0], sigma=sigma, **kw)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
@pytest.mark.parametrize("route", ["cmaes", "standard", "auto_chunked"])
def test_low_precision_all_infinite_sigma_is_rejected(route, dtype):
    """codex verdict5: bfloat16 (ml_dtypes, dtype.kind 'V') escaped the dtype whitelist."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cfg = CMAESConfig(max_generations=2, refine_with_nlsq=False)
    kw = {
        "cmaes": {"bounds": ([0.0], [5.0]), "method": "cmaes", "cmaes_config": cfg},
        "standard": {},
        "auto_chunked": {"method": "auto", "memory_limit_gb": 8e-7},
    }[route]
    with pytest.raises(ValueError, match="every row has zero weight"):
        curve_fit(
            _lin,
            x,
            2 * x,
            p0=[1.0],
            sigma=jnp.full((4,), jnp.inf, dtype=dtype),
            check_finite=False,
            **kw,
        )


def _line(x, a, b):
    return a + b * x


@pytest.mark.parametrize("route", ["standard", "class", "auto_chunked"])
def test_object_dtype_sigma_behaves_like_float_sigma(route):
    """claude C03 verdict5: np.isinf on an object sigma raised TypeError (HEAD worked)."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 50)
    y = 2 + 0.5 * x + 0.01 * rng.normal(size=50)
    sig_f = np.full(50, 0.1)
    sig_f[::7] = np.inf  # zero-weight rows must count the same way for object sigma
    sig_o = sig_f.astype(object)

    def run(sigma):
        kw = {"p0": [1.0, 1.0], "sigma": sigma, "check_finite": False}
        if route == "class":
            return CurveFit().curve_fit(_line, x, y, **kw)
        if route == "auto_chunked":
            return curve_fit(_line, x, y, method="auto", memory_limit_gb=3e-6, **kw)
        return curve_fit(_line, x, y, **kw)

    ref, got = run(sig_f), run(sig_o)
    np.testing.assert_allclose(got.popt, ref.popt, rtol=1e-8)
    np.testing.assert_allclose(got.pcov, ref.pcov, rtol=1e-6)
    # finite object sigma, as in the regression report
    ok = run(np.full(50, 0.1, dtype=object))
    np.testing.assert_allclose(ok.popt, [2.0, 0.5], atol=0.05)


def test_object_dtype_all_infinite_sigma_is_rejected_like_float():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="every row has zero weight"):
        curve_fit(
            _lin,
            x,
            2 * x,
            p0=[1.0],
            sigma=np.full(4, np.inf).astype(object),
            check_finite=False,
        )


def test_non_numeric_object_sigma_gets_no_preflight_error():
    from nlsq.core.problem import reject_all_zero_weight, zero_weight_rows

    reject_all_zero_weight(np.array(["a", "b"], dtype=object))
    assert zero_weight_rows(np.array([object(), object()]), np.ones(2, bool)) == 0


def test_object_dtype_sigma_through_curve_fit_large():
    x = np.linspace(0, 1, 50)
    y = 2 + 0.5 * x
    sig = np.full(50, 0.1)
    ref = nlsq.curve_fit_large(_line, x, y, p0=[1.0, 1.0], sigma=sig, size_threshold=10)
    got = nlsq.curve_fit_large(
        _line, x, y, p0=[1.0, 1.0], sigma=sig.astype(object), size_threshold=10
    )
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-8)


def _boxed(*values):
    arr = np.empty(len(values), dtype=object)
    arr[:] = values
    return arr


@pytest.mark.parametrize("route", ["standard", "class", "auto_chunked"])
@pytest.mark.parametrize(
    "sigma",
    [
        _boxed(*[np.complex128(1 + 2j)] * 4),
        _boxed(*[1 + 2j] * 4),
        _boxed(1.0, 2.0, 1 + 2j, 1.0),
    ],
    ids=["np_complex128", "py_complex", "mixed_real_complex"],
)
def test_boxed_complex_in_object_sigma_is_rejected(route, sigma):
    """codex verdict6: dtype=object holding complex was cast to real (ComplexWarning)."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            if route == "class":
                CurveFit().curve_fit(
                    _lin, x, 2 * x, p0=[1.0], sigma=sigma, check_finite=False
                )
            else:
                kw = (
                    {"method": "auto", "memory_limit_gb": 8e-7}
                    if route == "auto_chunked"
                    else {}
                )
                curve_fit(
                    _lin, x, 2 * x, p0=[1.0], sigma=sigma, check_finite=False, **kw
                )


def test_boxed_complex_in_object_xdata_ydata_and_tolerance_is_rejected():
    from nlsq.core._option_checks import check_not_nan

    x = np.array([1.0, 2.0, 3.0, 4.0])
    bad = _boxed(*[1 + 2j] * 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            curve_fit(_lin, x, bad, p0=[1.0], check_finite=False)
        with pytest.raises(ValueError, match="real-valued"):
            curve_fit(_lin, bad, 2 * x, p0=[1.0], check_finite=False)
        with pytest.raises(ValueError, match="real"):
            check_not_nan("gtol", np.array(1 + 2j, dtype=object))

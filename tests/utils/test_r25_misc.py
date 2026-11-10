"""Round-25 regression tests (CSV comments, Bounds, oscillation, threading, ier ...)."""

from __future__ import annotations

import contextlib
import threading
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import Bounds

from nlsq import CurveFit, curve_fit, curve_fit_large
from nlsq.utils.diagnostics import ConvergenceMonitor


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4, n)
    y = 3.0 * np.exp(-0.7 * x) + 0.01 * rng.standard_normal(n)
    return x, y


def test_csv_comment_with_unbalanced_quote_keeps_data(tmp_path):
    from nlsq.cli.data_loaders import DataLoader

    f = tmp_path / "d.csv"
    f.write_text('x,y\n# units,"s\n1,2\n3,4\n5,6\n7,8\n')
    x, y, _ = DataLoader().load(f, {"format": "csv", "csv": {"header": True}})
    np.testing.assert_allclose(x, [1, 3, 5, 7])
    np.testing.assert_allclose(y, [2, 4, 6, 8])


@pytest.mark.parametrize("kw", [{"multistart": True, "n_starts": 3}, {}])
def test_scipy_bounds_accepted_on_every_path(kw):
    x, y = _data()
    b = Bounds([0.0, 0.0], [10.0, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(_exp, x, y, p0=[1.0, 1.0], bounds=b, **kw)
    np.testing.assert_allclose(popt, [3.0, 0.7], rtol=0.05)


def test_scipy_bounds_accepted_by_curve_fit_large_default():
    x, y = _data(2000)
    b = Bounds([0.0, 0.0], [10.0, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit_large(_exp, x, y, p0=[1.0, 1.0], bounds=b)
    np.testing.assert_allclose(popt, [3.0, 0.7], rtol=0.05)


def test_oscillation_ignores_single_roundoff_blip():
    m = ConvergenceMonitor(window_size=10)
    costs = [10, 1, 1, 1, 1, 1, 1, 1, 1 + 1e-12, 1]
    params = [2, 1, 1, 1, 1, 1, 1, 1, 1 + 1e-13, 1]
    for c, p in zip(costs, params, strict=True):
        m.update(c, np.array([p], dtype=float))
    osc, score = m.detect_oscillation()
    assert not osc
    assert score < 0.5


def test_stability_auto_accepts_3d_xdata():
    from nlsq.stability.guard import estimate_condition_number

    X = np.random.default_rng(0).standard_normal((2, 3, 50))
    assert np.isfinite(estimate_condition_number(X))


def test_ld_memory_config_lowered_max_only():
    from nlsq.streaming.large_dataset import LDMemoryConfig

    cfg = LDMemoryConfig(max_chunk_size=500)
    assert cfg.min_chunk_size <= cfg.max_chunk_size == 500
    with pytest.raises(ValueError):
        LDMemoryConfig(min_chunk_size=5000, max_chunk_size=500)


def test_full_output_ier_reports_real_status():
    x, y = _data()
    out = curve_fit(_exp, x, y, p0=[1.0, 1.0], full_output=True, ftol=1e-3, xtol=1e-3)
    assert len(out) == 5
    assert out[4] in (1, 2, 3, 4)


def test_identifiability_correlation_scale_free():
    from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer
    from nlsq.diagnostics.types import DiagnosticsConfig

    def make():
        return IdentifiabilityAnalyzer(DiagnosticsConfig())

    rng = np.random.default_rng(0)
    J = rng.standard_normal((50, 3))
    base = make().analyze(J)
    tiny = make().analyze(J * 1e-170)
    assert tiny.correlation_matrix is not None
    assert np.all(np.isfinite(tiny.correlation_matrix))
    np.testing.assert_allclose(
        tiny.correlation_matrix, base.correlation_matrix, atol=1e-8
    )

    # FIM with badly scaled columns: correlations equal those of the same J
    Js = J * np.array([1.0, 1e4, 1e-4])
    rep = make().analyze_from_fim(Js.T @ Js)
    rep0 = make().analyze_from_fim(J.T @ J)
    np.testing.assert_allclose(
        rep.correlation_matrix, rep0.correlation_matrix, atol=1e-6
    )


def test_multistart_wrong_length_bounds_is_clear_error():
    x, y = _data()
    with pytest.raises(ValueError, match="p0 has 2 parameters"):
        curve_fit(
            _exp,
            x,
            y,
            p0=[1.0, 1.0],
            bounds=([0, 0, 0], [5, 5, 5]),
            multistart=True,
            n_starts=3,
        )


def test_single_chunk_large_fit_reports_nonconvergence():
    from nlsq import LargeDatasetFitter

    x, y = _data(500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = LargeDatasetFitter(memory_limit_gb=4.0).fit(
            _exp, x, y, p0=[1.0, 1.0], max_nfev=1
        )
    assert res.success is False
    # curve_fit_large keeps the curve_fit contract: warn, return best point
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        popt, _ = curve_fit_large(_exp, x, y, p0=[1.0, 1.0], max_nfev=1)
    assert popt.shape == (2,)


@pytest.mark.serial
def test_shared_curvefit_threads_do_not_poison_wrapper_cache():
    def lin(x, a, b):
        return a * x + b

    def expo(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    def run(cf, i):
        n = 20 + 7 * i
        xx = np.linspace(0, 1, n)
        if i % 2:
            return cf.curve_fit(lin, xx, (i + 1) * xx + i, p0=[1, 0])
        return cf.curve_fit(
            expo,
            xx,
            (i + 1) * np.exp(-2 * xx) + i,
            p0=[1, 1, 0],
            bounds=([0, 0, -1], [50, 10, 50]),
        )

    shared = CurveFit()

    def work(i):
        # sharing an instance across threads may fail; poisoning must not
        with contextlib.suppress(Exception):
            run(shared, i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
            [t.start() for t in ts]
            [t.join() for t in ts]
        # fresh instances afterwards must fit every model correctly
        for i in range(8):
            popt = np.asarray(run(CurveFit(), i).popt)
            expected = [i + 1, i] if i % 2 else [i + 1, 2, i]
            np.testing.assert_allclose(popt, expected, atol=1e-3)

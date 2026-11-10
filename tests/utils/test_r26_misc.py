"""Round-26 regression tests."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import Bounds

from nlsq import CurveFit, curve_fit


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


def test_adj_r_squared_matches_closed_form():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 30)
    y = 2 * x + 1 + 0.1 * rng.standard_normal(30)

    def lin(x, a, b):
        return a * x + b

    res = curve_fit(lin, x, y, p0=[1.0, 0.0])
    n, p = 30, 2
    r2 = 1 - np.sum((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2) / np.sum(
        (y - y.mean()) ** 2
    )
    expected = 1 - (1 - r2) * (n - 1) / (n - p)
    np.testing.assert_allclose(res.adj_r_squared, expected, rtol=1e-6)


def test_hybrid_infeasible_p0_is_projected_into_bounds():
    from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    def m(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 4000)
    y = 300 * np.exp(-1.3 * x) + 0.002 + 0.01 * rng.standard_normal(x.size)
    lb, ub = np.array([0, 1.4, -1.0]), np.array([1000, 10, 1.0])
    for norm in ("p0", "none"):
        o = AdaptiveHybridStreamingOptimizer(
            HybridStreamingConfig(
                normalization_strategy=norm, gauss_newton_max_iterations=30
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = o.fit(
                (x, y), m, np.array([100.0, 1.0, 0.0]), bounds=(lb, ub), verbose=0
            )
        xs = np.asarray(r["x"])
        assert np.all(xs >= lb - 1e-9) and np.all(xs <= ub + 1e-9)
        assert xs[1] > 1.39  # was stuck at the infeasible p0 value 1.0


def test_chunked_pcov_with_robust_loss_matches_full_fit():
    from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig

    rng = np.random.default_rng(3)
    n = 6000

    def e(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    x = np.linspace(0, 5, n)
    y = 3 * np.exp(-1.3 * x) + 0.5 + 0.05 * rng.standard_normal(n)
    y[::50] += 3
    cfg = LDMemoryConfig(
        memory_limit_gb=0.0001, min_chunk_size=100, max_chunk_size=1000
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = LargeDatasetFitter(config=cfg).fit(
            e, x, y, p0=[1, 1, 0], loss="huber", f_scale=0.1
        )
        _, c = curve_fit(e, x, y, p0=[1, 1, 0], loss="huber", f_scale=0.1)
    assert r.n_chunks > 1
    ratio = np.diag(r.pcov) / np.diag(c)
    assert np.all(ratio < 1.5) and np.all(ratio > 0.67), ratio


def test_sampler_dimension_limits():
    from nlsq.global_optimization.sampling import halton_sample

    out = halton_sample(8, 25)
    assert np.asarray(out).shape == (8, 25)


def test_sobol_above_limit_falls_back_to_lhs_with_warning():
    from nlsq.global_optimization.config import GlobalOptimizationConfig
    from nlsq.global_optimization.multi_start import MultiStartOrchestrator

    orch = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=4, sampler="sobol"))
    lb, ub = np.zeros(25), np.ones(25)
    with pytest.warns(UserWarning, match="Sobol"):
        pts = orch._generate_starting_points(n_params=25, lb=lb, ub=ub)
    assert np.asarray(pts).shape == (4, 25)


def test_detect_oscillation_accepts_scalar_params():
    from nlsq.utils.diagnostics import ConvergenceMonitor

    m = ConvergenceMonitor(window_size=5)
    for i in range(8):
        m.update(1.0 + (i % 2), np.array(float(i % 2)))
    assert isinstance(m.detect_oscillation()[0], (bool, np.bool_))


def test_scalar_scipy_bounds_accepted():
    x = np.linspace(0, 4, 100)
    y = 3.0 * np.exp(-0.7 * x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(_exp, x, y, p0=[1.0, 1.0], bounds=Bounds(-np.inf, np.inf))
    np.testing.assert_allclose(popt, [3.0, 0.7], rtol=1e-3)


def test_auto_method_nonconverged_chunked_fit_returns_not_raises():
    x = np.linspace(0, 4, 50_000)
    y = 2.5 * np.exp(-1.3 * x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = CurveFit().curve_fit(
            _exp, x, y, p0=[1, 1], max_nfev=2, method="auto", memory_limit_gb=0.01
        )
    assert np.asarray(r.popt).shape == (2,)


def test_csv_hash_inside_quoted_cell_does_not_drop_row(tmp_path):
    from nlsq.cli.data_loaders import DataLoader

    f = tmp_path / "d.csv"
    f.write_text('x,y\n1,2\n"#3",6\n4,8 # trailing comment\n')
    from nlsq.cli.errors import DataLoadError

    # the quoted "#3" row is kept (as a NaN that validation rejects), not
    # silently dropped as a comment
    with pytest.raises(DataLoadError, match="non-finite"):
        DataLoader().load(f, {"format": "csv", "csv": {"header": True}})
    f.write_text("x,y\n1,2\n4,8 # trailing comment\n")
    _x, y, _ = DataLoader().load(f, {"format": "csv", "csv": {"header": True}})
    np.testing.assert_allclose(y, [2, 8])


def test_cli_data_path_resolves_beside_yaml(tmp_path, monkeypatch):
    from nlsq.cli.commands.fit import run_fit

    here = tmp_path / "cwd"
    wf = tmp_path / "wf"
    here.mkdir()
    wf.mkdir()
    xs = np.linspace(0, 1, 20)
    (here / "data.csv").write_text(
        "\n".join(f"{a},{9 * a}" for a in xs)
    )  # decoy slope 9 in CWD
    (wf / "data.csv").write_text("\n".join(f"{a},{2 * a}" for a in xs))
    (wf / "w.yaml").write_text(
        "data:\n  input_file: data.csv\n  format: csv\n"
        "  columns: {x: 0, y: 1}\n"
        "model:\n  type: builtin\n  name: linear\n"
        f"export:\n  format: json\n  results_file: {tmp_path / 'out.json'}\n"
    )
    monkeypatch.chdir(here)
    res = run_fit(str(wf / "w.yaml"))
    assert abs(float(np.asarray(res["popt"])[0]) - 2.0) < 1e-6


def test_headerless_csv_with_default_header_keeps_first_row(tmp_path):
    from nlsq.cli.data_loaders import DataLoader

    f = tmp_path / "d.csv"
    f.write_text("1,2\n3,4\n5,6\n")
    x, y, _ = DataLoader().load(f, {"format": "csv"})  # default csv.header=True
    np.testing.assert_allclose(x, [1, 3, 5])
    np.testing.assert_allclose(y, [2, 4, 6])

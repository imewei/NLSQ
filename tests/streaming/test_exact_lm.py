"""Stage 0 tests for the exact chunk-accumulated LM: termination reasons and pass budget."""

import jax.numpy as jnp
import numpy as np
import pytest

import benchmarks.prototypes.exact_accumulated_lm as lm
from benchmarks.compare.problems import PROBLEMS

EXP = PROBLEMS["exp_decay"]
N = 3000


def run(model, x, y, p0, **kw):
    """Implementation under test (re-pointed to the library module in Task 3)."""
    kw.setdefault("chunk", 1024)  # 3000 points -> 3 chunks with a padded tail
    return lm.fit(model, x, y, p0, **kw)


def exp_model(x, a, b, c):
    return EXP.model(jnp, x, a, b, c)


@pytest.fixture(scope="module")
def data():
    return EXP.data(N)


@pytest.fixture(scope="module")
def default_run(data):
    return run(exp_model, *data, EXP.p0)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_budget_ends_max_nfev_with_exact_pass_count(data, default_run, k):
    assert default_run["nfev"] > 3  # premise: k <= 3 cannot converge
    out = run(exp_model, *data, EXP.p0, max_nfev=k)
    assert out["termination"] == "max_nfev"
    assert out["nfev"] == k
    assert out["success"] is False


def test_nonfinite_everywhere_but_p0_stalls(data):
    a0 = EXP.p0[0]

    def model(x, a, b, c):
        return jnp.where(a == a0, exp_model(x, a, b, c), jnp.nan)

    out = run(model, *data, EXP.p0)
    assert out["termination"] == "stalled"
    assert out["success"] is False


def test_forced_rejections_stall(data, monkeypatch):
    real = lm._pass
    first = {}

    def fake(stats, theta, x, y, chunk):
        if not first:
            first["v"] = real(stats, theta, x, y, chunk)
            return first["v"]
        c, g, H = first["v"]
        return 2.0 * c, g, H

    monkeypatch.setattr(lm, "_pass", fake)
    out = run(exp_model, *data, EXP.p0)
    assert out["termination"] == "stalled"
    assert out["success"] is False


def test_full_rank_start_at_optimum_one_pass(data):
    x, y = data
    tight = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
    opt = run(exp_model, x, y, EXP.p0, **tight)
    assert opt["success"]
    again = run(exp_model, x, y, opt["x"])
    assert again["nfev"] == 1
    assert again["success"] is True
    assert again["termination"] in ("gtol", "ftol")

    y_clean = EXP.model(np, x, *EXP.p_true)
    exact = run(exp_model, x, y_clean, EXP.p_true)
    assert exact["termination"] == "gtol"
    assert exact["nfev"] == 1


def test_step_termination_gates():
    F = 2.0
    big = {"s_norm": 1.0, "x_norm": 1.0, "ftol": 1e-8, "xtol": 1e-8}
    assert lm.step_termination(0.5e-8 * F, F, 0.1, **big) is None
    assert lm.step_termination(0.5e-8 * F, F, 0.5, **big) == "ftol"
    assert lm.step_termination(0.1 * F, F, 0.5, 1e-12 * 3.0, 3.0, 1e-8, 1e-8) == "xtol"


def test_newton_certificate_rescaling():
    g = np.array([0.0, 1.0])
    for big in (1e20, 1e30):
        assert not lm.newton_certificate(g, np.diag([big, 1.0]), 0.5, 1e-8)

    J = np.array([[1e15, 1e15], [1.0, -1.0]])
    r = np.array([0.0, 1.0])
    assert not lm.newton_certificate(J.T @ r, J.T @ J, 0.5, 1e-8)

    g_small = 1e-9 * np.ones(2)
    assert lm.newton_certificate(g_small, np.eye(2), 1.0, 1e-8)
    assert not lm.newton_certificate(g_small, np.eye(2), 1.0, 1e-8, bounds_active=True)


def test_zero_target_exact_fit(data):
    x, _ = data

    def line(x, a, c):  # exactly zero at (0, 0); full-rank Jacobian
        return a * x + c

    out = run(line, x, np.zeros_like(x), (0.0, 0.0))
    assert out["termination"] == "gtol"
    assert out["nfev"] == 1
    assert np.all(np.isfinite(out["x"])) and np.isfinite(out["cost"])
    assert not np.any(np.isnan(out["pcov"]))

    theta = np.asarray(EXP.p_true)
    y = EXP.model(np, x, *theta) + 1e-15 * x
    stats = lm.make_stats(exp_model)
    F, _, _ = lm._pass(stats, theta, x, y, 1024)
    sum_y2 = float(np.sum(y * y))
    assert 0.0 < F <= 1e-24 * sum_y2

    out = run(exp_model, x, y, theta)
    assert out["success"] is True
    assert np.all(np.isfinite(out["x"])) and np.isfinite(out["cost"])
    _, g, H = lm._pass(stats, out["x"], x, y, 1024)
    # near-exact regime: coordinatewise test, no division by sqrt(2F)
    assert np.all(np.abs(g) <= 1e-12 * np.sqrt(sum_y2) * np.sqrt(np.diag(H)))


def test_convergence_beats_budget(data, default_run):
    assert default_run["success"]
    k = default_run["nfev"]
    out = run(exp_model, *data, EXP.p0, max_nfev=k)
    assert out["success"] is True
    assert out["nfev"] == k
    assert out["termination"] == default_run["termination"]

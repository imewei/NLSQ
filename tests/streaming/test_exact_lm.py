"""Exact chunk-accumulated LM library module: termination reasons, pass budget, chunk size.

The same Stage 0 cases against the frozen prototype live in test_exact_lm_prototype.py.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq.streaming.exact_lm as lm
from benchmarks.compare.problems import PROBLEMS

EXP = PROBLEMS["exp_decay"]
N = 3000


def run(model, x, y, p0, **kw):
    """Implementation under test: nlsq.streaming.exact_lm.exact_lm_fit."""
    kw.setdefault("chunk_size", 1024)  # 3000 points -> 3 chunks with a padded tail
    return lm.exact_lm_fit(model, x, y, p0, **kw)


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
    real = lm.run_pass
    first = {}

    def fake(stats, theta, source, chunk, **kw):
        if not first:
            first["v"] = real(stats, theta, source, chunk, **kw)
            return first["v"]
        return first["v"]._replace(cost=2.0 * first["v"].cost)

    monkeypatch.setattr(lm, "run_pass", fake)
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
    stats = lm.make_chunk_stats(exp_model)
    F = lm.run_pass(stats, theta, (x, y, None), 1024).cost
    sum_y2 = float(np.sum(y * y))
    assert 0.0 < F <= 1e-24 * sum_y2

    out = run(exp_model, x, y, theta)
    assert out["success"] is True
    assert np.all(np.isfinite(out["x"])) and np.isfinite(out["cost"])
    st = lm.run_pass(stats, out["x"], (x, y, None), 1024)
    g, H = st.g, st.H
    # near-exact regime: coordinatewise test, no division by sqrt(2F)
    assert np.all(np.abs(g) <= 1e-12 * np.sqrt(sum_y2) * np.sqrt(np.diag(H)))


def test_convergence_beats_budget(data, default_run):
    assert default_run["success"]
    k = default_run["nfev"]
    out = run(exp_model, *data, EXP.p0, max_nfev=k)
    assert out["success"] is True
    assert out["nfev"] == k
    assert out["termination"] == default_run["termination"]


def zero_col_model(x, a, b, c, d):
    return EXP.model(jnp, x, a, b, c) + 0.0 * d


def collinear_model(x, a1, a2, b):  # columns 1 and 2 of J identical, H has rank 2
    return (a1 + a2) * jnp.exp(-b * x)


def test_zero_jacobian_column_converges(data):
    p0 = (*EXP.p0, 1.7)
    out = run(zero_col_model, *data, p0)
    assert out["success"] is True
    assert out["x"][3] == 1.7  # untouched, exactly
    assert np.all(np.isfinite(out["x"]))


def test_nan_in_H_stalls_within_attempt_cap(data, monkeypatch):
    real = lm.run_pass
    first = {}

    def fake(stats, theta, source, chunk, **kw):
        if not first:
            first["v"] = real(stats, theta, source, chunk, **kw)
            return first["v"]
        v = first["v"]
        return v._replace(cost=0.5 * v.cost, H=np.full_like(v.H, np.nan))

    monkeypatch.setattr(lm, "run_pass", fake)
    # LAM0=1e-300 needs ~46 rejections to reach the 1e14 ceiling, so the cap stops the run
    monkeypatch.setattr(lm, "LAM0", 1e-300)
    out = run(exp_model, *data, EXP.p0)
    assert out["termination"] == "stalled"
    assert out["success"] is False
    assert out["nfev"] == 1 + 20


def test_failed_solves_count_as_attempts_without_data_passes(data, monkeypatch):
    calls = []
    monkeypatch.setattr(lm, "_solve", lambda *a: calls.append(1))  # returns None
    monkeypatch.setattr(lm, "LAM0", 1e-300)
    out = run(exp_model, *data, EXP.p0)
    assert out["termination"] == "stalled"
    assert out["success"] is False
    assert out["nfev"] == 1  # no trial reached the data
    assert len(calls) == 20  # one failed solve per attempt, then the cap


def test_rank_two_h_for_p3_converges(data):
    x, y = data
    out = run(collinear_model, x, y - 0.5, (1.0, 0.5, 1.0))
    assert out["success"] is True


def test_noisy_collinear_start_at_optimum_success_or_stall(data):
    x, y = data
    first = run(collinear_model, x, y - 0.5, (1.0, 0.5, 1.0))
    assert first["success"]
    out = run(collinear_model, x, y - 0.5, first["x"])
    ok = out["success"] and out["termination"] in ("gtol", "ftol", "xtol")
    stalled = out["termination"] == "stalled" and "rank" in out["message"]
    assert ok or stalled, (out["termination"], out["message"])


def test_float32_input_equals_float64_input(data):
    def model(x, a, b, c):  # sqrt(x + 1) is rounded in float32 unless x is cast first
        return a * jnp.exp(-b * jnp.sqrt(x + 1.0)) + c

    x32, y32 = (np.asarray(a, dtype=np.float32) for a in data)
    a = run(model, x32, y32, EXP.p0)
    b = run(model, x32.astype(np.float64), y32.astype(np.float64), EXP.p0)
    assert np.array_equal(a["x"], b["x"])
    assert a["nfev"] == b["nfev"]


def test_nonfinite_data_raises(data):
    x, y = data
    y_bad = y.copy()
    y_bad[7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run(exp_model, x, y_bad, EXP.p0)
    x_bad = x.copy()
    x_bad[2999] = np.inf  # in the padded tail chunk
    with pytest.raises(ValueError, match="finite"):
        run(exp_model, x_bad, y, EXP.p0)
    with pytest.raises(ValueError, match="finite"):  # model is inf at p0
        run(lambda x, a, b, c: jnp.full_like(x, jnp.inf) + a, x, y, EXP.p0)


def test_accepted_step_cannot_push_lambda_past_ceiling(data, monkeypatch):
    # Every trial gains a vanishing cost, so rho ~ 0 and Nielsen's factor is ~2. Starting at
    # lam = 0.9e14 the update would reach 1.8e14, above the ceiling; an uncapped lam makes the
    # next iteration skip its trials and report "stalled" after 2 passes without trying a step.
    real = lm.run_pass
    calls = []

    def fake(stats, theta, source, chunk, **kw):
        if not calls:
            calls.append(real(stats, theta, source, chunk, **kw))
            return calls[0]
        calls.append(None)
        return calls[0]._replace(cost=calls[0].cost * (1.0 - 1e-15 * len(calls)))

    monkeypatch.setattr(lm, "run_pass", fake)
    monkeypatch.setattr(lm, "LAM0", 0.9e14)
    out = run(exp_model, *data, EXP.p0, ftol=0.0, xtol=0.0, max_nfev=4)
    assert out["termination"] == "max_nfev"
    assert out["nfev"] == 4


# --- Stage 1 module-only behaviour ---------------------------------------------------


def test_derive_chunk_size_precedence():
    # explicit positive chunk wins (even over an impossible budget) and is clamped to n
    assert lm.derive_chunk_size(3000, 3, None, 1024) == 1024
    assert lm.derive_chunk_size(3000, 3, None, 10**9) == 3000
    assert lm.derive_chunk_size(3000, 3, 1e-12, 512) == 512
    with pytest.raises(ValueError, match="chunk_size"):
        lm.derive_chunk_size(3000, 3, None, 0)
    # default 1 << 22, clamped to n
    assert lm.derive_chunk_size(10**9, 3, None, None) == 1 << 22
    assert lm.derive_chunk_size(5000, 3, None, None) == 5000
    # budget reduces the chunk: workspace is chunk * p * 8 B * 4
    gb = 2**30
    assert lm.derive_chunk_size(10**9, 4, (1 << 20) * 4 * 8 * 4 / gb, None) == 1 << 20
    # soft floor 1024, and n < 1024 gives n
    floor_gb = 1024 * 4 * 8 * 4 / gb
    assert lm.derive_chunk_size(10**9, 4, floor_gb, None) == 1024
    assert lm.derive_chunk_size(10**9, 4, floor_gb * 1.5, None) == 1536
    assert lm.derive_chunk_size(500, 4, None, None) == 500
    with pytest.raises(ValueError, match="memory_limit_gb"):
        lm.derive_chunk_size(10**9, 4, floor_gb * 0.5, None)


def test_stage2_inputs_rejected(data):
    with pytest.raises(ValueError, match="Stage 2"):
        run(exp_model, *data, EXP.p0, sigma=np.ones(N))
    with pytest.raises(ValueError, match="Stage 2"):
        run(exp_model, *data, EXP.p0, bounds=(0.0, np.inf))
    with pytest.raises(ValueError, match="n_points"):
        run(exp_model, *data, EXP.p0, n_points=N + 1)


def test_result_fields_and_check_finite_off(data, default_run):
    assert default_run["equivalent_to_full_batch"] is True
    assert default_run["n_points"] == N
    x, y = data
    y_bad = y.copy()
    y_bad[7] = np.nan
    # with the check off the NaN reaches the cost at p0, which is always rejected
    with pytest.raises(ValueError, match="not finite at p0"):
        run(exp_model, x, y_bad, EXP.p0, check_finite=False)


def test_show_progress_logs_one_line_per_pass(data, default_run, monkeypatch):
    lines = []
    monkeypatch.setattr(lm.logger, "info", lambda msg, **kw: lines.append(kw))
    out = run(exp_model, *data, EXP.p0, show_progress=True)
    assert [d["nfev"] for d in lines] == list(range(1, out["nfev"] + 1))
    assert lines[-1]["cost"] == out["cost"]


def test_jax_input_equals_numpy_input(data, default_run):
    out = run(exp_model, *(jnp.asarray(a) for a in data), EXP.p0)
    np.testing.assert_allclose(out["x"], default_run["x"], rtol=1e-12)
    assert out["nfev"] == default_run["nfev"]

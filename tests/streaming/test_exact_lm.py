"""Exact chunk-accumulated LM library module: termination reasons, pass budget, chunk size.

The same Stage 0 cases against the frozen prototype live in test_exact_lm_prototype.py.
"""

import types
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares

import nlsq.streaming.exact_lm as lm
from benchmarks.compare.problems import PROBLEMS
from nlsq import curve_fit
from nlsq.core.covariance import (
    normal_condition,
    pcov_from_jacobian,
    pcov_from_normal,
)
from nlsq.core.minpack import _covariance_svd
from nlsq.result.optimize_warning import OptimizeWarning
from tests.streaming.exact_lm_kkt import assert_kkt

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


@pytest.mark.filterwarnings(
    "ignore:normal-equation Hessian is ill-conditioned:UserWarning"
)
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


@pytest.mark.filterwarnings(
    "ignore:normal-equation Hessian is ill-conditioned:UserWarning"
)
def test_rank_two_h_for_p3_converges(data):
    x, y = data
    out = run(collinear_model, x, y - 0.5, (1.0, 0.5, 1.0))
    assert out["success"] is True


@pytest.mark.filterwarnings(
    "ignore:normal-equation Hessian is ill-conditioned:UserWarning"
)
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
    with pytest.raises(ValueError, match="x and y must be finite"):
        run(exp_model, x, y_bad, EXP.p0)
    x_bad = x.copy()
    x_bad[2999] = np.inf  # in the padded tail chunk
    with pytest.raises(ValueError, match="x and y must be finite"):
        run(exp_model, x_bad, y, EXP.p0)
    with pytest.raises(ValueError, match="not finite at p0"):  # model is inf at p0
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


def test_n_points_mismatch_rejected(data):
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


# --- Task 4: SciPy parity, chunk invariance, model contract, finite checks -----------

TIGHT = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
UNBOUNDED = ["exp_decay", "gaussian", "two_exp"]


def model_of(prob):
    return lambda x, *th: prob.model(jnp, x, *th)


def scipy_fit(prob, x, y, p0):
    return least_squares(
        lambda th: y - prob.model(np, x, *th),
        p0,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )


def by_rate(prob, theta):
    """two_exp is symmetric under swapping (a1, k1) with (a2, k2): order terms by rate."""
    t = np.asarray(theta).reshape(-1, 2) if prob.name == "two_exp" else None
    return theta if t is None else t[np.argsort(t[:, 1])].ravel()


def scaled_gradient(model, x, y, theta):
    st = lm.run_pass(lm.make_chunk_stats(model), theta, (x, y, None), 1024)
    return np.abs(st.g) / (np.sqrt(np.diag(st.H)) * np.sqrt(2.0 * st.cost))


def assert_close(a, b, rtol, atol=0.0):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize("name", UNBOUNDED)
def test_matches_scipy(name):
    prob = PROBLEMS[name]
    x, y = prob.data(N)
    ref = scipy_fit(prob, x, y, prob.p0)
    out = run(model_of(prob), x, y, prob.p0, **TIGHT)
    assert out["success"]
    # two_exp is a flat valley (kappa(J) ~ 4e2): the spec table compares it at rtol 1e-4,
    # and both costs still agree to 1e-9 below
    rtol = 1e-4 if name == "two_exp" else 1e-6
    assert_close(by_rate(prob, out["x"]), by_rate(prob, ref.x), rtol, 1e-8)
    assert_close(out["cost"], ref.cost, 1e-9)
    assert np.max(scaled_gradient(model_of(prob), x, y, out["x"])) <= 1e-6


@pytest.mark.parametrize("name", UNBOUNDED)
def test_chunk_size_independence(name):
    prob = PROBLEMS[name]
    x, y = prob.data(N)
    outs = [
        run(model_of(prob), x, y, prob.p0, chunk_size=c)
        for c in (N, N // 2, 512, 513, 7)
    ]
    for o in outs[1:]:
        assert_close(o["x"], outs[0]["x"], 1e-9)
        assert_close(o["cost"], outs[0]["cost"], 1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_randomized_problems_match_scipy(seed):
    rng = np.random.default_rng(seed)
    p_true = (rng.uniform(1.0, 4.0), rng.uniform(0.5, 2.0), rng.uniform(-1.0, 1.0))
    x = np.linspace(0.0, 4.0, 2000)
    y = EXP.model(np, x, *p_true) + 0.05 * rng.standard_normal(x.size)
    ref = scipy_fit(EXP, x, y, EXP.p0)
    out = run(exp_model, x, y, EXP.p0, **TIGHT)
    assert out["success"]
    assert_close(out["x"], ref.x, 1e-6, 1e-8)
    assert_close(out["cost"], ref.cost, 1e-9)
    assert np.max(scaled_gradient(exp_model, x, y, out["x"])) <= 1e-6


def test_non_pointwise_model_raises(data):
    def centred(x, a, b):  # depends on the batch through x.mean()
        return a * jnp.exp(-b * (x - x.mean()))

    with pytest.raises(ValueError, match="not pointwise"):
        run(centred, *data, (1.0, 1.0))


def test_check_model_contract_accepts_pointwise_and_flags_reversal_only(data):
    x, _ = data
    lm.check_model_contract(exp_model, EXP.p0, x[:1024])
    lm.check_model_contract(exp_model, EXP.p0, x[:1])  # nothing to compare on one point
    # first half agrees with the full batch, the reversed copy does not
    with pytest.raises(ValueError, match="not pointwise"):
        lm.check_model_contract(lambda x, a: a * jnp.cumsum(x), (1.0,), x[:1024])
    # a scalar output cannot be pointwise either
    with pytest.raises(ValueError, match="not pointwise"):
        lm.check_model_contract(lambda x, a: a * jnp.sum(x), (1.0,), x[:1024])


@pytest.mark.parametrize(
    "model",
    [
        lambda x, a: x.at[:].multiply(a),  # needs a JAX array x
        lambda x, a: a.astype(jnp.float64) * x,  # needs a JAX scalar a
    ],
    ids=["x_at", "a_astype"],
)
def test_contract_probe_uses_jax_float64_inputs(model):
    x = np.linspace(0.0, 2.0, 500)
    y = 3.0 * x
    lm.check_model_contract(model, (1.0,), x)  # NumPy chunk, as the caller may pass
    out = run(model, x, y, (1.0,), chunk_size=128)
    assert out["success"]
    assert_close(out["x"], [3.0], 1e-8)


def test_check_finite_flag(data):
    x, y = data
    y_bad = y.copy()
    y_bad[7] = np.nan
    with pytest.raises(ValueError, match="x and y must be finite"):
        run(exp_model, x, y_bad, EXP.p0, check_finite=True)
    with pytest.raises(ValueError, match="not finite at p0"):
        run(exp_model, x, y_bad, EXP.p0, check_finite=False)


@pytest.mark.filterwarnings(
    "ignore:normal-equation Hessian is ill-conditioned:UserWarning"
)
def test_infinite_x_caught_by_fused_data_check_not_by_p0_cost(data):
    x, y = data
    x_bad = x.copy()
    x_bad[3] = np.inf

    def const(x, a, b, c):  # ignores x, so every statistic stays finite despite the inf
        return (a + b + c) * jnp.ones_like(x)

    stats = lm.make_chunk_stats(const)
    st = lm.run_pass(stats, EXP.p0, (x_bad, y, None), 1024)
    assert np.isfinite(st.cost) and np.isfinite(st.g).all() and np.isfinite(st.H).all()
    with pytest.raises(ValueError, match="x and y must be finite"):
        run(const, x_bad, y, EXP.p0)
    # without the fused check the same data is accepted: the guard is the only line of defence
    out = run(const, x_bad, y, EXP.p0, check_finite=False)
    assert np.isfinite(out["cost"])


def test_tiny_and_p1_cases():
    def line(x, a):
        return a * x

    x, y = np.array([1.0, 2.0]), np.array([2.1, 3.9])
    one = run(line, x, y, (1.0,), chunk_size=2, **TIGHT)
    assert one["success"]
    assert_close(one["x"], [x @ y / (x @ x)], 1e-9)  # linear least squares, closed form
    assert_close(run(line, x, y, (1.0,), chunk_size=1, **TIGHT)["x"], one["x"], 1e-9)
    assert_close(run(line, x, y, (1.0,), chunk_size=None, **TIGHT)["x"], one["x"], 1e-9)

    x5 = np.linspace(0.0, 4.0, 5)  # n < the 1024-point chunk
    y5 = EXP.model(np, x5, *EXP.p_true) + 0.01 * np.arange(5)
    whole = run(exp_model, x5, y5, EXP.p0, chunk_size=5)
    for c in (1024, 2, 3):  # clamped to n, and multi-chunk with a padded tail
        other = run(exp_model, x5, y5, EXP.p0, chunk_size=c)
        assert_close(other["x"], whole["x"], 1e-9)
        assert_close(other["cost"], whole["cost"], 1e-12)


def test_permutation_invariance(data):
    x, y = data
    perm = np.random.default_rng(1).permutation(N)
    a = run(exp_model, x, y, EXP.p0, **TIGHT)
    b = run(exp_model, x[perm], y[perm], EXP.p0, **TIGHT)
    assert_close(b["x"], a["x"], 1e-9)


def test_derive_chunk_size_budget_and_edges():
    chunk = lm.derive_chunk_size(10**8, 4, 0.01, None)
    assert chunk >= 1024 and chunk * 4 * 8 * 4 <= 0.01 * 2**30
    assert lm.derive_chunk_size(10**8, 4, 0.01, 7) == 7
    assert lm.derive_chunk_size(2, 4, None, None) == 2
    with pytest.raises(ValueError, match="memory_limit_gb"):
        lm.derive_chunk_size(10**8, 4, 1e-9, None)


def test_cpu_chunks_are_views(data):
    x, y = data
    xc, yc, _, mask = next(lm.iter_blocks(x, y, None, 1000))
    assert np.shares_memory(xc, x) and np.shares_memory(yc, y)
    assert bool(mask.all())


def test_list_and_int_inputs():
    x = np.arange(0, 40)  # int64
    y = np.rint(3.0 * np.exp(-0.05 * x) * 10) / 10 + 0.5

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    ref = run(model, x.astype(float), y, (1.0, 0.1, 0.0))
    assert ref["success"]
    for xs, ys in ((x, y), (x.tolist(), y.tolist()), (x.tolist(), y)):
        out = run(model, xs, ys, (1.0, 0.1, 0.0))
        assert np.array_equal(out["x"], ref["x"]) and out["nfev"] == ref["nfev"]


@pytest.mark.serial
@pytest.mark.slow
def test_cpu_fit_peak_memory():
    import tracemalloc

    if jax.default_backend() != "cpu":
        pytest.skip("tracemalloc cannot see device buffers")

    n, chunk = 20_000_000, 1_000_000
    x = np.linspace(0.0, 4.0, n)
    y = EXP.model(np, x, *EXP.p_true)
    y += 0.05 * np.random.default_rng(0).standard_normal(n)
    tracemalloc.start()
    try:
        out = run(exp_model, x, y, EXP.p0, chunk_size=chunk, max_nfev=3)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert out["n_points"] == n
    assert peak < 8 * chunk * 8 + 64 * 2**20, peak  # a whole-array copy of x is 160 MB


# ---- Task 5: unweighted covariance contract -------------------------------------------

TIGHT = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
COV_WARN = "Covariance of the parameters could not be estimated"


@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("name", ["exp_decay", "gaussian"])
def test_pcov_matches_curve_fit(name, absolute_sigma):
    prob = PROBLEMS[name]
    x, y = prob.data(N)

    def model(x, *p):
        return prob.model(jnp, x, *p)

    out = run(model, x, y, prob.p0, absolute_sigma=absolute_sigma, **TIGHT)
    _, ref = curve_fit(
        model,
        x,
        y,
        p0=prob.p0,
        absolute_sigma=absolute_sigma,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    assert out["success"]
    assert out["pcov"].dtype == np.float64 and out["pcov"].shape == ref.shape
    # atol only absorbs ~1e-17-relative roundoff in structurally-zero off-diagonals
    atol = 1e-12 * np.abs(ref).max()
    np.testing.assert_allclose(out["pcov"], ref, rtol=1e-6, atol=atol)


@pytest.mark.filterwarnings(
    "ignore:normal-equation Hessian is ill-conditioned:UserWarning"
)
def test_rank_deficient_pcov_finite_and_zero_on_dropped(data):
    p0 = (*EXP.p0, 1.7)
    out = run(zero_col_model, *data, p0, absolute_sigma=True, **TIGHT)
    pcov = out["pcov"]
    assert np.all(np.isfinite(pcov))
    assert np.all(pcov[3, :] == 0.0) and np.all(pcov[:, 3] == 0.0)
    # the identified block equals the plain 3-parameter covariance
    ref = run(exp_model, *data, EXP.p0, absolute_sigma=True, **TIGHT)
    np.testing.assert_allclose(pcov[:3, :3], ref["pcov"], rtol=1e-6, atol=0.0)
    assert np.all(np.linalg.eigvalsh(pcov) >= -1e-12 * np.abs(pcov).max())


def poly7(x, *c):
    return sum(ck * x**k for k, ck in enumerate(c))


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_pcov_parity_to_cond_1e6(absolute_sigma):
    x = np.linspace(0.0, 1.0, N)
    c_true = np.array([1.0, -2.0, 3.0, -1.0, 0.5, 2.0, -1.5, 0.7])
    y = sum(ck * x**k for k, ck in enumerate(c_true))
    y = y + 0.01 * np.random.default_rng(0).standard_normal(N)
    J = np.vander(x, 8, increasing=True)
    kappa = np.linalg.cond(J)
    assert 1e4 <= kappa <= 1e6, kappa
    p0 = np.zeros(8)
    with pytest.warns(UserWarning, match="ill-conditioned"):
        out = run(poly7, x, y, p0, absolute_sigma=absolute_sigma, **TIGHT)
    _, ref = curve_fit(poly7, x, y, p0=p0, absolute_sigma=absolute_sigma)
    np.testing.assert_allclose(out["pcov"], ref, rtol=1e-3, atol=0.0)


def test_n_le_p_gives_inf_with_warning():
    x = np.array([0.0, 1.0, 2.0])
    y = EXP.model(np, x, *EXP.p_true)
    with pytest.warns(Warning, match=COV_WARN) as rec:
        out = run(exp_model, x, y, EXP.p0)
    assert any(w.category is OptimizeWarning for w in rec)
    assert out["pcov"].shape == (3, 3) and np.all(np.isposinf(out["pcov"]))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=COV_WARN)
        out = run(exp_model, x, y, EXP.p0, absolute_sigma=True)
    assert np.all(np.isfinite(out["pcov"]))


def test_ill_conditioned_warns():
    # effectively rank-deficient: k1 ~ k2 gives cond(J) ~ 1e16 at the fitted point
    prob = PROBLEMS["two_exp"]
    p_true = (2.0, 1.0, 1.5, 1.0001)
    x = np.linspace(0.0, 4.0, N)
    y = prob.model(np, x, *p_true) + 0.01 * np.random.default_rng(0).standard_normal(N)

    def model(x, a1, k1, a2, k2):
        return prob.model(jnp, x, a1, k1, a2, k2)

    with pytest.warns(UserWarning, match="ill-conditioned"):
        run(model, x, y, (1.0, 1.0, 1.0, 1.0001))


def test_compute_covariance_contract():
    s2 = 2 * 0.4 / (10 - 2)
    # a direction below p*eps*lambda_max is dropped: zero row and column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pc = lm.compute_covariance(np.diag([4.0, 1e-17]), 0.4, 10, 2, False)
    np.testing.assert_array_equal(pc, np.diag([s2 / 4.0, 0.0]))
    # just above the cutoff (4 * 2.2e-16 * 4 = 3.6e-15) it is kept
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pc = lm.compute_covariance(np.diag([4.0, 1e-13]), 0.4, 10, 2, True)
    np.testing.assert_allclose(pc, np.diag([0.25, 1e13]), rtol=1e-12)
    assert pc.dtype == np.float64
    # no surviving direction / non-finite H: all-inf with OptimizeWarning
    overflow = np.diag([1e-310, 1e-310])  # finite H whose inverse overflows
    for H in (np.zeros((2, 2)), np.array([[1.0, np.nan], [np.nan, 1.0]]), overflow):
        with pytest.warns(OptimizeWarning, match=COV_WARN):
            warnings.simplefilter("error", RuntimeWarning)  # no numpy overflow noise
            pc = lm.compute_covariance(H, 0.4, 10, 2, False)
        assert np.all(np.isposinf(pc))
    # conditioning warning threshold: sqrt(cond H) = 3162 quiet, 31623 warns
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lm.compute_covariance(np.diag([1.0, 1e-7]), 0.4, 10, 2, False)
    with pytest.warns(UserWarning, match="ill-conditioned"):
        lm.compute_covariance(np.diag([1.0, 1e-9]), 0.4, 10, 2, False)


def test_compute_covariance_overflow_safe():
    n, p, cost = 10, 1, 0.4
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning, no OptimizeWarning
        pc = lm.compute_covariance(np.array([[1e308]]), cost, n, p, True)
        np.testing.assert_allclose(pc, [[1e-308]], rtol=1e-12)
        # 2x2 with entries 1e188..1e200, both directions kept (sqrt cond ~ 1e6 warns)
        warnings.simplefilter("ignore", UserWarning)
        H = np.array([[1e200, 1e190], [1e190, 1e188]])
        pc = lm.compute_covariance(H, cost, n, 2, True)
    s = 1e200
    expect = np.linalg.pinv(H / s, rcond=2 * np.finfo(float).eps) / s
    assert np.linalg.matrix_rank(H / s) == 2
    np.testing.assert_allclose(pc, expect, rtol=1e-9, atol=0.0)
    # tiny H (1e-300) is representable: the covariance 1e300 stays finite
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pc = lm.compute_covariance(np.array([[1e-300]]), cost, n, p, True)
    np.testing.assert_allclose(pc, [[1e300]], rtol=1e-12)
    # a covariance beyond float64 range is still the documented all-inf + OptimizeWarning
    with pytest.warns(OptimizeWarning, match=COV_WARN):
        pc = lm.compute_covariance(np.array([[1e-310]]), cost, n, p, True)
    assert np.isposinf(pc).all()


def _reference_covariance(H, cost, n, p, absolute_sigma):
    """Unscaled reference (the pre-scaling implementation) for finite, moderate H."""
    w, V = np.linalg.eigh((H + H.T) / 2)
    keep = w > p * np.finfo(np.float64).eps * w[-1]
    cov = (V[:, keep] / w[keep]) @ V[:, keep].T
    return cov if absolute_sigma else cov * (2.0 * cost / (n - p))


def test_compute_covariance_matches_unscaled_reference():
    # H of an arbitrary non-power-of-two magnitude, cond up to 1e8: scaling must be exact
    rng = np.random.default_rng(0)
    p, n, cost = 4, 50, 1.7
    for k in range(10):
        Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        ev = np.logspace(0, rng.uniform(1, 8), p)
        H = (Q * ev) @ Q.T * rng.uniform(0.3, 7.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # ill-conditioned notice
            got = lm.compute_covariance(H, cost, n, p, False)
        np.testing.assert_allclose(
            got, _reference_covariance(H, cost, n, p, False), rtol=1e-12, atol=0.0
        )


def test_compute_covariance_cond_overflow_no_runtime_warning():
    # lam_max / lam_min = 1e320 overflows float64: still the documented cond UserWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.warns(
            UserWarning,
            match=r"^normal-equation Hessian is ill-conditioned \(sqrt\(cond\(H\)\) > "
            r"1e\+04\); the covariance may be inaccurate and the parameters may stop "
            r"short of the optimum: check the result$",
        ):
            lm.compute_covariance(np.diag([1.0, 1e-320]), 0.4, 10, 2, False)


# ---- Task 8: 1-D sigma (weights) and weighted covariance -------------------------------


def sig_of(x):
    """Heteroscedastic sigma = 0.02 + 0.1 * x (x shifted to start at 0)."""
    return 0.02 + 0.1 * (x - x[0])


def weighted_data(prob, n=N, seed=0):
    x, _ = prob.data(n)
    s = sig_of(x)
    y = prob.model(np, x, *prob.p_true) + s * np.random.default_rng(
        seed
    ).standard_normal(n)
    return x, y, s


@pytest.mark.parametrize("name", ["exp_decay", "two_exp"])
def test_weighted_matches_scipy(name):
    prob = PROBLEMS[name]
    x, y, s = weighted_data(prob)
    ref = least_squares(lambda th: (y - prob.model(np, x, *th)) / s, prob.p0, **TIGHT)
    out = run(model_of(prob), x, y, prob.p0, sigma=s, **TIGHT)
    assert out["success"]
    rtol = 1e-4 if name == "two_exp" else 1e-6
    assert_close(by_rate(prob, out["x"]), by_rate(prob, ref.x), rtol, 1e-8)
    assert_close(out["cost"], ref.cost, 1e-9)
    # the weights matter: the unweighted optimum is far from the weighted one
    plain = least_squares(lambda th: y - prob.model(np, x, *th), prob.p0, **TIGHT)
    assert np.max(np.abs(plain.x - ref.x) / np.abs(ref.x)) > 1e-3


def test_weighted_chunk_invariance():
    x, y, s = weighted_data(EXP)
    outs = [
        run(exp_model, x, y, EXP.p0, sigma=s, chunk_size=c)
        for c in (N, N // 2, 512, 513, 7)
    ]
    for o in outs[1:]:
        assert_close(o["x"], outs[0]["x"], 1e-9)
        assert_close(o["cost"], outs[0]["cost"], 1e-12)


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_weighted_pcov_matches_curve_fit(absolute_sigma):
    x, y, s = weighted_data(EXP)
    out = run(exp_model, x, y, EXP.p0, sigma=s, absolute_sigma=absolute_sigma, **TIGHT)
    _, ref = curve_fit(
        exp_model,
        x,
        y,
        p0=EXP.p0,
        sigma=s,
        absolute_sigma=absolute_sigma,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    assert out["success"]
    atol = 1e-12 * np.abs(ref).max()
    np.testing.assert_allclose(out["pcov"], ref, rtol=1e-6, atol=atol)
    # and it differs from the unweighted covariance
    plain = run(exp_model, x, y, EXP.p0, absolute_sigma=absolute_sigma, **TIGHT)
    assert not np.allclose(plain["pcov"], ref, rtol=1e-2)


def test_constant_sigma_scales_cost_and_absolute_pcov(data):
    c = 0.05
    plain = {
        a: run(exp_model, *data, EXP.p0, absolute_sigma=a, **TIGHT) for a in (0, 1)
    }
    w = {
        a: run(exp_model, *data, EXP.p0, sigma=np.full(N, c), absolute_sigma=a, **TIGHT)
        for a in (0, 1)
    }
    for a in (0, 1):
        assert_close(w[a]["x"], plain[a]["x"], 1e-6, 1e-8)
        assert_close(w[a]["cost"], plain[a]["cost"] / c**2, 1e-9)
    # relative sigma: pcov is invariant to a constant sigma; absolute: scales by c^2
    np.testing.assert_allclose(w[0]["pcov"], plain[0]["pcov"], rtol=1e-6)
    np.testing.assert_allclose(w[1]["pcov"], c**2 * plain[1]["pcov"], rtol=1e-6)


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf, -1.0])
def test_masked_rows_ignore_sigma(bad):
    x, y, s = weighted_data(EXP, n=64)
    mask = jnp.arange(64) < 40
    stats = lm.make_chunk_stats(exp_model, weighted=True)
    s_bad = s.copy()
    s_bad[40:] = bad
    got = stats(jnp.asarray(EXP.p0), x, y, s_bad, mask)
    want = stats(jnp.asarray(EXP.p0), x[:40], y[:40], s[:40], jnp.ones(40, bool))
    for a, b in zip(got, want, strict=True):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=1e-12)
    # and the weight is applied to both r and J: F = 0.5 sum((y - f)/s)^2, H = J_w^T J_w
    f = EXP.model(np, x[:40], *EXP.p0)
    assert_close(float(want[0]), 0.5 * np.sum(((y[:40] - f) / s[:40]) ** 2), 1e-12)
    J = jax.jacfwd(lambda q: EXP.model(jnp, x[:40], *q))(jnp.asarray(EXP.p0))
    Jw = np.asarray(J) / s[:40, None]
    assert_close(np.asarray(want[2]), Jw.T @ Jw, 1e-12)


def test_jax_and_list_sigma_equal_numpy_sigma():
    x, y, s = weighted_data(EXP)
    ref = run(exp_model, x, y, EXP.p0, sigma=s)
    for sg in (jnp.asarray(s), s.tolist()):
        out = run(exp_model, x, y, EXP.p0, sigma=sg)
        assert_close(out["x"], ref["x"], 1e-12)
        assert out["nfev"] == ref["nfev"]
    out = run(exp_model, jnp.asarray(x), jnp.asarray(y), EXP.p0, sigma=jnp.asarray(s))
    assert_close(out["x"], ref["x"], 1e-12)


def test_cpu_sigma_chunks_are_views(data):
    x, y = data
    s = np.ones(N)
    sc = next(lm.iter_blocks(x, y, s, 1000))[2]
    assert np.shares_memory(sc, s)
    *_, sc_tail, mask_tail = list(lm.iter_blocks(x, y, s, 1024))[-1]
    assert sc_tail.shape == (1024,) and not bool(mask_tail.all())


@pytest.mark.parametrize(
    ("make", "pattern"),
    [
        (lambda s: s[:-1], r"^sigma has length 2999, expected 3000"),
        (lambda s: np.where(np.arange(N) == 5, 0.0, s), r"^sigma must be positive"),
        (lambda s: np.where(np.arange(N) == 2999, -s, s), r"^sigma must be positive"),
        (lambda s: np.where(np.arange(N) == 7, np.nan, s), r"^sigma must be finite"),
        (lambda s: np.where(np.arange(N) == 2999, np.inf, s), r"^sigma must be finite"),
        (np.diag, r"^sigma must be 1-D"),
    ],
)
@pytest.mark.parametrize("check_finite", [True, False])
def test_bad_sigma_raises(data, make, pattern, check_finite):
    s = make(np.full(N, 0.05))
    with pytest.raises(ValueError, match=pattern):
        run(exp_model, *data, EXP.p0, sigma=s, check_finite=check_finite)


# --- Task 9: box bounds by active set, KKT at the returned point ----------------------

BEXP = PROBLEMS["bounded_exp"]
BLO, BHI = (np.asarray(b, dtype=np.float64) for b in BEXP.bounds)


def scipy_bounded(prob, x, y, p0, lo, hi):
    return least_squares(
        lambda th: y - prob.model(np, x, *th),
        p0,
        bounds=(lo, hi),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )


def snap_to_box(theta, lo, hi, tol=1e-9):
    """SciPy's TRF stays strictly inside the box: put near-bound coordinates on it."""
    t = np.array(theta, dtype=np.float64)
    for b in (lo, hi):
        near = np.isfinite(b) & (np.abs(t - b) <= tol * (1 + np.abs(b)))
        t[near] = b[near]
    return t


def test_bounded_exp_matches_scipy_trf():
    x, y = BEXP.data(N)
    model = model_of(BEXP)
    ref = scipy_bounded(BEXP, x, y, BEXP.p0, BLO, BHI)
    out = run(model, x, y, BEXP.p0, bounds=BEXP.bounds, **TIGHT)
    assert out["success"]
    assert out["x"][0] == BHI[0]  # premise: the upper bound on `a` is active
    assert_close(out["cost"], ref.cost, 1e-6)
    assert_close(out["x"], ref.x, 1e-4)
    assert_kkt(out, model, x, y, BLO, BHI)


def test_weighted_bounded_matches_scipy_trf():
    x, y, s = weighted_data(BEXP)
    model = model_of(BEXP)
    ref = least_squares(
        lambda th: (y - BEXP.model(np, x, *th)) / s,
        BEXP.p0,
        bounds=(BLO, BHI),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    out = run(model, x, y, BEXP.p0, sigma=s, bounds=BEXP.bounds, **TIGHT)
    assert out["success"]
    assert out["x"][0] == BHI[0]  # premise: the upper bound on `a` is active
    assert_close(out["cost"], ref.cost, 1e-6)
    assert_close(out["x"], ref.x, 1e-4)
    assert_kkt(out, model, x, y, BLO, BHI, sigma=s)


def _random_box(seed):
    """Box around a random truth; ``seed % 4`` parameters get a bound that excludes it."""
    rng = np.random.default_rng(seed)
    p_true = np.array(
        [rng.uniform(1.0, 4.0), rng.uniform(0.5, 2.0), rng.uniform(-1.0, 1.0)]
    )
    x = np.linspace(0.0, 4.0, 2000)
    y = EXP.model(np, x, *p_true) + 0.05 * rng.standard_normal(x.size)
    lo, hi = p_true - rng.uniform(0.5, 2.0, 3), p_true + rng.uniform(0.5, 2.0, 3)
    for i in rng.permutation(3)[: seed % 4]:
        if rng.random() < 0.5:
            hi[i] = p_true[i] - rng.uniform(0.05, 0.3)
        else:
            lo[i] = p_true[i] + rng.uniform(0.05, 0.3)
    lo[1] = max(lo[1], 0.1)  # a negative rate opens other local minima (growth fits)
    p0 = rng.uniform(lo, hi)
    return x, y, lo, hi, p0


def test_randomized_bounded_kkt():
    on_bound = set()
    for seed in range(48):  # seed 42 stalls at the optimum without the box certificate
        x, y, lo, hi, p0 = _random_box(seed)
        ref = scipy_bounded(EXP, x, y, p0, lo, hi)
        out = run(exp_model, x, y, p0, bounds=(lo, hi), **TIGHT)
        assert out["success"], (seed, out["termination"], out["message"])
        assert_close(out["cost"], ref.cost, 1e-6)
        assert_close(out["x"], ref.x, 1e-4, 1e-8)
        assert_kkt(out, exp_model, x, y, lo, hi)
        on_bound.add(int(np.sum((out["x"] == lo) | (out["x"] == hi))))
    assert {0, 1, 3} <= on_bound, on_bound  # none, one and every parameter on a bound


def test_free_set_both_signs():
    lo, hi = np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0])
    theta = np.array([0.0, 0.0, 1.0, 1.0])
    # g = J^T r is the descent direction: g<0 pushes below lb, g>0 pushes above ub
    g = np.array([-1.0, 1.0, 1.0, -1.0])
    np.testing.assert_array_equal(
        lm.free_set(theta, g, lo, hi), [False, True, False, True]
    )
    # g == 0 at a bound is held; interior points are always free
    np.testing.assert_array_equal(
        lm.free_set(np.array([0.0, 1.0, 0.5]), np.zeros(3), lo[:3], hi[:3]),
        [False, False, True],
    )


def test_start_on_bound():
    x, y = BEXP.data(N)
    p0 = np.array([BHI[0], BLO[1], 0.3])  # a on its (active) upper bound, b on lb = 0
    ref = scipy_bounded(BEXP, x, y, p0, BLO, BHI)
    out = run(model_of(BEXP), x, y, p0, bounds=BEXP.bounds, **TIGHT)
    assert out["success"]
    assert_close(out["cost"], ref.cost, 1e-6)
    assert_kkt(out, model_of(BEXP), x, y, BLO, BHI)


def nan_outside_box_model(x, a, b, c):
    """bounded_exp model that is NaN off the box: an unclipped p0 cannot be evaluated."""
    ok = (a >= 0) & (a <= 2) & (b >= 0) & (b <= 3) & (c >= 0) & (c <= 1)
    return jnp.where(ok, BEXP.model(jnp, x, a, b, c), jnp.nan)


def test_start_outside_box_is_clipped():
    x, y = BEXP.data(N)
    p0 = np.array([5.0, -1.0, 0.5])
    # at the clipped b = 0 the a and c columns coincide: singular H, warned covariance
    with pytest.warns(UserWarning, match="ill-conditioned"):
        one = run(nan_outside_box_model, x, y, p0, bounds=BEXP.bounds, max_nfev=1)
    np.testing.assert_array_equal(one["x"], np.clip(p0, BLO, BHI))
    out = run(nan_outside_box_model, x, y, p0, bounds=BEXP.bounds, **TIGHT)
    assert out["success"]
    assert_kkt(out, model_of(BEXP), x, y, BLO, BHI)
    ref = scipy_bounded(BEXP, x, y, np.clip(p0, BLO, BHI), BLO, BHI)
    assert_close(out["cost"], ref.cost, 1e-6)


def test_bound_cutting_flat_valley():
    prob = PROBLEMS["two_exp"]  # flat valley, kappa(J) ~ 4e2
    x, y = prob.data(N)
    lo = np.array([0.0, 0.0, 0.0, 1.2])
    hi = np.array([np.inf, 1.4, np.inf, np.inf])  # cuts between k1 = 1.0 and k2 = 1.5
    p0 = np.array([1.0, 0.5, 1.0, 2.0])
    ref = scipy_bounded(prob, x, y, p0, lo, hi)
    out = run(model_of(prob), x, y, p0, bounds=(lo, hi), **TIGHT)
    assert out["success"], out["message"]
    assert_close(out["cost"], ref.cost, 1e-6)
    assert_close(out["x"], ref.x, 1e-4, 1e-8)
    assert_kkt(out, model_of(prob), x, y, lo, hi)


# random boxes 21 and 23 end with one and two parameters on a bound; without the
# constrained model-decrease certificate both stall at the optimum (gradient noise > gtol)
@pytest.mark.parametrize(("case", "n_active"), [(None, 1), (21, 1), (23, 2)])
@pytest.mark.parametrize("tight", [False, True])
def test_bounded_start_at_constrained_optimum_succeeds(case, n_active, tight):
    if case is None:
        x, y = BEXP.data(N)
        lo, hi, model, prob, p0 = BLO, BHI, model_of(BEXP), BEXP, BEXP.p0
    else:
        x, y, lo, hi, p0 = _random_box(case)
        model, prob = exp_model, EXP
    start = snap_to_box(scipy_bounded(prob, x, y, p0, lo, hi).x, lo, hi)
    assert np.sum((start == lo) | (start == hi)) == n_active  # premise: on the bounds
    out = run(model, x, y, start, bounds=(lo, hi), **(TIGHT if tight else {}))
    assert out["success"], (out["termination"], out["nfev"])
    assert_kkt(out, model, x, y, lo, hi)


def test_stationary_does_not_certify_from_free_subspace():
    """A held variable whose release pays: free-subspace decrement 2, feasible gain >> 2."""
    H = np.array([[1.0, -0.9], [-0.9, 1.0]])
    g = np.array([0.0, 2.0])  # var 0 at its lower bound with g <= 0: held
    theta, lo, hi = np.zeros(2), np.array([0.0, -np.inf]), np.full(2, np.inf)
    free = lm.free_set(theta, g, lo, hi)
    assert free.tolist() == [False, True]
    # free-subspace decrement 0.5*g1^2/H11 = 2; the coupled step s = (9.47, 10.5) gains 10.5
    F = 3.0 / 1e-8  # ftol*F = 3 sits between the two
    assert lm.newton_certificate(g[free], H[np.ix_(free, free)], F, 1e-8)  # the trap
    assert lm._stationary(F, g, H, free, theta, lo, hi, 1e-8, 1e-8) is None
    # a tiny free gradient (above gtol) and an outward held one: the box certificate fires
    g0 = np.array([-1.0, 1e-9])
    assert lm._stationary(F, g0, H, free, theta, lo, hi, 1e-12, 1e-8) == "ftol"


@pytest.mark.parametrize(
    ("bounds", "pattern"),
    [
        ((1.0, 0.0), r"^each lower bound must be strictly less"),
        ((0.0, 0.0), r"^each lower bound must be strictly less"),
        (
            ([0.0, 0.0, 0.0], [1.0, 0.0, 1.0]),
            r"^each lower bound must be strictly less",
        ),
        ((np.inf, -np.inf), r"^each lower bound must be strictly less"),
        ((np.nan, np.inf), r"^each lower bound must be strictly less"),
        (([0.0, 0.0], [1.0, 1.0, 1.0]), r"^lower bound has shape \(2,\), expected"),
        ((0.0, np.ones((3, 1))), r"^upper bound has shape \(3, 1\), expected"),
    ],
)
def test_lb_ge_ub_raises(data, bounds, pattern):
    with pytest.raises(ValueError, match=pattern):
        run(exp_model, *data, EXP.p0, bounds=bounds)


# --- Task 9 fix round 1: box scaling, BVLS guard, gating on any variable at a bound ----

# H = D^1/2 [[1, .5], [.5, 1]] D^1/2 with D = diag(100, 0.01); unconstrained s = (0.2, 0)
BOX_H = np.array([[100.0, 0.5], [0.5, 0.01]])
BOX_G = BOX_H @ np.array([0.2, 0.0])  # (20, 0.1)
BOX_THETA, BOX_LO, BOX_HI = (
    np.zeros(2),
    np.array([0.0, -np.inf]),
    np.array([0.1, np.inf]),
)
# the far bound s0 <= 0.1 binds: s1 = (g1 - H10*0.1)/H11 = 5, gain 2 - 0.5 + 0.125
BOX_GAIN = 0.1 * 20.0 - 0.5 * 100.0 * 0.1**2 + 0.5 * (0.1 - 0.5 * 0.1) ** 2 / 0.01


def test_box_certificate_scaled_box_closed_form():
    assert pytest.approx(1.625) == BOX_GAIN
    ftol = 1e-8
    for factor, want in ((1.01, True), (0.99, False)):
        F = factor * BOX_GAIN / ftol
        got = lm.box_certificate(BOX_G, BOX_H, F, ftol, BOX_THETA, BOX_LO, BOX_HI)
        assert got is want, factor


def test_box_certificate_bvls_failure_is_no_certificate(monkeypatch):
    from types import SimpleNamespace

    F = 1.01 * BOX_GAIN / 1e-8  # the true gain certifies; a failed solve must not
    F_no = 0.5 * BOX_GAIN / 1e-8  # the true gain does not certify
    args = (BOX_G, BOX_H)
    box = (BOX_THETA, BOX_LO, BOX_HI)

    def boom(*a, **k):
        raise ValueError("bvls failed")

    monkeypatch.setattr(lm, "lsq_linear", boom)
    assert lm.box_certificate(*args, F, 1e-8, *box) is False
    # max_iter reached: a feasible but suboptimal u (zero step, gain 0) would certify
    monkeypatch.setattr(
        lm, "lsq_linear", lambda *a, **k: SimpleNamespace(status=0, x=np.zeros(2))
    )
    assert lm.box_certificate(*args, F_no, 1e-8, *box) is False


def test_stationary_box_certificate_when_at_bound_but_free():
    """On a bound with an inward gradient (free): the box certificate must still run."""
    H = np.array([[1.0, 0.999], [0.999, 1.0]])
    g = np.array([1e-12, 1e-4])
    theta, lo, hi = np.zeros(2), np.array([0.0, -np.inf]), np.full(2, np.inf)
    free = lm.free_set(theta, g, lo, hi)
    assert free.all()  # premise: var 0 sits on lb but g points inward
    assert not lm.newton_certificate(
        g, H, 1.0, 1e-8
    )  # unconstrained step leaves the box
    assert lm._stationary(1.0, g, H, free, theta, lo, hi, 1e-8, 1e-8) == "ftol"


# --- Task 9 fix round 2: scale-safe box certificate with a dual-residual bound ---------

# Codex reproducer: near-exact scale (F = 1e-24); the feasible step (0, 8e-21) gains 32% of F
CX_H = 1e16 * np.array([[1.0, -0.9], [-0.9, 1.0]])
CX_G = 1e-4 * np.array([-1.1, 0.8])
CX_THETA, CX_LO, CX_HI = np.zeros(2), np.zeros(2), np.ones(2)


def box_qp_max(g, H, theta, lo, hi):
    """Exact max of s^T g - s^T H s / 2 on the box, by enumerating the 3^p active sets."""
    import itertools

    p, best = len(g), 0.0
    for pattern in itertools.product((-1, 0, 1), repeat=p):
        s = np.zeros(p)
        fixed = np.array(pattern) != 0
        s[np.array(pattern) == -1] = (lo - theta)[np.array(pattern) == -1]
        s[np.array(pattern) == 1] = (hi - theta)[np.array(pattern) == 1]
        if not np.isfinite(s).all():
            continue
        fr = ~fixed
        if fr.any():
            rhs = g[fr] - H[np.ix_(fr, ~fr)] @ s[~fr]
            s[fr] = np.linalg.solve(H[np.ix_(fr, fr)], rhs)
        if np.all(theta + s >= lo) and np.all(theta + s <= hi):
            best = max(best, s @ g - 0.5 * s @ H @ s)
    return best


def test_box_certificate_codex_near_exact_reproducer():
    gain = box_qp_max(CX_G, CX_H, CX_THETA, CX_LO, CX_HI)
    assert gain == pytest.approx(3.2e-25, rel=1e-9)  # 32% of F
    got = lm.box_certificate(CX_G, CX_H, 1e-24, 1e-8, CX_THETA, CX_LO, CX_HI)
    assert got is False


@pytest.mark.parametrize("k", [1.0, 1e-8, 1e-16, 1e-24])
def test_box_certificate_decision_is_scale_invariant(k):
    gain = box_qp_max(CX_G, CX_H, CX_THETA, CX_LO, CX_HI)  # 3.2e-25 at k = 1
    ftol = 1e-8
    # g -> k g scales every gain by k^2; F scales with it, so the decision must not move
    for factor, want in ((1.01, True), (0.99, False)):
        F = factor * gain / ftol * k * k
        got = lm.box_certificate(k * CX_G, CX_H, F, ftol, CX_THETA, CX_LO, CX_HI)
        assert got is want, (k, factor)


def test_box_certificate_does_not_trust_solver_status(monkeypatch):
    from types import SimpleNamespace

    # a "successful" solve that returns the feasible but suboptimal start (gain 0)
    monkeypatch.setattr(
        lm, "lsq_linear", lambda *a, **k: SimpleNamespace(status=1, x=np.zeros(2))
    )
    assert lm.box_certificate(CX_G, CX_H, 1e-24, 1e-8, CX_THETA, CX_LO, CX_HI) is False


def test_near_exact_bounded_fit_does_not_stop_with_gain_left():
    """F ~ 1e-24 bounded linear fit: one pass must not certify a 32% cost reduction away."""
    cols = 1e8 * np.array([[1.0, 0.0, 0.0, 0.0], [-0.9, np.sqrt(0.19), 0.0, 0.0]])
    c1, c2 = jnp.asarray(cols[0]), jnp.asarray(cols[1])

    def model(x, a, b):
        i = x.astype(jnp.int32)
        return a * jnp.take(c1, i) + b * jnp.take(c2, i)

    x = np.arange(4.0)
    # unconstrained optimum (-2e-20, -1e-20) lies outside the box, so at theta = 0 the
    # gradient is the reproducer's g = 1e-4*(-1.1, 0.8); e is orthogonal to both columns
    e = np.array([0.0, 0.0, np.sqrt(6e-25), 0.0])
    y = cols.T @ np.array([-2e-20, -1e-20]) + e
    F0 = 0.5 * float(y @ y)
    st = lm.run_pass(lm.make_chunk_stats(model), np.zeros(2), (x, y, None), 4)
    np.testing.assert_allclose(st.g, CX_G, rtol=1e-9)
    gain = box_qp_max(
        st.g, st.H, CX_THETA, CX_LO, CX_HI
    )  # constrained optimum (0, 8e-21)
    F_opt = F0 - gain
    assert pytest.approx(1e-24, rel=1e-9) == F0 and F_opt / F0 == pytest.approx(0.68)
    out = run(model, x, y, [0.0, 0.0], bounds=([0.0, 0.0], [1.0, 1.0]))
    assert out["nfev"] > 1
    assert out["cost"] <= F_opt * (1 + 1e-6), (out["termination"], out["cost"])


def test_assert_kkt_rejects_undefined_objective():
    def sqrt_model(x, a):
        return jnp.sqrt(a) * jnp.ones_like(x)

    x = np.linspace(0.0, 1.0, 10)
    with pytest.raises(AssertionError, match="not finite"):
        assert_kkt({"x": np.array([-1.0])}, sqrt_model, x, np.ones(10), -2.0, 2.0)


# ---- TSQR covariance for ill-conditioned exact fits -------------------------------------

two_exp = lambda x, a1, k1, a2, k2: a1 * jnp.exp(-k1 * x) + a2 * jnp.exp(-k2 * x)
exp_decay = lambda x, a, b, c: a * jnp.exp(-b * x) + c
x = np.linspace(0.0, 5.0, 3001)
noise = 1e-3 * np.random.default_rng(0).standard_normal(x.size)
y_exp = np.asarray(exp_decay(x, 2.5, 1.3, 0.5)) + noise
# cond(J) ~ 8e6 here, ~2e5 at the fitted popt. The plan's original rate 1.0001 is
# numerically rank-deficient at popt; see test_rank_deficient_original_input_regression.
TRUE_ILL = (2.0, 1.0, 1.5, 1.02)
y_ill = np.asarray(two_exp(x, *TRUE_ILL)) + noise
P0_ILL = [2.0, 0.9, 1.5, 1.1]


def full_jac(model, x, theta, sigma=None):
    J = np.asarray(
        jax.jacfwd(lambda q: model(jnp.asarray(x), *q))(jnp.asarray(theta, float))
    )
    return J if sigma is None else J / np.asarray(sigma)[:, None]


def factory(xs, ys):
    return lambda cs: ((xs[i : i + cs], ys[i : i + cs]) for i in range(0, xs.size, cs))


@pytest.fixture
def spy_tsqr(monkeypatch):
    real = lm.tsqr_r_factor
    spy = types.SimpleNamespace(calls=0)

    def wrapper(*a, **k):
        spy.calls += 1
        return real(*a, **k)

    monkeypatch.setattr(lm, "tsqr_r_factor", wrapper)
    return spy


def _sv(A):
    return np.linalg.svd(np.asarray(A), compute_uv=False)


@pytest.mark.parametrize("weighted", [False, True])
def test_tsqr_r_factor_matches_full_jacobian(weighted):
    sigma = 0.5 + np.random.default_rng(1).random(x.size) if weighted else None
    theta = np.array(TRUE_ILL)
    src, w = lm._open_source(
        x, y_ill, sigma, 700, 4, None, None
    )  # 3001 = 4*700 + 201: padded tail
    R = lm.tsqr_r_factor(two_exp, theta, src, 700, weighted=w)
    ref = _sv(full_jac(two_exp, x, theta, sigma))
    # backward-stable QR and SVD fix each singular value to O(eps * s_max) absolutely
    # (Weyl), so a relative 1e-12 holds only for s_i >~ 1e-4 * s_max
    np.testing.assert_allclose(
        _sv(R), ref, rtol=1e-12, atol=64 * np.finfo(float).eps * ref[0]
    )


def test_tsqr_r_factor_factory_replay():
    theta = np.array(TRUE_ILL)
    src_a, _ = lm._open_source(x, y_ill, None, 700, 4, None, None)
    src_f, _ = lm._open_source(factory(x, y_ill), None, None, 700, 4, None, x.size)
    R_a = lm.tsqr_r_factor(two_exp, theta, src_a, 700, weighted=False)
    R_f = lm.tsqr_r_factor(two_exp, theta, src_f, 700, weighted=False)
    np.testing.assert_allclose(_sv(R_f), _sv(R_a), rtol=1e-13)


def test_tsqr_pass_detects_changed_replay_count():
    state = {"short": False}  # flips the factory to n-1 points for later passes

    def fac(cs):
        m = x.size - 1 if state["short"] else x.size
        return (
            (x[i : min(i + cs, m)], y_ill[i : min(i + cs, m)]) for i in range(0, m, cs)
        )

    src, _ = lm._open_source(fac, None, None, 700, 4, None, x.size)
    lm.tsqr_r_factor(
        two_exp, np.array(TRUE_ILL), src, 700, weighted=False
    )  # full count: ok
    state["short"] = True
    with pytest.raises(ValueError, match="points"):
        lm.tsqr_r_factor(two_exp, np.array(TRUE_ILL), src, 700, weighted=False)


def test_declared_count_mismatch_raises():
    with pytest.raises(ValueError, match="n_points"):
        lm.exact_lm_fit(
            two_exp,
            factory(x, y_ill),
            None,
            P0_ILL,
            chunk_size=700,
            n_points=x.size + 1,
        )


def test_tsqr_r_factor_rank_deficient():
    dup = lambda x, a, b, k: (
        a * jnp.exp(-k * x) + b * jnp.exp(-k * x)
    )  # a, b not separable
    theta = np.array([1.0, 1.0, 1.3])
    src, _ = lm._open_source(x, y_exp, None, 700, 3, None, None)
    R = lm.tsqr_r_factor(dup, theta, src, 700, weighted=False)
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        got, _ = pcov_from_jacobian(R, 0.5, x.size, 3, False, _covariance_svd)
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        ref, _ = pcov_from_jacobian(
            full_jac(dup, x, theta), 0.5, x.size, 3, False, _covariance_svd
        )
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-12 * np.abs(ref).max())


@pytest.mark.filterwarnings(
    "ignore:normal-equation Hessian is ill-conditioned:UserWarning"
)
def test_ill_conditioned_exact_uses_tsqr(spy_tsqr):
    res = lm.exact_lm_fit(two_exp, x, y_ill, P0_ILL, chunk_size=1000, **TIGHT)
    assert res.success
    ref, _ = pcov_from_jacobian(
        full_jac(two_exp, x, res.x), res.cost, x.size, 4, False, _covariance_svd
    )
    d = np.sqrt(np.outer(np.diag(ref), np.diag(ref)))
    assert res.covariance_method == "tsqr" and spy_tsqr.calls == 1
    assert np.max(np.abs(res.pcov - ref) / d) <= 1e-8


def test_well_conditioned_exact_skips_tsqr(spy_tsqr):
    res = lm.exact_lm_fit(exp_decay, x, y_exp, [1.0, 1.0, 0.0], chunk_size=1000)
    assert res.covariance_method == "normal" and spy_tsqr.calls == 0


def test_skip_never_calls_tsqr(spy_tsqr):
    res = lm.exact_lm_fit(
        two_exp, x, y_ill, P0_ILL, chunk_size=1000, compute_covariance=False, **TIGHT
    )
    assert (
        res.covariance_method == "skipped" and res.pcov is None and spy_tsqr.calls == 0
    )


@pytest.mark.filterwarnings(
    "ignore:normal-equation Hessian is ill-conditioned:UserWarning"
)
def test_failed_fit_still_reports_route(spy_tsqr):
    res = lm.exact_lm_fit(two_exp, x, y_ill, P0_ILL, chunk_size=1000, max_nfev=2)
    assert not res.success and res.covariance_method in ("normal", "tsqr")


def test_tsqr_route_warns_when_kappa_above_cond_warn():
    y2 = np.asarray(two_exp(x, 2.0, 1.0, 1.5, 1.00001)) + noise
    with pytest.warns(UserWarning, match="computed by QR"):
        lm.exact_lm_fit(two_exp, x, y2, P0_ILL, chunk_size=1000, **TIGHT)


def test_pcov_from_normal_warning_text_unchanged():
    with pytest.warns(UserWarning, match="the covariance may be inaccurate"):
        pcov_from_normal(np.diag([1.0, 1e-10]), 0.5, 10, 2, False)


@pytest.mark.parametrize(
    "H,expected",
    [
        (np.diag([1e300, 1e300]), 1.0),  # extreme but well-conditioned: no overflow
        (np.diag([4.0, 1e-6]), 2000.0),
        (np.diag([1.0, 0.0]), np.inf),
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), np.inf),
    ],
)
def test_normal_condition(H, expected):
    assert normal_condition(H) == pytest.approx(expected, rel=1e-12)


def _retained_svd(A, n):
    """Singular values kept by pcov_from_jacobian: s > eps * max(n, p) * s[0]."""
    s = np.asarray(_covariance_svd(np.asarray(A))[0])
    return int(np.sum(s > np.finfo(float).eps * max(n, A.shape[1]) * s[0]))


@pytest.mark.filterwarnings(
    "ignore:normal-equation Hessian is ill-conditioned:UserWarning"
)
def test_rank_deficient_original_input_regression():
    """The plan's original ill-conditioned input: second rate 1.0001, 3001 points, seed 0.

    The fit converges onto k1 == k2, where J is numerically rank-deficient (cond ~ 2e15).
    A pseudo-inverse truncated at the cutoff is ill-posed there: two backward-stable
    factorizations (the full-J reference included) disagree at the 1e-2 level, so no
    elementwise pcov tolerance is a sound criterion. Criterion instead:
    (a) covariance_method == "tsqr";
    (b) the rank-deficiency OptimizeWarning is emitted, as for the full-J reference;
    (c) the retained singular-value count of the TSQR R equals that of the full-J
        reference, both with the production cutoff eps * max(n, p) * s[0];
    (d) pcov is finite wherever the reference's is finite;
    (e) on the well-determined subspace ``V_k`` (reference right singular vectors with
        ``s_i >= 1e-6 * s_0``) the covariance of the TSQR ``R`` agrees with the reference.
        First order: either factorization is the exact one of ``J + dJ`` with
        ``||dJ|| <= c * eps * s_0``, ``c = n * p`` (Householder QR backward error,
        Higham Thm 19.4, worst case). A kept direction moves by ``dJ / s_i`` (relative
        ``1/s_i^2`` change ``2 c eps s_0 / s_i``, bounded by ``c eps (s_0/s_k)^2``), and the
        retained near-cutoff direction ``s_r`` (here 1.45e-6, just above the cutoff but
        below ``1e-6 s_0``) rotates into ``V_k`` by ``~ c eps s_0 / s_k``, adding
        ``(c eps s_0 / s_r)^2`` relative. Tolerance: the sum, scale-aware entrywise. Both
        sides are formed from their factors (``s``, ``V``) so no cancellation enters;
    (f) the stored ``pcov`` projected on ``V_k`` is positive definite (symmetric to the floor below) with each
        diagonal entry within a factor 2 of the reference's (an all-zero or garbage pcov
        fails). Its entrywise agreement is NOT asserted: the stored matrix is dominated by
        ``sigma^2 / s_r^2``, so its rounding alone is ``~ p^2 eps (s_0/s_r)^2`` (2.2 here)
        relative to the ``V_k`` block; the sensitivity of the near-cutoff direction is
        tolerated, not tested. Measured: factored 3.7e-15 (tolerance 4.5e-9), stored 2.3e-2.
    """
    y = np.asarray(two_exp(x, 2.0, 1.0, 1.5, 1.0001)) + noise
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        res = lm.exact_lm_fit(two_exp, x, y, P0_ILL, chunk_size=1000, **TIGHT)
    assert res.covariance_method == "tsqr"
    J = full_jac(two_exp, x, res.x)
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        ref, _ = pcov_from_jacobian(J, res.cost, x.size, 4, False, _covariance_svd)
    src, w = lm._open_source(x, y, None, 1000, 4, None, None)
    R = lm.tsqr_r_factor(two_exp, res.x, src, 1000, weighted=w)
    assert _retained_svd(R, x.size) == _retained_svd(J, x.size) < 4
    assert np.isfinite(res.pcov[np.isfinite(ref)]).all()

    n, p, eps = x.size, 4, np.finfo(float).eps
    s, VT = (np.asarray(a) for a in _covariance_svd(J))
    Vk = VT[s >= 1e-6 * s[0]].T
    k = Vk.shape[1]
    assert 1 <= k < 3  # the near-cutoff retained direction is outside V_k
    s_r = s[s > eps * max(n, p) * s[0]][-1]
    sig2 = 2 * res.cost / (n - p)

    def projected(
        A,
    ):  # sig2 * Vk^T pinv(A^T A) Vk from A's factors, cutoff as production
        sa, VTa = (np.asarray(a) for a in _covariance_svd(A))
        keep = sa > eps * max(n, p) * sa[0]
        W = Vk.T @ VTa[keep].T
        return sig2 * (W / sa[keep] ** 2) @ W.T

    got, want = projected(R), projected(J)
    c = n * p
    tol = c * eps * (s[0] / s[k - 1]) ** 2 + (c * eps * s[0] / s_r) ** 2
    d = np.sqrt(np.outer(np.diag(want), np.diag(want)))
    assert np.max(np.abs(got - want) / d) <= tol

    stored, ref_k = Vk.T @ res.pcov @ Vk, Vk.T @ ref @ Vk
    # symmetric up to the representation floor of the stored pcov (see (f))
    floor = p**2 * eps * np.abs(res.pcov).max()
    np.testing.assert_allclose(stored, stored.T, rtol=0, atol=floor)
    assert np.linalg.eigvalsh((stored + stored.T) / 2)[0] > 0
    ratio = np.diag(stored) / np.diag(ref_k)
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio


def test_tsqr_warning_names_non_finite_hessian():
    H = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.warns(UserWarning, match="non-finite") as rec:
        lm._warn_tsqr_route(H, normal_condition(H))
    assert not any("ill-conditioned" in str(r.message) for r in rec)


def test_tsqr_jacobian_kernel_cached_per_model_and_weighting():
    k = lm._chunk_jacobian(two_exp, False)
    assert lm._chunk_jacobian(two_exp, False) is k
    assert lm._chunk_jacobian(two_exp, True) is not k
    assert lm._chunk_jacobian(exp_decay, True) is not lm._chunk_jacobian(two_exp, True)


def test_tsqr_jacobian_kernel_cache_is_race_free(monkeypatch):
    """Another thread replacing the slot between this call's store and its return must
    not hand this call the other model's kernel (simulated deterministically)."""

    def a(x, q):
        return q * x

    def b(x, q):
        return 2.0 * q * x

    b_entry = (b, False, lm.make_chunk_jacobian(b))

    class Racy(list):
        def __setitem__(self, i, v):  # store, then "another thread" stores b
            super().__setitem__(i, v)
            super().__setitem__(0, b_entry)

    monkeypatch.setattr(lm, "_jac_kernel", Racy([None]))
    xs, mask = jnp.linspace(0.0, 1.0, 8), jnp.ones(8, dtype=bool)
    Ja = lm._chunk_jacobian(a, False)(jnp.array([3.0]), xs, mask)
    np.testing.assert_array_equal(np.asarray(Ja)[:, 0], np.asarray(xs))
    Jb = lm._chunk_jacobian(b, False)(jnp.array([3.0]), xs, mask)  # cache hit on b
    np.testing.assert_array_equal(np.asarray(Jb)[:, 0], 2.0 * np.asarray(xs))

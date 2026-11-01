"""Exact chunk-accumulated LM library module: termination reasons, pass budget, chunk size.

The same Stage 0 cases against the frozen prototype live in test_exact_lm_prototype.py.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares

import nlsq.streaming.exact_lm as lm
from benchmarks.compare.problems import PROBLEMS
from nlsq import curve_fit
from nlsq.result.optimize_warning import OptimizeWarning

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


def test_stage2_inputs_rejected(data):
    with pytest.raises(ValueError, match=r"finite bounds .*Stage 2"):
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

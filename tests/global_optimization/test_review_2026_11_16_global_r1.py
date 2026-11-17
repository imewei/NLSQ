"""Regression tests for the 2026-11-16 global_optimization review (round 1)."""

import warnings

import numpy as np
import pytest
from scipy.optimize import Bounds

import nlsq.global_optimization as go
from nlsq.global_optimization.bipop import BIPOPRestarter
from nlsq.global_optimization.cmaes_config import CMAESConfig
from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.global_optimization.multi_start import MultiStartOrchestrator
from nlsq.global_optimization.sampling import scale_samples_to_bounds
from nlsq.global_optimization.tournament import TournamentSelector


def _exp(x, a, b):
    import jax.numpy as jnp

    return a * jnp.exp(-b * x)


@pytest.fixture
def exp_data():
    x = np.linspace(0, 5, 20)
    return x, 3.0 * np.exp(-0.7 * x)


def test_n_starts_zero_accepts_ndarray_bounds(exp_data):
    x, y = exp_data
    orch = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=0))
    res = orch.fit(_exp, x, y, p0=[1, 1], bounds=np.array([[0, 0], [10, 5]]))
    np.testing.assert_allclose(res.popt, [3.0, 0.7], rtol=1e-5)


def test_multistart_accepts_scipy_bounds(exp_data):
    x, y = exp_data
    orch = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=3))
    res = orch.fit(_exp, x, y, p0=[1, 1], bounds=Bounds([0, 0], [10, 5]))
    np.testing.assert_allclose(res.popt, [3.0, 0.7], rtol=1e-5)


def test_cmaes_accepts_scipy_bounds(exp_data):
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

    x, y = exp_data
    opt = CMAESOptimizer(
        CMAESConfig(max_generations=5, restart_strategy="none", seed=1)
    )
    res = opt.fit(_exp, x, y, p0=[1.0, 1.0], bounds=Bounds([0, 0], [10, 5]))
    assert np.all(np.isfinite(res["popt"]))


def test_dir_lists_lazy_symbols():
    names = dir(go)
    for name in (
        "CMAESOptimizer",
        "CMAESPreempted",
        "BIPOPRestarter",
        "MethodSelector",
    ):
        assert name in names


def test_config_seed_controls_lhs_starts():
    lb, ub = np.zeros(2), np.ones(2) * 10

    def starts(seed):
        orch = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=4, seed=seed))
        return orch._generate_starting_points(2, lb, ub, None)

    np.testing.assert_array_equal(starts(5), starts(5))
    assert not np.allclose(starts(5), starts(6))


def test_config_seed_validated():
    with pytest.raises((TypeError, ValueError)):
        GlobalOptimizationConfig(seed=-1)
    assert GlobalOptimizationConfig(seed=3).to_dict()["seed"] == 3


def test_sobol_does_not_waste_a_start_on_the_corner():
    lb, ub = np.zeros(2), np.ones(2) * 10
    orch = MultiStartOrchestrator(
        GlobalOptimizationConfig(n_starts=8, sampler="sobol", center_on_p0=False)
    )
    pts = orch._generate_starting_points(2, lb, ub, None)
    assert not np.any(np.all(pts == lb, axis=1))


def test_tournament_weights_observations_not_batches():
    cfg = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=0, batches_per_round=2
    )
    sel = TournamentSelector(np.array([[1.0], [0.0]]), cfg)
    batches = iter([(np.zeros(100), np.zeros(100)), (np.zeros(1), np.ones(1))])
    best = sel.run_tournament(batches, lambda x, a: a + 0 * x)
    np.testing.assert_allclose(best[0], [0.0])


def test_tournament_uses_sigma():
    # Candidate 0 fits the noisy half badly, candidate 1 the precise half badly;
    # unweighted they tie-ish, weighted by sigma candidate 0 must win.
    cfg = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=0, batches_per_round=1
    )
    sel = TournamentSelector(np.array([[0.0], [1.0]]), cfg)
    x = np.zeros(2)
    y = np.array([0.0, 1.0])  # point0 -> a=0 exact, point1 -> a=1 exact
    sigma = np.array([100.0, 0.01])  # point1 dominates the weighted loss
    best = sel.run_tournament(iter([(x, y, sigma)]), lambda x, a: a + 0 * x)
    np.testing.assert_allclose(best[0], [1.0])


def test_tournament_from_checkpoint_rejects_length_mismatch():
    cfg = GlobalOptimizationConfig(n_starts=3, elimination_rounds=1)
    sel = TournamentSelector(np.arange(6.0).reshape(3, 2), cfg)
    ck = sel.to_checkpoint()
    ck["survival_mask"] = ck["survival_mask"][:2]
    with pytest.raises(ValueError, match="survival_mask"):
        TournamentSelector.from_checkpoint(ck, cfg)


@pytest.mark.parametrize("flag", ["return_eval", "full_output"])
def test_output_flags_do_not_refit_the_winner(flag):
    from nlsq import curve_fit

    x = np.linspace(0, 1, 20)
    y = 4 * x
    kw = {
        "p0": [0.1],
        "bounds": ([-10], [10]),
        "multistart": True,
        "n_starts": 1,
        "max_nfev": 2,
    }
    f = lambda x, a: a * a * x
    plain = curve_fit(f, x, y, **kw)
    out = curve_fit(f, x, y, **kw, **{flag: True})
    np.testing.assert_allclose(out[0], plain.popt)


def test_failed_start_not_counted_successful():
    from nlsq import curve_fit

    x = np.linspace(0, 1, 20)
    r = curve_fit(
        lambda x, a: a * a * x,
        x,
        4 * x,
        p0=[0.1],
        bounds=([-10], [10]),
        multistart=True,
        n_starts=1,
        max_nfev=2,
    )
    d = r.multistart_diagnostics
    assert not r.success
    assert d["n_starts_successful"] == 0
    assert d["n_starts_finite"] == 1


def test_converged_start_beats_cheaper_unconverged_start():
    class _R:
        def __init__(self, success):
            self.success = success

    from nlsq.global_optimization.multi_start import _start_rank

    cheap_unconverged = (np.zeros(1), 1.0, _R(False))
    dear_converged = (np.zeros(1), 5.0, _R(True))
    failed = (np.zeros(1), float("inf"), None)
    ranked = sorted([failed, cheap_unconverged, dear_converged], key=_start_rank)
    assert ranked[0] is dear_converged and ranked[-1] is failed


def test_cmaes_max_generations_is_not_success():
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

    x = np.arange(1.0, 5.0)
    res = CMAESOptimizer(
        CMAESConfig(
            max_generations=1,
            restart_strategy="none",
            refine_with_nlsq=False,
            seed=3,
        )
    ).fit(lambda x, a: a * x, x, 10 * x, bounds=([0.0], [20.0]), p0=[0.1])
    assert res["cmaes_diagnostics"]["convergence_reason"] == "max_generations"
    assert res["success"] is False
    assert res["status"] == 0


def test_bipop_large_population_grows_across_restarts():
    r = BIPOPRestarter(8, 3, seed=1)
    sizes = []
    for _ in range(6):
        sizes.append(r.get_next_popsize())
        r.register_restart()
    large = sizes[0::2]
    assert large == sorted(large) and len(set(large)) == len(large)
    assert all(s <= 8 for s in sizes[1::2])


def test_scale_samples_rejects_nonfinite_bounds():
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="finite"):
        scale_samples_to_bounds(
            jnp.array([[0.0], [0.5]]), jnp.array([0.0]), jnp.array([jnp.inf])
        )


def test_bipop_clean_convergence_does_not_end_the_restart_schedule():
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

    x = np.linspace(0, 3, 30)
    y = 2.0 * x + 1.0
    res = CMAESOptimizer(
        CMAESConfig(
            max_generations=300,
            restart_strategy="bipop",
            max_restarts=2,
            refine_with_nlsq=False,
            seed=2,
        )
    ).fit(lambda x, a, b: a * x + b, x, y, bounds=([-10, -10], [10, 10]), p0=[1, 0])
    diag = res["cmaes_diagnostics"]
    assert diag["total_restarts"] == 2
    assert res["success"] is True


# ---- round 2 ----------------------------------------------------------------


def _nan_data():
    x = np.linspace(0, 5, 50)
    y = 3.0 * np.exp(-0.7 * x)
    y[5] = np.nan
    return x, y


def _plain_kw():
    return {"p0": [1.0, 1.0], "bounds": ([0, 0], [10, 5]), "nan_policy": "omit"}


@pytest.mark.parametrize("route", ["multistart", "cmaes", "fallback"])
@pytest.mark.parametrize("flag", ["return_eval", "full_output"])
def test_output_flags_use_rows_actually_fit(route, flag):
    from nlsq import curve_fit

    if route == "cmaes":
        pytest.importorskip("evosax")
    x, y = _nan_data()
    extra = {
        "multistart": {"multistart": True, "n_starts": 3},
        "cmaes": {
            "method": "cmaes",
            "cmaes_config": CMAESConfig(
                max_generations=3, restart_strategy="none", seed=1
            ),
        },
        "fallback": {"fallback": True},
    }[route]
    out = curve_fit(_exp, x, y, **_plain_kw(), **extra, **{flag: True})
    ref = curve_fit(_exp, x, y, **_plain_kw(), **{flag: True})
    if flag == "return_eval":
        assert np.shape(out[2]) == np.shape(ref[2]) == (49,)
    else:
        assert np.shape(out[2]["fvec"]) == np.shape(ref[2]["fvec"]) == (49,)


def test_multistart_timeit_reports_elapsed_time(exp_data):
    from nlsq import curve_fit

    x, y = exp_data
    out = curve_fit(
        _exp, x, y, p0=[1, 1], bounds=([0, 0], [10, 5]),
        multistart=True, n_starts=2, timeit=True,
    )  # fmt: skip
    assert out[4] > 0.0


def test_seed_range_validated():
    with pytest.raises(ValueError, match="seed"):
        GlobalOptimizationConfig(seed=2**64)


def test_bipop_budget_stops_schedule_loudly():
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer
    from nlsq.result import OptimizeWarning

    x = np.linspace(0, 3, 30)
    y = 2.0 * x + 1.0
    budget = 400
    opt = CMAESOptimizer(
        CMAESConfig(
            max_generations=100,
            restart_strategy="bipop",
            max_restarts=9,
            max_total_evaluations=budget,
            tol_x=1e-300,  # no run converges: the budget is what stops the search
            tol_fun=1e-300,
            refine_with_nlsq=False,
            seed=2,
        )
    )
    with pytest.warns(OptimizeWarning, match="max_total_evaluations"):
        res = opt.fit(
            lambda x, a, b: a * x + b, x, y, bounds=([-10, -10], [10, 10]), p0=[1, 0]
        )
    diag = res["cmaes_diagnostics"]
    spent = sum(r["popsize"] * r["generations"] for r in diag["restart_history"])
    assert spent <= budget + max(r["popsize"] for r in diag["restart_history"])
    assert diag["convergence_reason"] == "max_evaluations"
    assert diag["budget_exhausted"] is True
    assert diag["total_restarts"] < 9


def test_max_total_evaluations_validated():
    with pytest.raises(ValueError, match="max_total_evaluations"):
        CMAESConfig(max_total_evaluations=0)
    assert CMAESConfig().max_total_evaluations is None


def test_unrefined_cmaes_nonconvergence_warns():
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer
    from nlsq.result import OptimizeWarning

    x = np.arange(1.0, 5.0)
    opt = CMAESOptimizer(
        CMAESConfig(
            max_generations=1, restart_strategy="none", refine_with_nlsq=False, seed=3
        )
    )
    with pytest.warns(OptimizeWarning, match="without converging"):
        opt.fit(lambda x, a: a * x, x, 10 * x, bounds=([0.0], [20.0]), p0=[0.1])


def test_cheaper_unconverged_start_is_reported(monkeypatch, exp_data):
    x, y = exp_data

    class _R(dict):
        def __init__(self, popt, success):
            super().__init__()
            self.popt, self.success = np.asarray(popt), success

    orch = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=2))
    results = [
        (np.array([3.0, 0.7]), 5.0, _R([3.0, 0.7], True)),
        (np.array([1.0, 1.0]), 1.0, _R([1.0, 1.0], False)),
    ]
    results.sort(
        key=__import__(
            "nlsq.global_optimization.multi_start", fromlist=["_start_rank"]
        )._start_rank
    )
    monkeypatch.setattr(
        orch, "evaluate_starting_points", lambda **kw: (results, {"parallel": False})
    )
    res = orch.fit(_exp, x, y, p0=[1, 1], bounds=([0, 0], [10, 5]))
    assert res.multistart_diagnostics["cheaper_unconverged_loss"] == 1.0


# ---- round 2, Codex ---------------------------------------------------------


def test_bipop_budget_is_checked_before_convergence_breaks():
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer
    from nlsq.result import OptimizeWarning

    x = np.arange(1.0, 5.0)
    opt = CMAESOptimizer(
        CMAESConfig(
            popsize=4,
            max_generations=3,
            max_restarts=2,
            restart_strategy="bipop",
            max_total_evaluations=1,
            tol_x=1.0,  # every run "converges" on its first generation
            refine_with_nlsq=False,
            seed=3,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)  # converged best run: silent
        res = opt.fit(lambda x, a: a * x, x, 10 * x, bounds=([0.0], [20.0]), p0=[0.1])
    diag = res["cmaes_diagnostics"]
    hist = diag["restart_history"]
    assert sum(r["popsize"] * r["generations"] for r in hist) <= 1 + hist[0]["popsize"]
    assert diag["budget_exhausted"] is True
    assert diag["convergence_reason"] == "xtol"
    assert res["success"] is True


def test_tournament_weights_rounds_by_observations():
    cfg = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=2, elimination_fraction=0.1, batches_per_round=1
    )
    sel = TournamentSelector(np.array([[1.0], [0.0]]), cfg)
    batches = iter([(np.zeros(100), np.zeros(100)), (np.zeros(1), np.ones(1))])
    best = sel.run_tournament(batches, lambda x, a: a + 0 * x)
    np.testing.assert_allclose(best[0], [0.0])
    np.testing.assert_allclose(sel.cumulative_losses, [100 / 101, 1 / 101])


def test_tournament_checkpoint_roundtrip_and_old_format_rejected():
    cfg = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=2, elimination_fraction=0.1, batches_per_round=1
    )
    sel = TournamentSelector(np.array([[1.0], [0.0]]), cfg)
    sel.run_tournament(
        iter([(np.zeros(100), np.zeros(100)), (np.zeros(1), np.ones(1))]),
        lambda x, a: a + 0 * x,
    )
    ck = sel.to_checkpoint()
    back = TournamentSelector.from_checkpoint(ck, cfg)
    np.testing.assert_allclose(back.cumulative_losses, sel.cumulative_losses)
    old = {k: v for k, v in ck.items() if k not in ("loss_sums", "obs_counts")}
    with pytest.raises(ValueError, match="older"):
        TournamentSelector.from_checkpoint(old, cfg)


def test_multistart_timeit_keeps_solver_timing():
    from nlsq import curve_fit

    x = np.arange(1.0, 6.0)
    y = 2 * x
    f = lambda x, a: a * x
    ref = curve_fit(f, x, y, p0=[1.0], timeit=True)
    out = curve_fit(f, x, y, p0=[1.0], multistart=True, n_starts=1, timeit=True)
    assert len(out) == len(ref) == 5
    assert ref[2]["all_times"]["ftimes"]
    assert len(out[2]["all_times"]["ftimes"]) == len(ref[2]["all_times"]["ftimes"])
    assert len(out[2]["all_times"]["jtimes"]) == len(ref[2]["all_times"]["jtimes"])
    assert out[3] > 0.0
    assert "fun" not in out[2] and "jac" not in out[2]


def test_scale_samples_to_bounds_is_jittable():
    import jax
    import jax.numpy as jnp

    s = jnp.array([[0.0, 0.5], [1.0, 0.25]])
    lb, ub = jnp.array([0.0, -1.0]), jnp.array([2.0, 1.0])
    np.testing.assert_allclose(
        jax.jit(scale_samples_to_bounds)(s, lb, ub), scale_samples_to_bounds(s, lb, ub)
    )
    with pytest.raises(ValueError, match="finite"):
        scale_samples_to_bounds(s, lb, jnp.array([jnp.inf, 1.0]))


# ---- round 3, Codex ---------------------------------------------------------


@pytest.mark.parametrize(
    "sigma",
    [np.array([[1.0], [10.0]]), np.eye(2), np.ones(3)],
    ids=["col", "cov", "len"],
)
def test_tournament_rejects_misshaped_sigma(sigma):
    cfg = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=0, batches_per_round=1
    )
    sel = TournamentSelector(np.array([[1.0], [0.0]]), cfg)
    batch = (np.zeros(2), np.array([0.0, 1.0]), sigma)
    with pytest.raises(ValueError, match="sigma"):
        sel.run_tournament(iter([batch]), lambda x, a: a + 0 * x)


def test_tournament_accepts_scalar_and_1d_sigma():
    cfg = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=0, batches_per_round=1
    )
    for sigma, expect in ((np.array([1.0, 10.0]), 0.0), (2.0, None)):
        sel = TournamentSelector(np.array([[1.0], [0.0]]), cfg)
        best = sel.run_tournament(
            iter([(np.zeros(2), np.array([0.0, 1.0]), sigma)]), lambda x, a: a + 0 * x
        )
        if expect is not None:
            np.testing.assert_allclose(best[0], [expect])


# ---- round 3, Claude --------------------------------------------------------


class _Captured(BaseException):
    pass


@pytest.mark.parametrize("route", ["multistart", "cmaes", "auto_global"])
def test_return_eval_with_user_data_mask_matches_plain(route):
    from nlsq import curve_fit
    from nlsq import fit as nlsq_fit

    if route == "cmaes":
        pytest.importorskip("evosax")
    x = np.linspace(0, 4, 50)
    y = 3.0 * np.exp(-0.7 * x)
    mask = np.ones(50, bool)
    mask[:7] = False
    kw = {"p0": [1.0, 1.0], "bounds": ([0, 0], [10, 5]), "data_mask": mask}
    ref = curve_fit(_exp, x, y, return_eval=True, **kw)
    assert np.shape(ref[2]) == (50,)
    if route == "multistart":
        out = curve_fit(_exp, x, y, return_eval=True, multistart=True, n_starts=2, **kw)
    elif route == "cmaes":
        cfg = CMAESConfig(max_generations=3, restart_strategy="none", seed=1)
        out = curve_fit(
            _exp, x, y, return_eval=True, method="cmaes", cmaes_config=cfg, **kw
        )
    else:
        out = nlsq_fit(_exp, x, y, workflow="auto_global", return_eval=True, **kw)
    assert np.shape(out[2]) == (50,)


def test_auto_route_caps_bipop_budget_but_explicit_cmaes_does_not(monkeypatch):
    pytest.importorskip("evosax")
    from nlsq import curve_fit
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

    seen = []

    def spy(self, *a, **k):
        seen.append(self.config)
        raise _Captured

    monkeypatch.setattr(CMAESOptimizer, "fit", spy)
    x = np.linspace(0, 5, 40)
    y = 3.0 * np.exp(-0.7 * x)
    bounds = ([1e-3, 1e-3], [1e1, 1e5])  # wide scale ratio -> auto picks CMA-ES
    with pytest.raises(_Captured):
        curve_fit(_exp, x, y, p0=[1.0, 1.0], bounds=bounds, method="auto")
    assert seen[-1].max_total_evaluations is not None
    with pytest.raises(_Captured):
        curve_fit(_exp, x, y, p0=[1.0, 1.0], bounds=bounds, method="cmaes")
    assert seen[-1].max_total_evaluations is None
    assert CMAESConfig().max_total_evaluations is None


def test_single_run_honours_evaluation_budget():
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer
    from nlsq.result import OptimizeWarning

    x = np.arange(1.0, 5.0)
    opt = CMAESOptimizer(
        CMAESConfig(
            popsize=4,
            max_generations=50,
            restart_strategy="none",
            max_total_evaluations=8,
            refine_with_nlsq=False,
            seed=3,
        )
    )
    with pytest.warns(OptimizeWarning, match="max_total_evaluations"):
        res = opt.fit(lambda x, a: a * x, x, 10 * x, bounds=([0.0], [20.0]), p0=[0.1])
    assert res["cmaes_diagnostics"]["total_generations"] <= 2


def test_single_run_budget_is_silent_when_converged():
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer
    from nlsq.result import OptimizeWarning

    x = np.arange(1.0, 5.0)
    opt = CMAESOptimizer(
        CMAESConfig(
            popsize=4,
            max_generations=3,
            restart_strategy="none",
            max_total_evaluations=1,
            tol_x=1.0,
            refine_with_nlsq=False,
            seed=3,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        res = opt.fit(lambda x, a: a * x, x, 10 * x, bounds=([0.0], [20.0]), p0=[0.1])
    assert res["cmaes_diagnostics"]["budget_exhausted"] is True
    assert res["success"] is True


def test_tournament_skips_zero_observation_batches():
    cfg = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=0, batches_per_round=2
    )
    sel = TournamentSelector(np.array([[1.0], [0.0]]), cfg)
    batches = iter([(np.zeros(0), np.zeros(0)), (np.zeros(2), np.array([0.0, 1.0]))])
    best = sel.run_tournament(batches, lambda x, a: a + 0 * x)
    assert np.all(np.isfinite(sel.cumulative_losses))
    assert len(best) == 1


# ---- round 4, Codex ---------------------------------------------------------


def test_tournament_empty_first_round_does_not_eliminate():
    cfg = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=2, elimination_fraction=0.5, batches_per_round=1
    )
    sel = TournamentSelector(np.array([[0.0], [1.0]]), cfg)
    batches = iter(
        [(np.zeros(0), np.zeros(0)), (np.zeros(0), np.zeros(0)), (np.array([0.0]), np.array([0.0]))]
    )  # fmt: skip
    best = sel.run_tournament(batches, lambda x, a: a + 0 * x)
    np.testing.assert_allclose(best[0], [0.0])
    assert sel.round_history[0]["batches_evaluated"] == 1


def test_tournament_only_empty_batches_fails_clearly():
    cfg = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=2, elimination_fraction=0.5, batches_per_round=1
    )
    sel = TournamentSelector(np.array([[0.0], [1.0]]), cfg)
    with pytest.raises(RuntimeError, match="no data batches"):
        sel.run_tournament(iter([(np.zeros(0), np.zeros(0))]), lambda x, a: a + 0 * x)
    assert sel.n_survivors == 2


def test_resumed_single_run_keeps_spent_evaluation_budget(tmp_path):
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

    x = np.arange(1.0, 5.0)

    def run():
        opt = CMAESOptimizer(
            CMAESConfig(
                popsize=4,
                max_generations=20,
                restart_strategy="none",
                max_total_evaluations=8,
                refine_with_nlsq=False,
                checkpoint_dir=str(tmp_path),
                run_id="r",
                model_id="m",
                seed=5,
            )
        )
        with pytest.warns(UserWarning, match="max_total_evaluations"):
            return opt.fit(
                lambda x, a: a * x, x, 10 * x, bounds=([0.0], [20.0]), p0=[0.1]
            )

    first = run()["cmaes_diagnostics"]["total_generations"]
    second = run()["cmaes_diagnostics"]["total_generations"]
    assert first == second == 2


def test_bipop_cannot_be_checkpointed_so_budget_resume_is_moot(tmp_path):
    with pytest.raises(NotImplementedError):
        CMAESConfig(
            restart_strategy="bipop",
            checkpoint_dir=str(tmp_path),
            run_id="r",
            model_id="m",
            seed=1,
        )


@pytest.mark.parametrize("route", ["multistart", "cmaes", "auto_global"])
def test_return_eval_with_omit_and_data_mask_matches_plain(route):
    from nlsq import curve_fit
    from nlsq import fit as nlsq_fit

    if route == "cmaes":
        pytest.importorskip("evosax")
    x = np.arange(6.0)
    y = 2 * x
    y[2] = np.nan
    mask = np.array([True, True, True, True, False, True])
    f = lambda x, a: a * x
    kw = {
        "p0": [1.0],
        "bounds": ([0.0], [10.0]),
        "data_mask": mask,
        "nan_policy": "omit",
        "return_eval": True,
    }
    ref = curve_fit(f, x, y, **kw)
    assert np.shape(ref[2]) == (5,)
    if route == "multistart":
        out = curve_fit(f, x, y, multistart=True, n_starts=2, **kw)
    elif route == "cmaes":
        cfg = CMAESConfig(max_generations=3, restart_strategy="none", seed=1)
        out = curve_fit(f, x, y, method="cmaes", cmaes_config=cfg, **kw)
    else:
        out = nlsq_fit(f, x, y, workflow="auto_global", **kw)
    np.testing.assert_allclose(out[2], ref[2])


def test_auto_route_converged_fit_emits_no_budget_warning():
    pytest.importorskip("evosax")
    from nlsq import curve_fit
    from nlsq.result import OptimizeWarning

    x = np.linspace(0, 5, 200)
    y = 3.0 * np.exp(-0.7 * x) + np.random.default_rng(0).normal(0, 0.01, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        res = curve_fit(
            _exp, x, y, p0=[1.0, 1.0], bounds=([0, 0], [1e4, 1.0]), method="auto"
        )
    assert res.success
    np.testing.assert_allclose(res.popt, [3.0, 0.7], rtol=0.05)


_TIMEIT_BOUNDS = ([1e-3, 1e-3], [1e1, 1e5])  # ratio > 1000: method='auto' picks CMA-ES


@pytest.mark.parametrize(
    "route",
    [
        {
            "method": "cmaes",
            "cmaes_config": CMAESConfig(
                max_generations=3, restart_strategy="none", seed=1
            ),
        },
        {"method": "auto"},
        {"multistart": True, "n_starts": 2},
    ],
    ids=["cmaes", "auto-cmaes", "multistart"],
)
def test_timeit_reports_measured_times_and_drops_fun_jac(route):
    pytest.importorskip("evosax")
    from nlsq import curve_fit

    x = np.linspace(0, 4, 40)
    y = 3.0 * np.exp(-0.7 * x)
    out = curve_fit(
        _exp, x, y, p0=[1.0, 1.0], bounds=_TIMEIT_BOUNDS, timeit=True, **route
    )
    assert isinstance(out, tuple) and len(out) == 5
    _popt, _pcov, res, post_time, ctime = out
    assert ctime > 0.0 and post_time >= 0.0
    assert "fun" not in res and "jac" not in res  # as the ordinary timeit path
    assert isinstance(res.get("all_times", {}), dict)


def test_auto_chunked_timeit_reports_measured_times_and_drops_fun_jac(monkeypatch):
    from nlsq import curve_fit
    from nlsq.core import workflow as wf

    def force_chunked(
        self, n_points, n_params, memory_limit_gb=None, verbose=False, **k
    ):
        budget = wf.MemoryBudget.compute(n_points=n_points, n_params=n_params)
        return self._create_chunked_config(budget, n_params, None, n_points)

    monkeypatch.setattr(wf.MemoryBudgetSelector, "select", force_chunked)
    x = np.linspace(0, 4, 2000)
    y = 3.0 * np.exp(-0.7 * x) + np.random.default_rng(0).normal(0, 0.01, 2000)
    out = curve_fit(_exp, x, y, p0=[2.0, 1.0], method="auto", timeit=True)
    assert isinstance(out, tuple) and len(out) == 5
    _popt, _pcov, res, post_time, ctime = out
    assert ctime > 0.0 and post_time > 0.0
    assert "fun" not in res and "jac" not in res
    assert isinstance(res.get("all_times", {}), dict)


def test_implicit_auto_budget_warning_names_the_automatic_budget(monkeypatch):
    pytest.importorskip("evosax")
    from nlsq import curve_fit
    from nlsq.core import minpack
    from nlsq.result import OptimizeWarning

    monkeypatch.setattr(minpack, "_auto_bipop_budget", lambda config, n: 50)
    x = np.linspace(0, 4, 40)
    y = 3.0 * np.exp(-0.7 * x)
    with pytest.warns(OptimizeWarning, match=r"automatic budget.*method='cmaes'"):
        curve_fit(_exp, x, y, p0=[1.0, 1.0], bounds=_TIMEIT_BOUNDS, method="auto")

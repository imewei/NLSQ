"""Regression tests for the 2026-11-04 round-2 review of adaptive_hybrid.py
(multistart bounds, group-variance regularization units, scan cost, history
aliasing, progress callback units)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig


def _exp_model(x, a, b):
    return a * jnp.exp(-b * x)


def _fast_config(**kw):
    base = {
        "enable_checkpoints": False,
        "verbose": 0,
        "max_warmup_iterations": 20,
        "warmup_iterations": 5,
        "gauss_newton_max_iterations": 10,
    }
    return HybridStreamingConfig(**{**base, **kw})


class TestMultistartOneSidedBounds:
    """HYB1: lb + u*(ub-lb) with an infinite bound gave NaN candidates."""

    @pytest.mark.parametrize("center_on_p0", [True, False])
    def test_candidates_finite_and_inside_bounds(self, center_on_p0):
        opt = AdaptiveHybridStreamingOptimizer(
            _fast_config(enable_multistart=True, n_starts=6, center_on_p0=center_on_p0)
        )
        lb = np.array([0.0, -np.inf, -np.inf])
        ub = np.array([10.0, np.inf, 5.0])
        opt.original_p0 = jnp.array([1.0, 1.0, 1.0])
        opt._generate_multistart_candidates((jnp.asarray(lb), jnp.asarray(ub)))
        cand = np.asarray(opt.multistart_candidates)
        assert np.all(np.isfinite(cand)), cand
        assert np.all(cand >= lb) and np.all(cand <= ub)

    def test_fit_with_one_sided_bound_returns_finite(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 5, 500)
        y = 3.0 * np.exp(-0.7 * x) + 0.01 * rng.normal(size=x.size)
        opt = AdaptiveHybridStreamingOptimizer(
            _fast_config(enable_multistart=True, n_starts=6)
        )
        bounds = (np.array([0.0, -np.inf]), np.array([10.0, np.inf]))
        r = opt.fit((x, y), _exp_model, [1.0, 1.0], bounds=bounds, verbose=0)
        assert np.all(np.isfinite(np.asarray(r["x"]))), r["x"]

    def test_tournament_drops_nonfinite_candidates(self):
        x = np.linspace(0, 5, 200)
        y = 3.0 * np.exp(-0.7 * x)
        opt = AdaptiveHybridStreamingOptimizer(
            _fast_config(enable_multistart=True, n_starts=4)
        )
        opt._setup_normalization(_exp_model, jnp.array([1.0, 1.0]), None)
        cand = np.array([[1.0, 1.0], [np.nan, 1.0], [2.0, 0.5], [3.0, 0.7]])
        opt.multistart_candidates = jnp.asarray(cand)
        with pytest.warns(UserWarning, match="non-finite"):
            best = opt._run_tournament_selection((x, y), _exp_model)
        assert np.all(np.isfinite(np.asarray(best)))


class TestPhaseHistoryNotAliased:
    """HYB5: a second fit() mutated the first result's phase_history."""

    def test_second_fit_does_not_mutate_first_result(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 5, 300)
        y = 3.0 * np.exp(-0.7 * x) + 0.01 * rng.normal(size=x.size)
        opt = AdaptiveHybridStreamingOptimizer(_fast_config())
        r1 = opt.fit((x, y), _exp_model, [1.0, 1.0], verbose=0)
        h1 = r1["streaming_diagnostics"]["phase_history"]
        n1 = len(h1)
        r2 = opt.fit((x, y), _exp_model, [1.0, 1.0], verbose=0)
        h2 = r2["streaming_diagnostics"]["phase_history"]
        assert len(h1) == n1
        assert h1 is not h2
        assert len(h2) == n1


def _step_model(x, c1, c2):
    return jnp.where(x < 0.5, c1, c2)


class TestGroupVariancePhysicalUnits:
    """HYB2: the group-variance penalty was computed on normalized params."""

    @pytest.mark.slow
    def test_normalize_true_matches_normalize_false(self):
        rng = np.random.default_rng(1)
        x = np.linspace(0, 1, 400)
        y = np.where(x < 0.5, 2.0, 2.2) + 0.01 * rng.normal(size=x.size)
        out = {}
        for norm in (False, True):
            cfg = _fast_config(
                normalize=norm,
                enable_group_variance_regularization=True,
                group_variance_lambda=1e4,
                group_variance_indices=[(0, 2)],
            )
            r = AdaptiveHybridStreamingOptimizer(cfg).fit(
                (x, y), _step_model, [1.0, 3.0], verbose=0
            )
            out[norm] = np.asarray(r["x"])
        np.testing.assert_allclose(out[True], out[False], rtol=1e-2)

    def test_phase2_penalty_gradient_and_hessian_match_autodiff(self):
        def model(x, a, b, c):
            return a + b * x + c * x**2

        x = jnp.linspace(0, 1, 50)
        y = model(x, 1.0, 3.0, 0.5)
        lam = 10.0
        cfg = _fast_config(
            enable_group_variance_regularization=True,
            group_variance_lambda=lam,
            group_variance_indices=[(0, 3)],
        )
        opt = AdaptiveHybridStreamingOptimizer(cfg)
        opt._setup_normalization(model, jnp.array([1.0, 3.0, 0.5]), None)
        scales = np.asarray(opt.normalizer.scales)
        assert np.ptp(scales) > 0  # otherwise the test cannot see the bug

        captured = {}
        orig = opt._solve_gauss_newton_step

        def spy(JTJ, JTr, trust_radius):
            captured["JTJ"], captured["JTr"] = JTJ, JTr
            return orig(JTJ, JTr, trust_radius)

        opt._solve_gauss_newton_step = spy
        q = opt.normalizer.normalize(jnp.array([1.5, 2.0, 0.2]))
        opt._gauss_newton_iteration((x, y), q, 1.0)
        JTJ_reg, JTr_reg = captured["JTJ"], captured["JTr"]
        n = len(x)

        cfg0 = _fast_config(
            enable_group_variance_regularization=False,
            group_variance_indices=[(0, 3)],
        )
        opt0 = AdaptiveHybridStreamingOptimizer(cfg0)
        opt0._setup_normalization(model, jnp.array([1.0, 3.0, 0.5]), None)
        opt0._solve_gauss_newton_step = lambda J, r, t: (
            captured.__setitem__("J0", J),
            captured.__setitem__("r0", r),
            orig(J, r, t),
        )[2]
        opt0._gauss_newton_iteration((x, y), q, 1.0)

        def penalty(qq):
            return jnp.var(opt.normalizer.denormalize(qq))

        # JTr = -grad(0.5*cost); regularizer adds 0.5*lam*n*Var
        d_grad = np.asarray(JTr_reg - captured["r0"])
        np.testing.assert_allclose(
            d_grad, -0.5 * lam * n * np.asarray(jax.grad(penalty)(q)), rtol=1e-8
        )
        d_hess = np.asarray(JTJ_reg - captured["J0"])
        np.testing.assert_allclose(
            d_hess, 0.5 * lam * n * np.asarray(jax.hessian(penalty)(q)), rtol=1e-8
        )


class TestScanCostMatchesLoop:
    """HYB4: scan cost recompiled on every call."""

    def test_scan_equals_loop_and_does_not_recompile(self):
        x = jnp.linspace(0, 5, 4000)
        y = _exp_model(x, 3.0, 0.7) + 0.01 * jnp.sin(37 * x)
        costs = {}
        opts = {}
        for ls in ("scan", "loop"):
            o = AdaptiveHybridStreamingOptimizer(
                _fast_config(loop_strategy=ls, chunk_size=1000)
            )
            o._setup_normalization(_exp_model, jnp.array([1.0, 1.0]), None)
            opts[ls] = o
            costs[ls] = [
                float(o._compute_cost_only(jnp.array([1.0 + 0.1 * i, 1.0]), x, y))
                for i in range(3)
            ]
        np.testing.assert_allclose(costs["scan"], costs["loop"], rtol=1e-12)

        # repeated calls must not trigger new XLA compilations
        import logging

        n_compiles = [0]

        class H(logging.Handler):
            def emit(self, record):
                if "Finished XLA compilation" in record.getMessage():
                    n_compiles[0] += 1

        lg = logging.getLogger("jax")
        h = H()
        lg.addHandler(h)
        old_level = lg.level
        lg.setLevel(logging.DEBUG)
        try:
            with jax.log_compiles(True):
                for i in range(3):
                    opts["scan"]._compute_cost_only(
                        jnp.array([1.0 + 0.01 * i, 1.0]), x, y
                    )
        finally:
            lg.removeHandler(h)
            lg.setLevel(old_level)
        assert n_compiles[0] == 0, n_compiles[0]


class TestCallbackReceivesPhysicalParams:
    """HYB6: progress callback got normalized params."""

    def test_callback_params_are_physical(self):
        x = np.linspace(0, 5, 400)
        y = 10.0 * np.exp(-0.1 * x)
        seen = []
        cfg = _fast_config(
            callback_frequency=1,
            enable_warm_start_detection=False,
            gauss_newton_max_iterations=5,
        )
        opt = AdaptiveHybridStreamingOptimizer(cfg)
        opt.fit(
            (x, y),
            _exp_model,
            [8.0, 0.2],
            verbose=0,
            callback=lambda it, cost, p, info: seen.append(np.asarray(p)),
        )
        assert seen
        # normalized p0 would be [1, 1]; physical p0 is [8, 0.2]
        np.testing.assert_allclose(seen[0], [8.0, 0.2], rtol=1e-6)


class TestMaxIterationsWarning:
    """Phase 2 stopping at max iterations warns once (success semantics unchanged)."""

    def _fit(self, **kw):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 5, 300)
        y = 3.0 * np.exp(-0.7 * x) + 0.01 * rng.normal(size=x.size)
        opt = AdaptiveHybridStreamingOptimizer(_fast_config(**kw))
        return opt.fit((x, y), _exp_model, [1.0, 1.0], verbose=0)

    def test_warns_when_stopped_at_max_iterations(self):
        from nlsq.result import OptimizeWarning

        with pytest.warns(OptimizeWarning, match="max iterations") as rec:
            r = self._fit(gauss_newton_max_iterations=1)
        assert r["message"] == "Maximum iterations reached"
        assert (
            sum(
                "hybrid streaming stopped at max iterations" in str(w.message)
                for w in rec
            )
            == 1
        )

    def test_no_warning_when_converged(self):
        import warnings

        from nlsq.result import OptimizeWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            r = self._fit(gauss_newton_max_iterations=50)
        assert r["message"] != "Maximum iterations reached"


class TestHybridAbsoluteSigma:
    """F2: hybrid covariance ignored absolute_sigma=True."""

    @pytest.mark.parametrize("absolute_sigma", [True, False])
    def test_pcov_matches_curve_fit(self, absolute_sigma):
        from nlsq import curve_fit
        from nlsq.streaming.strategy import build_large_strategy

        rng = np.random.default_rng(3)
        x = np.linspace(0, 10, 2000)
        y = 2.0 * x + 1.0 + 0.2 * rng.normal(size=x.size)

        def lin(x, a, b):
            return a * x + b

        _, ref = curve_fit(lin, x, y, p0=[1, 1], absolute_sigma=absolute_sigma)
        strategy = build_large_strategy(
            "hybrid", _fast_config(gauss_newton_max_iterations=50)
        )
        native = strategy.fit(
            lin,
            x,
            y,
            np.array([1.0, 1.0]),
            absolute_sigma=absolute_sigma,
            verbose=0,
        )
        np.testing.assert_allclose(
            np.asarray(native["pcov"]), np.asarray(ref), rtol=1e-3
        )


class TestMaxIterationsWarningAttribution:
    def test_warning_points_at_caller_even_through_strategy_and_multistart(self):
        import warnings

        from nlsq.result import OptimizeWarning
        from nlsq.streaming.strategy import build_large_strategy

        rng = np.random.default_rng(0)
        x = np.linspace(0, 5, 300)
        y = 3.0 * np.exp(-0.7 * x) + 0.01 * rng.normal(size=x.size)
        cfg = _fast_config(
            gauss_newton_max_iterations=1, enable_multistart=True, n_starts=3
        )
        strategy = build_large_strategy("hybrid", cfg)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            strategy.fit(_exp_model, x, y, np.array([1.0, 1.0]), verbose=0)
        hits = [
            w
            for w in rec
            if issubclass(w.category, OptimizeWarning)
            and "stopped at max iterations" in str(w.message)
        ]
        assert len(hits) == 1
        assert hits[0].filename == __file__

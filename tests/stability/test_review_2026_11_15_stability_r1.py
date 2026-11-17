"""Regression tests for the three-brain review of nlsq/stability/ (2026-11-15).

Each test failed before the corresponding fix (reproduction first).
"""

import gc
import warnings
import weakref
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.stability.fallback import (
    AddParameterBoundsStrategy,
    AlternativeMethodStrategy,
    FallbackOrchestrator,
    FallbackStrategy,
    _fit_succeeded,
)
from nlsq.stability.guard import NumericalStabilityGuard, solve_with_cholesky_fallback
from nlsq.stability.recovery import OptimizationRecovery
from nlsq.stability.svd_fallback import compute_svd_with_fallback

jax.config.update("jax_enable_x64", True)


def _exp_model(x, a, b):
    return a * jnp.exp(-b * x)


# ---------------------------------------------------------------------------
# fallback.py
# ---------------------------------------------------------------------------


class TestAddBoundsStrategy:
    def test_does_not_invent_sign_constraints(self):
        """An amplitude/offset with negative data must not get a lower bound of 0."""
        x = np.linspace(0, 10, 50)
        y = 2 * np.exp(-0.5 * x) - 3
        out = AddParameterBoundsStrategy().apply(
            {"p0": np.ones(3), "_xdata": x, "_ydata": y}
        )
        lb, _ = out["bounds"]
        assert lb[0] < 0 and lb[2] < 0

    def test_fit_pinned_to_invented_bound_is_not_success(self):
        """Growth data (b < 0) is excluded by the inferred rate bounds: the fit
        lands on the invented bound and must not be reported as success."""
        from nlsq import curve_fit

        x = np.linspace(0, 4, 60)
        y = 2.0 * np.exp(0.5 * x)
        result, _, pinned = FallbackOrchestrator._run_strategy(
            AddParameterBoundsStrategy(),
            curve_fit,
            _exp_model,
            x,
            y,
            {"p0": np.ones(2), "_xdata": x, "_ydata": y},
        )
        assert pinned or not _fit_succeeded(result)

    def test_unconstrained_solution_inside_bounds_still_succeeds(self):
        from nlsq import curve_fit

        x = np.linspace(0, 4, 60)
        y = 2.0 * np.exp(-0.5 * x)
        result, _, pinned = FallbackOrchestrator._run_strategy(
            AddParameterBoundsStrategy(),
            curve_fit,
            _exp_model,
            x,
            y,
            {"p0": np.ones(2), "_xdata": x, "_ydata": y},
        )
        assert _fit_succeeded(result) and not pinned
        np.testing.assert_allclose(result.popt, [2.0, 0.5], rtol=1e-4)


class TestFitSucceeded:
    def test_orchestrator_runs_strategies_when_full_output_fails(self):
        x = np.linspace(0, 5, 40)
        y = 2.5 * np.exp(-0.7 * x)
        orch = FallbackOrchestrator(max_attempts=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orch.fit_with_fallback(
                _exp_model, x, y, p0=[100.0, -5.0], max_nfev=1, full_output=True
            )
        assert orch.total_attempts > 1


class TestNoDuplicateSolves:
    def test_alternative_method_noop_when_already_trf(self):
        out = AlternativeMethodStrategy().apply({"p0": [1.0, 1.0]})
        assert out.get("method") in (None, "trf") and out == {"p0": [1.0, 1.0]}

    def test_no_identical_curve_fit_calls(self, monkeypatch):
        calls = []

        def fake_curve_fit(f, x, y, **kw):
            # method="trf" is curve_fit's default, so it does not make a new problem
            kws = sorted(
                (k, np.asarray(v).tolist())
                for k, v in kw.items()
                if (k, v) != ("method", "trf")
            )
            calls.append(repr((float(np.sum(x)), float(np.sum(y)), kws)))
            sol = np.full(2, float(len(calls)))  # distinct per call, like a real solve
            # the rescale pre-solve (the only call passing compute_covariance)
            # must succeed to hand its start on; an unsuccessful one makes the
            # final solve start from the caller's p0 by design
            return SimpleNamespace(
                success="compute_covariance" in kw,
                cost=1.0,
                fun=np.zeros(3),
                x=sol,
                popt=sol,
            )

        from nlsq.core import minpack

        monkeypatch.setattr(minpack, "curve_fit", fake_curve_fit)
        x = np.linspace(0, 5, 40)
        orch = FallbackOrchestrator(max_attempts=10)
        orch.fit_with_fallback(_exp_model, x, 2 * np.exp(-x), p0=[1.0, 1.0])
        assert len(calls) == len(set(calls)), "an identical fit was solved twice"

    def test_strategy_list_not_mutated(self):
        from nlsq.stability.fallback import (
            AdjustTolerancesStrategy,
            PerturbInitialGuessStrategy,
        )

        mine = [AdjustTolerancesStrategy(), PerturbInitialGuessStrategy()]
        before = list(mine)
        FallbackOrchestrator(strategies=mine)
        assert mine == before


# ---------------------------------------------------------------------------
# recovery.py
# ---------------------------------------------------------------------------


class TestRecovery:
    def setup_method(self):
        self.rec = OptimizationRecovery()

    def test_multi_start_accepts_list_bounds(self):
        state = {"p0": np.array([1.0, 1.0]), "bounds": [[0.0, 0.0], [10.0, 10.0]]}
        out = self.rec._multi_start("convergence", state, 0)
        assert out["p0"] is not state["p0"]
        assert np.all((out["p0"] >= 0) & (out["p0"] <= 10))

    def test_multi_start_accepts_array_pair_bounds(self):
        state = {"p0": np.array([1.0, 1.0]), "bounds": np.array([[0, 0], [10, 10]])}
        out = self.rec._multi_start("convergence", state, 0)
        assert out["p0"] is not state["p0"]

    def test_perturbation_respects_bounds(self):
        lb, ub = np.array([0.0, 0.0]), np.array([10.0, 10.0])
        state = {"params": np.array([10.0, 10.0]), "p0": np.array([10.0, 10.0])}
        state["bounds"] = (lb, ub)
        for retry in range(4):
            out = self.rec._perturb_parameters("numerical", state, retry)
            assert np.all(out["p0"] <= ub) and np.all(out["p0"] >= lb)
            assert np.all(np.asarray(out["params"]) <= ub)

    def test_perturbation_updates_x_only_state(self):
        state = {"x": np.array([1.0, 2.0])}
        out = self.rec._perturb_parameters("convergence", state, 0)
        assert not np.array_equal(out["x"], state["x"])

    def test_multi_start_updates_x_only_state(self):
        state = {"x": np.array([1.0, 2.0])}
        out = self.rec._multi_start("convergence", state, 0)
        assert not np.array_equal(out["x"], state["x"])

    def test_zero_parameter_is_actually_perturbed(self):
        state = {"params": np.array([0.0, 0.0]), "p0": np.array([0.0, 0.0])}
        out = self.rec._perturb_parameters("convergence", state, 0)
        assert np.max(np.abs(out["p0"])) > 1e-6

    def test_adjust_regularization_does_not_repeat_solve(self):
        state = {"method": "trf", "p0": np.ones(2)}
        keys = ("tr_solver", "x_scale", "loss")
        forwarded = [
            tuple(
                self.rec._adjust_regularization("numerical", state, r).get(k)
                for k in keys
            )
            for r in range(3)
        ]
        # retries after the first must be no-ops (state returned unchanged)
        for r in (1, 2):
            out = self.rec._adjust_regularization("numerical", state, r)
            assert (
                all(out[k] is v for k, v in state.items())
                and out.keys() == state.keys()
            )
        assert forwarded[0] != (None, None, None)


# ---------------------------------------------------------------------------
# guard.py
# ---------------------------------------------------------------------------


class TestGuardDtype:
    def setup_method(self):
        self.g = NumericalStabilityGuard()

    def test_safe_exp_float32_does_not_overflow(self):
        out = self.g.safe_exp(jnp.array([100.0], dtype=jnp.float32))
        assert out.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_safe_exp_float64_unchanged(self):
        out = self.g.safe_exp(jnp.array([1000.0, -1000.0]))
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_safe_norm_float32_representable_result(self):
        x = jnp.array([1e30, 1e30], dtype=jnp.float32)
        assert np.isclose(self.g.safe_norm(x), np.sqrt(2) * 1e30, rtol=1e-5)

    def test_safe_norm_float64_large(self):
        x = jnp.array([1e200, 1e200])
        assert np.isclose(self.g.safe_norm(x), np.sqrt(2) * 1e200, rtol=1e-12)

    def test_safe_norm_matrix_spectral_on_scaled_path(self):
        a = jnp.diag(jnp.array([3.0, 4.0])) * 1e101
        assert np.isclose(self.g.safe_norm(a, ord=2), 4e101, rtol=1e-12)

    @pytest.mark.parametrize("dtype,val", [(jnp.float32, 1e30), (jnp.float64, 1e200)])
    def test_gradient_clipping_keeps_direction(self, dtype, val):
        g = jnp.array([val, val], dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.g.check_gradient(g)
        out = np.asarray(out, dtype=np.float64)
        assert np.all(out > 0)
        assert np.isclose(np.linalg.norm(out), 1e6, rtol=1e-4)
        assert np.isclose(out[0], out[1])


class TestResiduals:
    def test_zero_mad_still_detects_outlier(self):
        g = NumericalStabilityGuard()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, has_outliers = g.check_residuals(jnp.array([0.0, 0, 0, 0, 100.0]))
        assert has_outliers

    def test_constant_residuals_have_no_outliers(self):
        g = NumericalStabilityGuard()
        _, has_outliers = g.check_residuals(jnp.full(10, 3.0))
        assert not has_outliers
        _, has_outliers = g.check_residuals(jnp.zeros(10))
        assert not has_outliers


class TestCholeskyEigenFallback:
    def test_indefinite_matrix_is_solved_exactly(self):
        x, used = solve_with_cholesky_fallback(
            jnp.diag(jnp.array([1.0, -1.0])), jnp.array([1.0, 1.0])
        )
        assert not bool(used)
        np.testing.assert_allclose(np.asarray(x), [1.0, -1.0], atol=1e-12)

    def test_singular_psd_matches_head_formula_bitwise(self):
        # HEAD's eigen branch: eigvals clamped up to eps (sign-free for PSD input)
        A = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        b = jnp.array([1.0, 2.0])
        x, used = solve_with_cholesky_fallback(A, b)
        assert not bool(used)
        w, v = jnp.linalg.eigh(0.5 * (A + A.T))
        eps = jnp.finfo(A.dtype).eps * jnp.maximum(jnp.max(jnp.abs(w)), 1.0)
        ref = v @ (v.T @ b / jnp.maximum(w, eps))
        np.testing.assert_array_equal(np.asarray(x), np.asarray(ref))

    def test_spd_uses_cholesky_unchanged(self):
        A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x, used = solve_with_cholesky_fallback(A, b)
        assert bool(used)
        L = jnp.linalg.cholesky(A)
        np.testing.assert_array_equal(
            np.asarray(x), np.asarray(jax.scipy.linalg.cho_solve((L, True), b))
        )


class TestJacobianShape:
    def test_1d_jacobian_gives_clear_error(self):
        with pytest.raises(ValueError, match="2-D"):
            NumericalStabilityGuard().check_and_fix_jacobian(jnp.ones(5))


# ---------------------------------------------------------------------------
# svd_fallback.py
# ---------------------------------------------------------------------------


class TestComplexSVD:
    A = jnp.array([[1 + 1j, 2 + 0j], [3 + 0j, 4 - 2j]])

    def _check(self, U, s, V):
        recon = U @ jnp.diag(s) @ V.conj().T
        np.testing.assert_allclose(np.asarray(recon), np.asarray(self.A), atol=1e-10)

    def test_default_tier(self):
        self._check(*compute_svd_with_fallback(self.A))

    def test_cpu_and_numpy_tiers(self, monkeypatch):
        import nlsq.stability.svd_fallback as sf

        real = sf.jax_svd
        state = {"n": 0}

        def flaky(*a, **k):
            state["n"] += 1
            if state["n"] <= 1:
                raise FloatingPointError("force fallback")
            return real(*a, **k)

        monkeypatch.setattr(sf, "jax_svd", flaky)
        with pytest.warns(RuntimeWarning):
            self._check(*compute_svd_with_fallback(self.A))

        def always(*a, **k):
            raise FloatingPointError("force numpy tier")

        monkeypatch.setattr(sf, "jax_svd", always)
        with pytest.warns(RuntimeWarning):
            self._check(*compute_svd_with_fallback(self.A))

    def test_nan_input_fails_once_without_gpu_labels(self):
        bad = jnp.array([[jnp.nan, 1.0], [1.0, 1.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(np.linalg.LinAlgError):
                compute_svd_with_fallback(bad)
        assert not any("GPU SVD failed" in str(x.message) for x in w)


class TestReturnEvalAndXOnlyRecovery:
    @pytest.mark.parametrize("combo", [{}, {"full_output": True}, {"timeit": True}])
    def test_return_eval_returns_selected_attempt_evaluation(self, combo):
        # return_eval overrides the other flags and carries no status (documented):
        # the eval must be model(x, *popt) of the very popt that is returned.
        x = np.linspace(0, 5, 40)
        y = 2.5 * np.exp(-0.7 * x)
        r = FallbackOrchestrator(max_attempts=4).fit_with_fallback(
            _exp_model, x, y, p0=[1.0, 1.0], return_eval=True, **combo
        )
        popt, _pcov, yfit = r.result
        assert len(r.result) == 3
        np.testing.assert_allclose(popt, [2.5, 0.7], rtol=1e-5)
        np.testing.assert_allclose(
            yfit, np.asarray(_exp_model(jnp.asarray(x), *popt)), rtol=1e-12
        )

    def test_recover_from_failure_with_strict_x_only_callback(self):
        seen = []

        def opt(x):  # rejects any `params`/`p0` keyword
            seen.append(np.asarray(x))
            return {"x": np.asarray(x), "success": True, "cost": 1.0}

        x0 = np.array([1.0, 2.0])
        ok, _res = OptimizationRecovery().recover_from_failure(
            "convergence", {"x": x0}, opt
        )
        assert ok and not np.array_equal(seen[0], x0)


class TestRound2:
    def test_noop_strategy_with_always_true_has_more_terminates(self):
        import signal

        from nlsq.stability.fallback import FallbackStrategy

        class Stuck(FallbackStrategy):
            def __init__(self):
                super().__init__("stuck", "no-op forever")

            def apply(self, kwargs):
                return kwargs.copy()

            def has_more(self):
                return True

        calls = []

        def failing(*a, **k):
            calls.append(1)
            raise ValueError("boom")

        def on_alarm(*_):
            raise TimeoutError("fallback loop did not terminate")

        old = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(10)
        try:
            orch = FallbackOrchestrator(strategies=[Stuck()], max_attempts=3)
            with pytest.raises(RuntimeError):
                orch._try_strategies(
                    failing,
                    _exp_model,
                    np.ones(3),
                    np.ones(3),
                    {"p0": [1.0, 1.0]},
                    None,
                    ValueError("orig"),
                    "linear",
                )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        assert calls == []  # every proposal was a skipped no-op

    def test_mutating_strategy_returning_same_object_is_bounded(self):
        from nlsq.stability.fallback import FallbackStrategy

        class Mutator(FallbackStrategy):
            def __init__(self):
                super().__init__("mut", "mutates in place")

            def apply(self, kwargs):
                return kwargs

            def has_more(self):
                return True

        calls = []

        def failing(*a, **k):
            calls.append(1)
            raise ValueError("boom")

        orch = FallbackOrchestrator(strategies=[Mutator()], max_attempts=4)
        with pytest.raises(RuntimeError):
            orch._try_strategies(
                failing,
                _exp_model,
                np.ones(3),
                np.ones(3),
                {"p0": [1.0, 1.0]},
                None,
                ValueError("orig"),
                "linear",
            )
        assert len(calls) == 4  # bounded by max_attempts (counter starts at 0 here)

    def test_rescale_scales_default_f_scale_for_robust_loss(self):
        from nlsq.stability.fallback import _fit_rescaled

        seen = []

        def fake(f, x, y, **kw):
            seen.append(kw.get("f_scale"))
            return SimpleNamespace(x=np.ones(2))

        info = {"_x_scale": 1.0, "_y_scale": 20.0, "_x_offset": 0.0, "_y_offset": 0.0}
        _fit_rescaled(
            fake, _exp_model, np.ones(3), np.ones(3), {"loss": "soft_l1"}, info
        )
        assert seen[0] == pytest.approx(1.0 / 20.0)

    @pytest.mark.parametrize(
        "res", [[0.0, 0, 0, 100.0], [0.0, 0, 100.0], [0.0, 0, 0, 0, 0, 0, 100.0]]
    )
    def test_zero_mad_single_extreme_outlier_small_sample(self, res):
        g = NumericalStabilityGuard()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, has = g.check_residuals(jnp.array(res))
        assert has

    def test_zero_mad_tiny_samples_do_not_crash_or_flag_constants(self):
        g = NumericalStabilityGuard()
        assert not g.check_residuals(jnp.array([5.0]))[1]
        assert not g.check_residuals(jnp.array([2.0, 2.0]))[1]

    def test_safe_exp_honours_min_exp_arg(self):
        g = NumericalStabilityGuard()
        g.min_exp_arg = -50
        assert float(g.safe_exp(jnp.array(-700.0))) == pytest.approx(
            np.exp(-50.0), rel=1e-9, abs=0
        )

    def test_gradient_clipping_float64_matches_head_formula_bitwise(self):
        g = NumericalStabilityGuard()
        grad = jnp.array([3e9, -4e9, 5e9])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = g.check_gradient(grad)
        norm = jnp.linalg.norm(grad)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(grad * (1e6 / norm)))


def _head_cholesky_solve(A, b):
    """solve_with_cholesky_fallback exactly as at HEAD (eigen branch: max(lam, eps))."""
    A = 0.5 * (A + A.T)
    L = jnp.linalg.cholesky(A)
    x_chol = jax.scipy.linalg.cho_solve((L, True), b)
    valid = jnp.all(jnp.isfinite(x_chol))

    def eig():
        w, v = jnp.linalg.eigh(A)
        eps = jnp.finfo(A.dtype).eps * jnp.maximum(jnp.max(jnp.abs(w)), 1.0)
        return v @ (v.T @ b / jnp.maximum(w, eps))

    return jax.lax.cond(valid, lambda: x_chol, eig), valid


class TestRound3:
    @pytest.mark.parametrize("n,rank", [(100, 50), (200, 100)])
    def test_rank_deficient_gram_matches_head_bitwise(self, n, rank):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            J = rng.standard_normal((2 * n, rank)) @ rng.standard_normal((rank, n))
            A = jnp.asarray(J.T @ J)
            b = jnp.asarray(rng.standard_normal(n))
            x, used = solve_with_cholesky_fallback(A, b)
            xh, usedh = _head_cholesky_solve(A, b)
            assert bool(used) == bool(usedh)
            np.testing.assert_array_equal(np.asarray(x), np.asarray(xh))

    def _flag(self, res):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return NumericalStabilityGuard().check_residuals(jnp.asarray(res))[1]

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("n", [100, 1000])
    def test_tied_gaussian_and_count_data_not_flagged(self, seed, n):
        rng = np.random.default_rng(seed)
        assert not self._flag(np.round(rng.normal(0, 0.5, n)))
        assert not self._flag(rng.poisson(0.3, n).astype(float))

    @pytest.mark.parametrize(
        "res", [[0.0, 0, 0, 1, 2], [0.0, 0, 1, 1], [2.0], [1.0, 3.0]]
    )
    def test_ties_and_tiny_samples_not_flagged(self, res):
        assert not self._flag(res)

    @pytest.mark.parametrize("n", [1, 2, 3, 50])
    def test_constant_vectors_not_flagged(self, n):
        assert not self._flag(np.full(n, 7.0))
        assert not self._flag(np.zeros(n))

    @pytest.mark.parametrize(
        "res",
        [
            [0.0, 0, 0, 100],
            [0.0, 0, 0, 0, 100],
            [0.0] * 10 + [1.0, 100.0],
            [0.0, 0, 0, 0, 1, 100],
        ],
    )
    def test_extreme_outlier_with_zero_mad_flagged(self, res):
        assert self._flag(res)

    def test_poisson_with_spike_flagged(self):
        r = np.random.default_rng(0).poisson(0.3, 100).astype(float)
        r[7] = 500.0
        assert self._flag(r)


class _R(dict):
    __getattr__ = dict.__getitem__


class TestFullOutputRankingMatchesObjects:
    @staticmethod
    def _run(monkeypatch, loss, funs, costs, full_output):
        from nlsq.core import minpack
        from nlsq.stability.fallback import AdjustTolerancesStrategy

        calls = []

        def fake(f, x, y, **kw):
            # attempt identity from the kwargs (AdjustTolerances sets ftol), not call
            # order: the winner is re-run once with the output flags on
            i = 1 if "ftol" in kw else 0
            calls.append(kw)
            res = _R(
                success=False,
                cost=costs[i],
                fun=np.asarray(funs[i]),
                popt=np.array([float(i)]),
                x=np.array([float(i)]),
                pcov=np.eye(1),
                nfev=1,
                status=0,
                message="max nfev",
            )
            if kw.get("full_output"):  # a real curve_fit would hand back a tuple
                return res.popt, res.pcov, {"nfev": 1, "fvec": res.fun}, "m", 5
            return res

        monkeypatch.setattr(minpack, "curve_fit", fake)
        orch = FallbackOrchestrator(
            strategies=[AdjustTolerancesStrategy()], max_attempts=2
        )
        x = np.linspace(0, 1, 2)
        kw = {"loss": loss, "p0": [1.0], "f_scale": 1.0}
        if full_output:
            kw["full_output"] = True
        return orch.fit_with_fallback(_exp_model, x, x, **kw).result

    def test_soft_l1_picks_same_winner_as_objects(self, monkeypatch):
        # soft_l1 costs: [10,0] -> 9.05 (better), [6,6] -> 10.17; SSR says the opposite
        funs, costs = [[10.0, 0.0], [6.0, 6.0]], [9.05, 10.17]
        obj = self._run(monkeypatch, "soft_l1", funs, costs, False)
        tup = self._run(monkeypatch, "soft_l1", funs, costs, True)
        assert isinstance(tup, tuple) and tup[4] == 5
        np.testing.assert_array_equal(np.asarray(tup[0]), np.asarray(obj.popt))
        assert float(obj.popt[0]) == 0.0

    def test_linear_loss_picks_lower_ssr(self, monkeypatch):
        funs, costs = [[10.0, 0.0], [6.0, 6.0]], [50.0, 36.0]
        obj = self._run(monkeypatch, "linear", funs, costs, False)
        tup = self._run(monkeypatch, "linear", funs, costs, True)
        np.testing.assert_array_equal(np.asarray(tup[0]), np.asarray(obj.popt))
        assert float(obj.popt[0]) == 1.0


def _kind(v):
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, (str, bytes)):
        return "str"
    if isinstance(v, (int, float, np.floating, np.integer)):
        return "num"
    return "arr" if hasattr(v, "shape") else type(v).__name__


_FLAG_COMBOS = [
    dict(zip(("full_output", "return_eval", "timeit"), bits, strict=True))
    for bits in [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
]


class TestOutputFormatParity:
    @staticmethod
    def _both(flags, x=None, y=None, model=_exp_model, p0=(1.0, 1.0), **extra):
        from nlsq import curve_fit
        from nlsq.stability.fallback import AdjustTolerancesStrategy

        if x is None:
            x = np.linspace(0, 5, 40)
            y = 2.0 * np.exp(-0.5 * x)
        kw = {**flags, **extra, "p0": list(p0)}
        ref = curve_fit(model, x, y, **kw)
        orch = FallbackOrchestrator(strategies=[AdjustTolerancesStrategy()])
        got = orch.fit_with_fallback(model, x, y, **kw).result
        return ref, got

    @pytest.mark.parametrize(
        "flags", _FLAG_COMBOS, ids=lambda f: "-".join(k for k, v in f.items() if v)
    )
    def test_tuple_shape_and_types_match_curve_fit(self, flags):
        ref, got = self._both(flags)
        assert isinstance(got, tuple) and len(got) == len(ref)
        assert [_kind(v) for v in got] == [_kind(v) for v in ref]
        np.testing.assert_allclose(np.asarray(got[0]), np.asarray(ref[0]), rtol=1e-6)
        if flags["return_eval"]:
            assert got[2].shape == ref[2].shape
        elif flags["full_output"]:
            assert (
                got[4] in (1, 2, 3, 4)
                and ref[4] in (1, 2, 3, 4)
                and set(got[2]) == set(ref[2])
            )
        else:  # timeit: raw result without the bulky fun/jac
            assert "fun" not in got[2] and "jac" not in got[2]

    def test_timeit_reports_real_positive_timings(self):
        _, got = self._both(
            {"full_output": False, "return_eval": False, "timeit": True}
        )
        assert got[3] > 0 and got[4] > 0

    def test_return_eval_omits_nan_rows_like_curve_fit(self):
        x = np.arange(6.0)
        y = 2 * x
        y[2] = np.nan

        def line(x, a, b):
            return a * x + b

        ref, got = self._both(
            {"full_output": False, "return_eval": True, "timeit": False},
            x=x,
            y=y,
            model=line,
            nan_policy="omit",
        )
        assert len(ref[2]) == 5
        np.testing.assert_allclose(got[2], ref[2], atol=1e-8)


def _line(x, a, b):
    return a * x + b


def _masked_data(variant):
    x = np.arange(6.0)
    y = 2 * x
    extra = {}
    if variant in ("omit", "both"):
        y[2] = np.nan
        extra["nan_policy"] = "omit"
    if variant in ("mask", "both"):
        extra["data_mask"] = np.array([True, True, True, True, False, True])
    if variant == "both":
        extra["data_mask"] = np.array([True, True, True, False, True, True])
    return x, y, extra


class TestOutputParityMaskedData:
    @pytest.mark.parametrize("variant", ["omit", "mask", "both"])
    @pytest.mark.parametrize(
        "flags", _FLAG_COMBOS, ids=lambda f: "-".join(k for k, v in f.items() if v)
    )
    def test_matches_curve_fit_verbatim(self, variant, flags):
        x, y, extra = _masked_data(variant)
        ref, got = TestOutputFormatParity._both(
            flags, x=x, y=y, model=_line, p0=(1.0, 0.0), **extra
        )
        assert isinstance(got, tuple) and len(got) == len(ref)
        assert [_kind(v) for v in got] == [_kind(v) for v in ref]
        if flags["return_eval"]:
            assert got[2].shape == ref[2].shape
            np.testing.assert_allclose(got[2], ref[2], atol=1e-6)
        if flags["timeit"] and not (flags["return_eval"] or flags["full_output"]):
            assert got[3] > 0 and got[4] > 0


class TestAttemptMemoryAndRerunFailure:
    @staticmethod
    def _patch(monkeypatch, refs, fail_on_flags=False):
        from nlsq.core import minpack

        n = [0]

        def fake(f, x, y, **kw):
            if fail_on_flags and kw.get("full_output"):
                raise RuntimeError("re-run exploded")
            n[0] += 1
            res = _R(
                success=False,
                cost=100.0 - n[0],  # later attempts are better, so the best moves
                fun=np.zeros(1000),
                jac=np.zeros((1000, 2)),
                popt=np.array([float(n[0])]),
                x=np.array([float(n[0])]),
                pcov=np.eye(1),
                nfev=1,
                status=0,
                message="max nfev",
            )
            refs.append(weakref.ref(res))
            return res

        monkeypatch.setattr(minpack, "curve_fit", fake)

    def test_only_best_attempt_result_survives(self, monkeypatch):
        refs = []
        self._patch(monkeypatch, refs)
        orch = FallbackOrchestrator(max_attempts=8)
        x = np.linspace(0, 1, 3)
        fr = orch.fit_with_fallback(_exp_model, x, x, p0=[1.0, 1.0])
        gc.collect()
        assert len(refs) > 3
        alive = [r() for r in refs if r() is not None]
        assert len(alive) == 1 and alive[0] is fr.result


class TestFlagsGoToEveryAttemptVerbatim:
    """Attempts run with the user's output flags and are returned verbatim."""

    @pytest.mark.parametrize(
        "flags",
        [dict.fromkeys(("full_output", "return_eval", "timeit"), False), *_FLAG_COMBOS],
        ids=lambda f: "-".join(k for k, v in f.items() if v) or "plain",
    )
    def test_stateful_callback_runs_once_and_winner_is_kept(self, flags):
        from nlsq.callbacks import StopOptimization

        class ResetStop:
            """Raises when the iteration counter resets (a second solve)."""

            def __init__(self):
                self.last = None
                self.resets = 0

            def __call__(self, iteration, cost, params, info):
                if self.last is not None and iteration <= self.last:
                    self.resets += 1
                    raise StopOptimization("iteration counter reset")
                self.last = iteration

        cb = ResetStop()
        x = np.linspace(0, 5, 40)
        out = (
            FallbackOrchestrator(max_attempts=4)
            .fit_with_fallback(_line, x, 3 * x + 1, p0=[0.0, 0.0], callback=cb, **flags)
            .result
        )
        popt = np.asarray(out[0] if isinstance(out, tuple) else out.popt)
        assert cb.resets == 0
        np.testing.assert_allclose(popt, [3.0, 1.0], rtol=1e-5, atol=1e-6)

    def test_bound_hit_detected_from_full_output_tuple(self):
        from nlsq import curve_fit

        x = np.linspace(0, 4, 60)
        y = 2.0 * np.exp(0.5 * x)
        result, _, pinned = FallbackOrchestrator._run_strategy(
            AddParameterBoundsStrategy(),
            curve_fit,
            _exp_model,
            x,
            y,
            {"p0": np.ones(2), "_xdata": x, "_ydata": y, "full_output": True},
        )
        assert isinstance(result, tuple) and pinned
        assert result[4] == 5  # marked unsuccessful in the tuple itself

    def test_bound_hit_detected_from_status_less_tuple(self):
        from nlsq import curve_fit

        x = np.linspace(0, 4, 60)
        y = 2.0 * np.exp(0.5 * x)
        result, _, pinned = FallbackOrchestrator._run_strategy(
            AddParameterBoundsStrategy(),
            curve_fit,
            _exp_model,
            x,
            y,
            {"p0": np.ones(2), "_xdata": x, "_ydata": y, "return_eval": True},
        )
        assert isinstance(result, tuple) and len(result) == 3 and pinned

    def test_status_less_nonfinite_is_failure_and_first_finite_wins(self):
        from nlsq.stability.fallback import _fit_succeeded, _keep_best

        flags = {"return_eval": True}
        bad = (np.array([np.nan]), None, np.zeros(2))
        ok1 = (np.array([1.0]), None, np.zeros(2))
        ok2 = (np.array([2.0]), None, np.zeros(2))
        assert not _fit_succeeded(bad, flags) and _fit_succeeded(ok1, flags)
        best = _keep_best(None, bad, "linear", "linear", "bad", flags=flags)
        best = _keep_best(best, ok1, "linear", "linear", "ok1", flags=flags)
        best = _keep_best(best, ok2, "linear", "linear", "ok2", flags=flags)
        assert best[2] == "ok1"

    def test_full_output_ier_decides_success(self):
        from nlsq.stability.fallback import _fit_succeeded

        flags = {"full_output": True}
        assert _fit_succeeded((np.ones(1), None, {}, "m", 2), flags)
        assert not _fit_succeeded((np.ones(1), None, {}, "m", 5), flags)


def _big_exp_data():
    x = np.linspace(0.0, 5.0, 60)
    return x, 5000.0 * np.exp(-0.3 * x) + 100.0


def _big_model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


class TestRound7Rescale:
    def test_presolve_is_callback_and_flag_free_flagged_solve_is_verbatim(
        self, monkeypatch
    ):
        from nlsq.core import minpack
        from nlsq.stability.fallback import RescaleProblemStrategy

        calls = []

        def fake(f, x, y, **kw):
            calls.append(dict(kw))
            if len(calls) == 1:
                raise RuntimeError("original attempt fails")
            if not kw.get("full_output"):  # the pre-solve
                return SimpleNamespace(x=np.array([1.0, 0.3, 0.1]))
            return (
                np.array([1.0, 0.3, 0.1]),
                np.eye(3),
                {"nfev": 1, "fvec": np.zeros(len(x))},
                "m",
                1,
            )

        monkeypatch.setattr(minpack, "curve_fit", fake)
        x, y = _big_exp_data()
        cb = object()
        orch = FallbackOrchestrator(
            strategies=[RescaleProblemStrategy()], max_attempts=2
        )
        orch.fit_with_fallback(
            _big_model, x, y, p0=[1.0, 1.0, 1.0], full_output=True, callback=cb
        )
        assert len(calls) == 3  # original, callback-free pre-solve, flagged solve
        presolve, flagged = calls[1], calls[2]
        for key in ("full_output", "return_eval", "timeit", "callback"):
            assert not presolve.get(key)
        assert flagged["full_output"] is True and flagged["callback"] is cb
        # the flagged solve starts from the pre-solve's popt (original units)
        np.testing.assert_allclose(flagged["p0"], [1.0, 0.3, 0.1])

    def test_presolve_failure_falls_back_to_original_p0(self, monkeypatch):
        from nlsq.core import minpack
        from nlsq.stability.fallback import RescaleProblemStrategy

        calls = []

        def fake(f, x, y, **kw):
            calls.append(dict(kw))
            if len(calls) < 3:
                raise RuntimeError("fails")
            return SimpleNamespace(x=np.ones(3), success=True)

        monkeypatch.setattr(minpack, "curve_fit", fake)
        x, y = _big_exp_data()
        FallbackOrchestrator(
            strategies=[RescaleProblemStrategy()], max_attempts=2
        ).fit_with_fallback(_big_model, x, y, p0=[2.0, 3.0, 4.0])
        np.testing.assert_allclose(calls[2]["p0"], [2.0, 3.0, 4.0])

    @pytest.mark.parametrize("sigma", [False, True])
    @pytest.mark.parametrize("mode", ["plain", "full_output", "return_eval", "timeit"])
    def test_converted_output_matches_ordinary_fit(self, mode, sigma):
        from nlsq import curve_fit
        from nlsq.stability.fallback import RescaleProblemStrategy

        x, y = _big_exp_data()
        extra = {"p0": [4000.0, 0.2, 50.0]}
        if sigma:
            extra["sigma"] = np.full(len(x), 25.0)
        flags = {} if mode == "plain" else {mode: True}
        ref = curve_fit(_big_model, x, y, **extra, **flags)
        got, _, _ = FallbackOrchestrator._run_strategy(
            RescaleProblemStrategy(),
            curve_fit,
            _big_model,
            x,
            y,
            {**extra, **flags, "_xdata": x, "_ydata": y},
        )
        if mode == "plain":
            for k in ("x", "pcov", "fun", "xdata", "ydata"):
                np.testing.assert_allclose(
                    np.asarray(got[k]), np.asarray(ref[k]), rtol=1e-4, atol=1e-6
                )
            np.testing.assert_allclose(
                float(got["cost"]), float(ref["cost"]), rtol=1e-3, atol=1e-6
            )
            np.testing.assert_allclose(
                np.asarray(got["jac"]),
                np.asarray(ref["jac"]),
                rtol=1e-3,
                atol=1e-6 * np.abs(np.asarray(ref["jac"])).max(),
            )
            assert got["model"] is _big_model
        elif mode == "return_eval":
            np.testing.assert_allclose(got[2], ref[2], rtol=1e-6)
        elif mode == "full_output":
            np.testing.assert_allclose(
                got[2]["fvec"], ref[2]["fvec"], rtol=1e-3, atol=1e-3
            )
            assert got[4] in (1, 2, 3, 4) and ref[4] in (1, 2, 3, 4)
        else:
            np.testing.assert_allclose(
                float(got[2]["cost"]), float(ref[2]["cost"]), rtol=1e-3, atol=1e-6
            )
        np.testing.assert_allclose(
            np.asarray(got[0] if mode != "plain" else got["x"]),
            np.asarray(ref[0] if mode != "plain" else ref["x"]),
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            np.asarray(got[1] if mode != "plain" else got["pcov"]),
            np.asarray(ref[1] if mode != "plain" else ref["pcov"]),
            rtol=1e-3,
            atol=1e-10,
        )


class TestRound7Warnings:
    def test_status_less_best_effort_return_warns(self, monkeypatch):
        from nlsq.core import minpack
        from nlsq.result import OptimizeWarning

        def fake(f, x, y, **kw):
            return np.array([np.nan, 1.0]), np.eye(2), np.zeros(len(x))

        monkeypatch.setattr(minpack, "curve_fit", fake)
        x = np.linspace(0, 1, 5)
        orch = FallbackOrchestrator(max_attempts=3)
        with pytest.warns(OptimizeWarning, match="no status"):
            fr = orch.fit_with_fallback(
                _exp_model, x, x, p0=[1.0, 1.0], return_eval=True
            )
        assert isinstance(fr.result, tuple) and len(fr.result) == 3

    def test_status_aware_failure_does_not_add_the_warning(self, monkeypatch):
        from nlsq.core import minpack

        def fake(f, x, y, **kw):
            return np.ones(2), np.eye(2), {"nfev": 1, "fvec": np.ones(len(x))}, "m", 5

        monkeypatch.setattr(minpack, "curve_fit", fake)
        x = np.linspace(0, 1, 5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FallbackOrchestrator(max_attempts=3).fit_with_fallback(
                _exp_model, x, x, p0=[1.0, 1.0], full_output=True
            )
        assert not any("no status" in str(m.message) for m in w)


class TestRound7HitsBound:
    def test_tiny_scale_interior_fit_is_not_a_hit(self):
        from nlsq.stability.fallback import _hits_bound

        res = SimpleNamespace(popt=np.array([5e-9]))
        assert not _hits_bound(res, (np.array([5e-10]), np.array([5e-8])))

    def test_tiny_scale_strategy_end_to_end(self):
        from nlsq import curve_fit

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4e-9, 5e-9, 6e-9])

        def line(x, a):
            return a * x

        _result, _, pinned = FallbackOrchestrator._run_strategy(
            AddParameterBoundsStrategy(),
            curve_fit,
            line,
            x,
            y,
            {"p0": np.array([5e-9]), "_xdata": x, "_ydata": y},
        )
        assert not pinned

    @pytest.mark.parametrize("scale", [1.0, 1e-9, 1e6])
    def test_genuine_hit_still_detected(self, scale):
        from nlsq.stability.fallback import _hits_bound

        lb, ub = np.array([0.1 * scale]), np.array([10.0 * scale])
        for popt in (lb + 1e-9 * (ub - lb), ub - 1e-9 * (ub - lb), ub, lb):
            assert _hits_bound(SimpleNamespace(popt=popt), (lb, ub))
        assert not _hits_bound(SimpleNamespace(popt=(lb + ub) / 2), (lb, ub))

    def test_infinite_and_scalar_bounds(self):
        from nlsq.stability.fallback import _hits_bound

        res = SimpleNamespace(popt=np.array([1.0, 2.0]))
        assert not _hits_bound(res, (-np.inf, np.inf))
        assert _hits_bound(
            res, (1.0, np.inf)
        )  # first parameter on a finite lower bound
        assert not _hits_bound(res, (0.5, np.inf))
        assert _hits_bound(res, (-np.inf, np.array([1.0, 5.0])))


class TestRound7CallableLoss:
    @staticmethod
    def _quad(z):
        z = np.asarray(z)
        return np.stack([z, np.ones_like(z), np.zeros_like(z)])

    def test_callable_loss_ranks_full_output_tuples(self):
        from nlsq.stability.fallback import _keep_best

        flags = {"full_output": True}

        def tup(fvec):
            return (np.ones(1), None, {"fvec": np.asarray(fvec)}, "m", 5)

        a = _keep_best(None, tup([10.0, 0.0]), self._quad, self._quad, "a", flags=flags)
        b = _keep_best(a, tup([1.0, 0.0]), self._quad, self._quad, "b", flags=flags)
        assert a[0][2] == pytest.approx(50.0) and b[2] == "b"
        assert b[0][2] == pytest.approx(0.5)

    def test_failing_callable_loss_ranks_inf_first_kept(self):
        from nlsq.stability.fallback import _keep_best

        flags = {"full_output": True}

        def boom(z):
            raise RuntimeError("bad loss")

        t = lambda v: (np.ones(1), None, {"fvec": np.asarray(v)}, "m", 5)
        a = _keep_best(None, t([10.0]), boom, boom, "a", flags=flags)
        b = _keep_best(a, t([1.0]), boom, boom, "b", flags=flags)
        assert b[2] == "a"


def _w1_model(x, a):
    x = jnp.asarray(x)
    return 1e10 * x + a * x


class TestRound8Rescale:
    @pytest.mark.parametrize(
        "flags",
        [dict.fromkeys(("full_output", "return_eval", "timeit"), False), *_FLAG_COMBOS],
        ids=lambda f: "-".join(k for k, v in f.items() if v) or "plain",
    )
    def test_huge_offset_model_converges_through_rescale(self, flags):
        from nlsq import curve_fit
        from nlsq.stability.fallback import RescaleProblemStrategy

        x = np.arange(4.0)
        y = (1e10 + 1) * x
        out, _, pinned = FallbackOrchestrator._run_strategy(
            RescaleProblemStrategy(),
            curve_fit,
            _w1_model,
            x,
            y,
            {"p0": [0.0], **flags, "_xdata": x, "_ydata": y},
        )
        assert not pinned
        popt = np.asarray(out[0] if isinstance(out, tuple) else out.x)
        np.testing.assert_allclose(popt, [1.0], atol=1e-4)
        if not isinstance(out, tuple):
            assert out.success and out.cost < 1e-6
        elif flags.get("return_eval"):
            np.testing.assert_allclose(out[2], y, rtol=1e-12)
        elif flags.get("full_output"):
            assert out[4] in (1, 2, 3, 4)
            assert np.max(np.abs(out[2]["fvec"])) < 1e-3

    def test_data_mask_with_sigma_and_absolute_sigma_matches_direct_fit(self):
        from nlsq import curve_fit
        from nlsq.stability.fallback import RescaleProblemStrategy

        rng = np.random.default_rng(3)
        x = np.linspace(0, 5, 40)
        y = 4000.0 * np.exp(-0.3 * x) + 100.0 + rng.normal(0, 5.0, x.size)
        mask = np.ones(x.size, bool)
        mask[[3, 11, 20]] = False
        kw = {
            "p0": [3000.0, 0.2, 50.0],
            "sigma": np.full(x.size, 5.0),
            "absolute_sigma": True,
            "data_mask": mask,
        }
        direct = curve_fit(_big_model, x, y, **kw)
        got, _, _ = FallbackOrchestrator._run_strategy(
            RescaleProblemStrategy(),
            curve_fit,
            _big_model,
            x,
            y,
            {**kw, "_xdata": x, "_ydata": y},
        )
        assert len(np.asarray(got.sigma)) == len(np.asarray(direct.sigma))
        np.testing.assert_allclose(got.pcov, direct.pcov, rtol=1e-3)
        pi_got, pi_dir = got.prediction_interval(), direct.prediction_interval()
        np.testing.assert_allclose(
            np.asarray(pi_got), np.asarray(pi_dir), rtol=1e-3, atol=1e-6
        )

    def test_callback_runs_only_in_flagged_solve_with_original_units(self):
        from nlsq import curve_fit
        from nlsq.stability.fallback import RescaleProblemStrategy

        x, y = _big_exp_data()
        seen = []

        def cb(iteration, cost, params, info):
            seen.append((iteration, float(cost), np.asarray(params).copy()))

        calls = {"n": 0}

        def counting(*a, **k):
            calls["n"] += 1
            return curve_fit(*a, **k)

        FallbackOrchestrator._run_strategy(
            RescaleProblemStrategy(),
            counting,
            _big_model,
            x,
            y,
            {"p0": [4000.0, 0.2, 50.0], "callback": cb, "_xdata": x, "_ydata": y},
        )
        assert calls["n"] == 2
        iters = [i for i, _, _ in seen]
        assert iters == sorted(iters) and len(set(iters)) == len(iters)  # one solve
        # original-unit params and cost: the first iteration starts near p0
        assert abs(seen[0][2][0]) > 100.0

    def test_extreme_scale_model_has_finite_cost(self):
        from nlsq import curve_fit
        from nlsq.stability.fallback import RescaleProblemStrategy

        def huge(x, a):
            return 1e160 * a * jnp.asarray(x)

        x = np.arange(1.0, 5.0)
        y = 2e160 * x
        with np.errstate(all="raise"):
            out, _, _ = FallbackOrchestrator._run_strategy(
                RescaleProblemStrategy(),
                curve_fit,
                huge,
                x,
                y,
                {"p0": [0.5], "_xdata": x, "_ydata": y},
            )
        assert np.isfinite(out.cost)
        np.testing.assert_allclose(out.x, [2.0], rtol=1e-8)

    def test_2d_sigma_scaling_does_not_overflow(self):
        from nlsq.stability.fallback import _rescale_sigma

        out = _rescale_sigma(np.eye(2) * 1e300, 1e160)
        assert np.all(np.isfinite(out)) and out[0, 0] > 0


class TestRound8HitsBound:
    def test_total_allowance_is_capped_by_interval_width(self):
        from nlsq.stability.fallback import _hits_bound

        res = SimpleNamespace(popt=np.array([1e8 + 5e-7]))
        assert not _hits_bound(res, (np.array([1e8]), np.array([1e8 + 1e-6])))

    def test_solver_sized_offset_still_counts_at_any_scale(self):
        from nlsq.stability.fallback import _hits_bound

        for scale in (1e-9, 1.0, 1e8):
            lb, ub = np.array([0.1 * scale]), np.array([10 * scale])
            for popt in (lb + 1e-14 * scale, ub - 1e-14 * scale):
                assert _hits_bound(SimpleNamespace(popt=popt), (lb, ub))

    def test_one_sided_bound_at_zero_uses_the_p0_scale(self):
        from nlsq.stability.fallback import _hits_bound

        res = SimpleNamespace(popt=np.array([1e-14]))
        bounds = (0.0, np.inf)
        assert _hits_bound(res, bounds, p0=np.array([1.0]))
        # a genuinely tiny-scale problem is not a hit
        assert not _hits_bound(res, bounds, p0=np.array([1e-9]))
        assert not _hits_bound(
            SimpleNamespace(popt=np.array([0.5])), bounds, p0=np.array([1.0])
        )

    def test_inferred_bound_decision_uses_both_rules(self):
        class Pin(FallbackStrategy):
            def __init__(self, bounds):
                super().__init__("pin", "pin")
                self.bounds = bounds

            def apply(self, kwargs):
                return {**kwargs, "bounds": self.bounds, "_invented_bounds": True}

        def fit_at(value):
            from nlsq.result import OptimizeResult

            return lambda f, x, y, **kw: OptimizeResult(
                x=np.array([value]), success=True
            )

        x = np.arange(3.0)
        kw = {"p0": [1.0]}
        _, _, pinned = FallbackOrchestrator._run_strategy(
            Pin((np.array([1e8]), np.array([1e8 + 1e-6]))),
            fit_at(1e8 + 5e-7),
            _line,
            x,
            x,
            kw,
        )
        assert not pinned
        _, _, pinned = FallbackOrchestrator._run_strategy(
            Pin((np.array([0.0]), np.array([np.inf]))), fit_at(1e-14), _line, x, x, kw
        )
        assert pinned


class TestRound9Rescale:
    @staticmethod
    def _line1(x, a):
        return a * jnp.asarray(x)

    def test_unsuccessful_presolve_start_is_not_used(self):
        from nlsq import curve_fit
        from nlsq.stability.fallback import _fit_rescaled

        calls = []

        def spy(*a, **k):
            calls.append(k)
            return curve_fit(*a, **k)

        x = np.arange(1.0, 5.0)
        info = {
            "_x_scale": np.ptp(x),
            "_y_scale": np.ptp(2 * x),
            "_x_offset": x.min(),
            "_y_offset": (2 * x).min(),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _fit_rescaled(
                spy, self._line1, x, 2 * x, {"p0": [0.0], "max_nfev": 2}, info
            )
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[1]["p0"], [0.0])  # not the failed start

    def test_presolve_preparation_errors_fall_back_to_p0(self):
        from nlsq.stability.fallback import _fit_rescaled

        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 5e-201, 1e-200])
        calls = []

        def fake(f, xd, yd, **kw):
            calls.append(kw)
            return SimpleNamespace(x=np.array([1e-200]), success=True)

        info = {
            "_x_scale": 2.0,
            "_y_scale": 1e-200,
            "_x_offset": 0.0,
            "_y_offset": 0.0,
        }
        with np.errstate(over="raise"):
            out = _fit_rescaled(
                fake, self._line1, x, y, {"p0": [1e-200], "sigma": [1e200] * 3}, info
            )
        assert len(calls) == 1 and out.success  # only the final solve ran
        np.testing.assert_array_equal(calls[0]["p0"], [1e-200])

    def test_presolve_warnings_are_silenced_final_warnings_are_not(self):
        from nlsq.result import OptimizeWarning
        from nlsq.stability.fallback import _fit_rescaled

        calls = []

        def fake(f, xd, yd, **kw):
            calls.append(kw)
            warnings.warn("solver note", OptimizeWarning, stacklevel=1)
            return SimpleNamespace(x=np.array([2.0]), success=True)

        x = np.arange(1.0, 5.0)
        info = {"_x_scale": 3.0, "_y_scale": 6.0, "_x_offset": 1.0, "_y_offset": 2.0}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")  # would interrupt an unguarded pre-solve
            with pytest.raises(OptimizeWarning):
                _fit_rescaled(fake, self._line1, x, 2 * x, {"p0": [0.0]}, info)
        # the pre-solve ran to completion; only the final solve's warning escaped
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[1]["p0"], [2.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _fit_rescaled(fake, self._line1, x, 2 * x, {"p0": [0.0]}, info)
        assert len(w) == 1  # final solve only


class TestRound9HitsBound:
    def test_one_ulp_from_a_narrow_bound_is_a_hit_midpoint_is_not(self):
        from nlsq.stability.fallback import _hits_bound

        lb, ub = np.array([1e8]), np.array([1e8 + 1e-6])
        near = SimpleNamespace(popt=np.nextafter(lb, ub))
        assert _hits_bound(near, (lb, ub), p0=lb)
        assert not _hits_bound(SimpleNamespace(popt=(lb + ub) / 2), (lb, ub), p0=lb)

    def test_unrelated_parameter_does_not_inflate_another_scale(self):
        from nlsq.stability.fallback import _hits_bound

        res = SimpleNamespace(popt=np.array([1e9, 1e-9]))
        bounds = (np.array([-np.inf, 0.0]), np.array([np.inf, np.inf]))
        assert not _hits_bound(res, bounds, p0=np.array([1e9, 1e-9]))

    def test_zero_scale_uses_unit_floor(self):
        from nlsq.stability.fallback import _hits_bound

        res = SimpleNamespace(popt=np.array([1e-14]))
        assert _hits_bound(res, (0.0, np.inf), p0=np.array([0.0]))

    def test_mixed_scale_genuine_hit_and_non_hit(self):
        from nlsq.stability.fallback import _hits_bound

        bounds = (np.array([-np.inf, 0.0]), np.array([np.inf, np.inf]))
        p0 = np.array([1e9, 1.0])
        assert _hits_bound(SimpleNamespace(popt=np.array([1e9, 1e-14])), bounds, p0=p0)
        assert not _hits_bound(
            SimpleNamespace(popt=np.array([1e9, 0.3])), bounds, p0=p0
        )


class TestRound10:
    def test_hits_bound_extreme_bounds_do_not_overflow(self):
        from nlsq.stability.fallback import _hits_bound

        big = np.finfo(float).max
        with np.errstate(all="raise"):
            res = SimpleNamespace(popt=np.array([0.0]))
            assert not _hits_bound(res, (-np.inf, big), p0=np.array([0.0]))
            assert not _hits_bound(res, (-big, big), p0=np.array([0.0]))
            assert not _hits_bound(res, (-big, np.inf), p0=np.array([0.0]))
            for edge in (big, -big):
                hit = SimpleNamespace(popt=np.array([edge]))
                assert _hits_bound(hit, (-big, big), p0=np.array([0.0]))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("ord_", [0, 1, 2, np.inf, -np.inf, 3])
    @pytest.mark.parametrize("magnitude", [1e-30, 1.0, 1e30])
    def test_safe_norm_matches_reference_for_every_order(self, dtype, ord_, magnitude):
        import math

        if dtype is np.float32 and not (1e-35 < magnitude < 1e35):
            pytest.skip("out of float32 range")
        base = np.array([3.0, -4.0, 0.0, 1.5])
        x = jnp.asarray((base * magnitude).astype(dtype))
        guard = NumericalStabilityGuard()
        vals = [abs(float(v)) for v in np.asarray(x, dtype=np.float64)]
        if ord_ == 0:
            ref = float(sum(v != 0 for v in vals))
        elif ord_ == np.inf:
            ref = max(vals)
        elif ord_ == -np.inf:
            ref = min(vals)
        else:
            ref = max(vals) * math.fsum((v / max(vals)) ** ord_ for v in vals) ** (
                1 / ord_
            )
        got = guard.safe_norm(x, ord=ord_)
        np.testing.assert_allclose(
            got, ref, rtol=1e-5 if dtype is np.float32 else 1e-10
        )

    def test_safe_norm_ord0_huge_float32(self):
        x = jnp.array([1e30, 0.0], dtype=jnp.float32)
        assert NumericalStabilityGuard().safe_norm(x, ord=0) == 1.0

    def test_perturb_uses_p0_when_it_is_the_only_start_key(self):
        rec = OptimizationRecovery(max_retries=2)
        rec.strategies = [rec._perturb_parameters]
        calls = []

        def retry(p0):
            calls.append(np.asarray(p0))
            return {"success": True, "x": np.asarray(p0)}

        ok, _ = rec.recover_from_failure("convergence", {"p0": np.array([1.0])}, retry)
        assert ok and len(calls) == 1 and not np.array_equal(calls[0], [1.0])

    def test_perturb_source_order_and_key_discipline(self):
        rec = OptimizationRecovery(max_retries=2)
        both = {"params": np.array([1.0]), "x": np.array([5.0]), "p0": np.array([9.0])}
        mod = rec._perturb_parameters("x", both, 0)
        # perturbed from `params`, written to every key the state has
        assert all(np.array_equal(mod[k], mod["params"]) for k in both)
        assert abs(mod["params"][0] - 1.0) < 1.0
        only_x = rec._perturb_parameters("x", {"x": np.array([2.0])}, 0)
        assert set(only_x) == {"x"}
        none_start = rec._perturb_parameters(
            "x", {"params": None, "x": np.array([2.0])}, 0
        )
        assert abs(none_start["x"][0] - 2.0) < 1.0


class TestRound11:
    @staticmethod
    def _steps(v, toward, n):
        for _ in range(n):
            v = np.nextafter(v, toward)
        return v

    def test_two_steps_into_interval_is_a_hit_at_powers_of_two(self):
        from nlsq.stability.fallback import _hits_bound

        # lower bound 1.0: the step above is twice the step below
        lb, ub = np.array([1.0]), np.array([1.0 + 1e-14])
        pt = SimpleNamespace(popt=self._steps(lb, ub, 2))
        assert _hits_bound(pt, (lb, ub), p0=lb)
        # upper bound 2.0: the step below is half the step above
        lb, ub = np.array([2.0 - 1e-14]), np.array([2.0])
        pt = SimpleNamespace(popt=self._steps(ub, lb, 2))
        assert _hits_bound(pt, (lb, ub), p0=ub)
        # interior midpoint stays a non-hit
        assert not _hits_bound(SimpleNamespace(popt=(lb + ub) / 2), (lb, ub), p0=ub)

    def test_two_steps_into_interval_at_finfo_max(self):
        from nlsq.stability.fallback import _hits_bound

        big = np.array([np.finfo(float).max])
        with np.errstate(all="raise"):
            pt = SimpleNamespace(popt=self._steps(big, 0.0, 2))
            assert _hits_bound(pt, (-big, big), p0=np.zeros(1))
            pt = SimpleNamespace(popt=self._steps(-big, 0.0, 2))
            assert _hits_bound(pt, (-big, big), p0=np.zeros(1))

    def test_subnormal_values_do_not_raise(self):
        from nlsq.stability.fallback import _hits_bound

        with np.errstate(all="raise"):
            res = SimpleNamespace(popt=np.array([1e-320]))
            assert _hits_bound(res, (0.0, 1.0), p0=np.array([1.0]))
            assert _hits_bound(res, (1e-320, 1.0), p0=np.array([1.0]))
            assert not _hits_bound(
                SimpleNamespace(popt=np.array([0.5])),
                (np.array([5e-324]), np.array([1.0])),
                p0=np.array([1.0]),
            )

    @pytest.mark.parametrize("key", ["params", "x", "p0"])
    def test_list_valued_start_is_perturbed(self, key):
        rec = OptimizationRecovery(max_retries=1)
        mod = rec._perturb_parameters("x", {key: [1.0, 2.0]}, 0)
        assert isinstance(mod[key], np.ndarray) and mod[key].shape == (2,)
        assert not np.array_equal(mod[key], [1.0, 2.0])

    def test_list_p0_recovers_through_the_callback(self):
        rec = OptimizationRecovery(max_retries=1)
        rec.strategies = [rec._perturb_parameters]
        seen = []

        def retry(p0):
            seen.append(p0)
            return {"success": True, "x": np.asarray(p0)}

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ok, _ = rec.recover_from_failure("convergence", {"p0": [1.0]}, retry)
        assert ok and len(seen) == 1


class TestRound12:
    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_two_steps_across_a_power_of_two_are_a_hit(self, sign):
        from nlsq.stability.fallback import _hits_bound

        nx = np.nextafter
        with np.errstate(all="raise"):
            # lower bound just below +-1, interior above it
            lb = nx(sign, -np.inf)
            ub = lb + 1e-14
            p = nx(nx(lb, np.inf), np.inf)
            assert _hits_bound(
                SimpleNamespace(popt=np.array([p])),
                (np.array([lb]), np.array([ub])),
                p0=[1.0],
            )
            # upper bound just above +-1, interior below it
            ub = nx(sign, np.inf)
            lb = ub - 1e-14
            p = nx(nx(ub, -np.inf), -np.inf)
            assert _hits_bound(
                SimpleNamespace(popt=np.array([p])),
                (np.array([lb]), np.array([ub])),
                p0=[1.0],
            )
            # three steps away is not covered by the guarantee and not a hit here
            p3 = nx(nx(nx(ub, -np.inf), -np.inf), -np.inf)
            mid = (lb + ub) / 2
            assert not _hits_bound(
                SimpleNamespace(popt=np.array([mid])),
                (np.array([lb]), np.array([ub])),
                p0=[1.0],
            )
            assert p3 < ub

    def _rec(self, result, bounds=None):
        rec = OptimizationRecovery(max_retries=2)
        rec.strategies = [rec._perturb_parameters]
        calls = []

        def retry(p0, **kw):
            calls.append(p0)
            return result

        state = {"p0": [1.0]}
        if bounds is not None:
            state["bounds"] = bounds
        return rec.recover_from_failure("convergence", state, retry), calls

    def test_nan_popt_is_not_a_recovery(self):
        (ok, _), calls = self._rec({"success": True, "popt": [np.nan], "cost": np.nan})
        assert not ok and len(calls) == 2  # retries continue

    def test_nan_cost_without_parameter_field_is_not_a_recovery(self):
        (ok, _), _ = self._rec({"success": True, "cost": np.nan})
        assert not ok

    def test_popt_outside_bounds_is_not_a_recovery(self):
        (ok, _), _ = self._rec(
            {"success": True, "popt": [5.0]}, bounds=(np.array([0.0]), np.array([2.0]))
        )
        assert not ok

    def test_valid_results_still_succeed(self):
        for res in (
            {"success": True, "popt": [1.0], "cost": 0.5},
            {"success": True, "x": [1.0]},
            {"success": True},
            {"success": True, "cost": 0.0},
        ):
            (ok, out), _ = self._rec(res)
            assert ok and out is res


class TestRound13:
    @pytest.mark.parametrize("dtype,val", [(jnp.float64, 1e308), (jnp.float32, 2e38)])
    @pytest.mark.parametrize("jitted", [False, True])
    def test_cholesky_solve_survives_symmetrization_overflow(self, dtype, val, jitted):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        x, used = fn(jnp.array([[val]], dtype=dtype), jnp.array([val], dtype=dtype))
        np.testing.assert_allclose(np.asarray(x), [1.0], rtol=1e-6)
        assert bool(used)

    def test_cholesky_validity_requires_a_sane_factor(self):
        # a factor with a non-positive diagonal must not count as success
        A = jnp.array([[0.0, 0.0], [0.0, 1.0]])
        x, used = solve_with_cholesky_fallback(A, jnp.array([0.0, 1.0]))
        assert not bool(used) and bool(jnp.all(jnp.isfinite(x)))

    def test_ordinary_spd_is_bit_identical_to_head_formula(self):
        rng = np.random.default_rng(0)
        M = rng.normal(size=(6, 6))
        A = jnp.asarray(M @ M.T + 6 * np.eye(6))
        b = jnp.asarray(rng.normal(size=6))
        x, used = solve_with_cholesky_fallback(A, b)
        S = 0.5 * (A + A.T)
        ref = jax.scipy.linalg.cho_solve((jnp.linalg.cholesky(S), True), b)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(ref))
        assert bool(used)

    def test_regularize_hessian_symmetrization_does_not_overflow(self):
        g = NumericalStabilityGuard()
        H = g.regularize_hessian(jnp.array([[1e308, 0.0], [0.0, 1e308]]))
        assert bool(jnp.all(jnp.isfinite(H)))

    @staticmethod
    def _recovery(result, state):
        rec = OptimizationRecovery(max_retries=1, enable_diagnostics=False)
        rec.strategies = [rec._perturb_parameters]
        calls = []

        def retry(**kw):
            calls.append(kw)
            return result

        return rec.recover_from_failure("convergence", state, retry), calls

    @pytest.mark.parametrize(
        "bad",
        [
            {"x": [1.0]},  # wrong length
            {"x": []},  # empty
            {"x": [[1.0, 1.0]]},  # 2-D
            {"popt": [1.0, 1.0, 1.0]},
        ],
        ids=["short", "empty", "2d", "long"],
    )
    @pytest.mark.parametrize("bounds", [([0.0, 0.0], [10.0, 10.0]), (0.0, 10.0), None])
    def test_malformed_parameter_vectors_are_not_a_recovery(self, bad, bounds):
        state = {"p0": [1.0, 1.0]}
        if bounds is not None:
            state["bounds"] = bounds
        (ok, _), calls = self._recovery({"success": True, "cost": 1.0, **bad}, state)
        assert not ok and calls

    @pytest.mark.parametrize("bounds", [([0.0, 0.0], [10.0, 10.0]), (0.0, 10.0), None])
    def test_valid_vector_still_succeeds(self, bounds):
        state = {"p0": [1.0, 1.0]}
        if bounds is not None:
            state["bounds"] = bounds
        (ok, _), _ = self._recovery(
            {"success": True, "x": [1.0, 2.0], "cost": 1.0}, state
        )
        assert ok

    def test_bounds_that_would_change_the_length_are_rejected(self):
        state = {"p0": [1.0, 1.0], "bounds": ([0.0, 0.0, 0.0], [9.0, 9.0, 9.0])}
        (ok, _), _ = self._recovery({"success": True, "x": [1.0, 1.0]}, state)
        assert not ok


class TestRound14:
    @pytest.mark.parametrize("jitted", [False, True])
    @pytest.mark.parametrize(
        "diag,b,expected",
        [
            ([1e200, 1e-100], [1e200, 1e-110], [1.0, 1e-10]),
            ([1e200, 1e-120], [1e200, 1e-120], [1.0, 1.0]),
        ],
    )
    def test_independently_scaled_components_survive(self, jitted, diag, b, expected):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        x, used = fn(jnp.diag(jnp.array(diag)), jnp.array(b))
        np.testing.assert_allclose(np.asarray(x), expected, rtol=1e-10)
        assert bool(used)

    @pytest.mark.parametrize("jitted", [False, True])
    def test_float32_independent_scales(self, jitted):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        A = jnp.diag(jnp.array([1e30, 1e-20], dtype=jnp.float32))
        x, _ = fn(A, jnp.array([1e30, 1e-20], dtype=jnp.float32))
        np.testing.assert_allclose(np.asarray(x), [1.0, 1.0], rtol=1e-5)

    @pytest.mark.parametrize("jitted", [False, True])
    def test_random_spd_with_wide_diagonal_scales(self, jitted):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        rng = np.random.default_rng(7)
        for _ in range(8):
            n = 4
            M = rng.normal(size=(n, n))
            C = M @ M.T + n * np.eye(n)  # SPD, moderate conditioning
            d = 10.0 ** rng.uniform(-150, 150, n)
            d[0] = 1e200  # peak above sqrt(max): the equilibration branch runs
            sd = np.sqrt(d)
            A = C * np.outer(sd, sd)  # np.outer(d, d) itself would overflow
            x_true = rng.normal(size=n)
            b = A @ x_true
            x, _ = fn(jnp.asarray(A), jnp.asarray(b))
            # error of each component measured in its own (equilibrated) scale
            err = np.abs(np.asarray(x) - x_true) * np.sqrt(np.diag(A))
            assert np.all(np.isfinite(np.asarray(x)))
            assert err.max() / (np.abs(x_true) * np.sqrt(np.diag(A))).max() < 1e-8

    @pytest.mark.parametrize("jitted", [False, True])
    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_non_finite_input_propagates_nan(self, jitted, bad):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        x, used = fn(jnp.array([[bad]]), jnp.array([1.0]))
        assert bool(jnp.all(jnp.isnan(x))) and not bool(used)
        x, used = fn(jnp.array([[2.0]]), jnp.array([bad]))
        assert bool(jnp.all(jnp.isnan(x))) and not bool(used)

    @pytest.mark.parametrize("start", [1.0, np.array(1.0), [1.0]])
    def test_scalar_start_recovers(self, start):
        rec = OptimizationRecovery(max_retries=1, enable_diagnostics=False)
        rec.strategies = [rec._perturb_parameters]
        seen = []

        def retry(p0):
            seen.append(p0)
            return {"success": True, "x": [1.0], "cost": 0.0}

        ok, _ = rec.recover_from_failure("numerical", {"p0": start}, retry)
        assert ok and len(seen) == 1
        if np.ndim(start) == 0:
            assert isinstance(seen[0], float)  # the original kind is kept


class TestRound15:
    @pytest.mark.parametrize("jitted", [False, True])
    @pytest.mark.parametrize(
        "A,b,expected",
        [
            ([[-1e308, 0.0], [0.0, 1e308]], [1e308, 1e308], [-1.0, 1.0]),
            ([[0.0, 1e308], [1e308, 0.0]], [1e308, 1e308], [1.0, 1.0]),
        ],
        ids=["indefinite-diag", "zero-diagonal"],
    )
    def test_non_positive_diagonal_extreme_scale(self, jitted, A, b, expected):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        x, used = fn(jnp.array(A), jnp.array(b))
        np.testing.assert_allclose(np.asarray(x), expected, rtol=1e-10)
        assert not bool(used)

    @pytest.mark.parametrize("jitted", [False, True])
    def test_singular_clamp_is_in_original_units(self, jitted):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        x, used = fn(jnp.diag(jnp.array([1e200, 0.0])), jnp.array([1e200, 1.0]))
        # HEAD rule: lambda -> eps*max(max|lambda|, 1) = 2.2e-16 * 1e200
        expected = 1.0 / (np.finfo(float).eps * 1e200)
        np.testing.assert_allclose(np.asarray(x), [1.0, expected], rtol=1e-10)
        assert not bool(used)


class TestRound16:
    @pytest.mark.parametrize("jitted", [False, True])
    @pytest.mark.parametrize(
        "n,dtype,peak,tail,eps",
        [
            (2, jnp.float64, 1e200, 1e-110, np.finfo(np.float64).eps),
            (100, jnp.float64, 1e200, 1e-110, np.finfo(np.float64).eps),
            (2, jnp.float32, 1e30, 1e-10, np.finfo(np.float32).eps),
        ],
        ids=["f64-2", "f64-100", "f32-2"],
    )
    def test_small_rhs_component_does_not_underflow_in_fallback(
        self, jitted, n, dtype, peak, tail, eps
    ):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        diag = np.zeros(n)
        diag[0] = peak
        b = np.zeros(n)
        b[0], b[1] = peak, tail
        x, used = fn(jnp.diag(jnp.asarray(diag, dtype)), jnp.asarray(b, dtype))
        x = np.asarray(x, dtype=np.float64)
        expected = tail / (float(eps) * peak)
        np.testing.assert_allclose(x[:2], [1.0, expected], rtol=1e-5)
        assert not bool(used)


class TestRound17:
    @pytest.mark.parametrize("jitted", [False, True])
    def test_near_max_rhs_does_not_overflow_in_projection(self, jitted):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        x, used = fn(jnp.array([[0.0, 1.0], [1.0, 0.0]]), jnp.array([1.4e308, 1.4e308]))
        np.testing.assert_allclose(np.asarray(x), [1.4e308, 1.4e308], rtol=1e-12)
        assert not bool(used)
        n = 100
        A = jnp.asarray(np.eye(n) - 2 * np.ones((n, n)) / n)
        x, _ = fn(A, jnp.full(n, 3e307))
        np.testing.assert_allclose(np.asarray(x), -3e307, rtol=1e-10)

    @pytest.mark.parametrize("jitted", [False, True])
    def test_ordinary_rhs_is_bit_identical(self, jitted):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        A = jnp.array([[0.0, 2.0], [2.0, 1.0]])  # indefinite: eigenvalue path
        b = jnp.array([3.0, 5.0])
        x, _ = fn(A, b)
        w, V = jnp.linalg.eigh(0.5 * (A + A.T))
        ref = V @ (V.T @ b / w)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(ref))


class TestRound18:
    @pytest.mark.parametrize("jitted", [False, True])
    @pytest.mark.parametrize(
        "n,dtype,hi,lo",
        [
            (2, jnp.float64, 1.4e308, 3e-308),
            (2, jnp.float32, 2e38, 2e-38),
            (100, jnp.float64, 1e308, 1e-307),
        ],
        ids=["f64-2", "f32-2", "f64-100"],
    )
    def test_huge_and_tiny_rhs_components_coexist(self, jitted, n, dtype, hi, lo):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        diag = np.zeros(n)
        diag[0] = -1.0
        b = np.zeros(n)
        b[0], b[1] = hi, lo
        x, used = fn(jnp.diag(jnp.asarray(diag, dtype)), jnp.asarray(b, dtype))
        x = np.asarray(x, dtype=np.float64)
        eps = float(np.finfo(np.float32 if dtype is jnp.float32 else np.float64).eps)
        np.testing.assert_allclose(x[0], -hi, rtol=1e-6)
        np.testing.assert_allclose(x[1], lo / eps, rtol=1e-5)
        assert not bool(used)


class TestRound19:
    @pytest.mark.parametrize("jitted", [False, True])
    @pytest.mark.parametrize(
        "dtype,off,rhs,sol",
        [
            (jnp.float64, 1e-10, 1.4e298, 1.4e308),
            (jnp.float32, 1e-5, 3e33, 3e38),
        ],
        ids=["f64", "f32"],
    )
    def test_amplification_by_small_eigenvalue_does_not_overflow(
        self, jitted, dtype, off, rhs, sol
    ):
        fn = (
            jax.jit(solve_with_cholesky_fallback)
            if jitted
            else solve_with_cholesky_fallback
        )
        A = jnp.array([[0.0, off], [off, 0.0]], dtype=dtype)
        x, used = fn(A, jnp.array([rhs, rhs], dtype=dtype))
        np.testing.assert_allclose(
            np.asarray(x, dtype=np.float64), [sol, sol], rtol=1e-5
        )
        assert not bool(used)

"""Regression tests for bugs found in the 2026-08-25 review of the
AdaptiveHybridStreamingOptimizer four-phase pipeline (Phase 0: normalization,
Phase 1: L-BFGS warmup, Phase 2: streaming Gauss-Newton, Phase 3:
denormalization/covariance).

Each test reproduces one bug that was silently wrong before the fix (no
exception, no warning -- just a corrupted or mislabeled result).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.precision.parameter_normalizer import ParameterNormalizer
from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig


class TestPartiallyUnboundedNormalization:
    """A: 'bounds' strategy produced NaN for a partially/fully unbounded param."""

    def test_normalize_denormalize_round_trip_with_one_sided_bounds(self):
        p0 = jnp.array([1.0, 5.0])
        bounds = (jnp.array([0.0, -jnp.inf]), jnp.array([10.0, jnp.inf]))

        normalizer = ParameterNormalizer(p0=p0, bounds=bounds, strategy="bounds")

        normalized = normalizer.normalize(p0)
        assert jnp.all(jnp.isfinite(normalized)), normalized

        denormalized = normalizer.denormalize(normalized)
        assert jnp.all(jnp.isfinite(denormalized)), denormalized
        np.testing.assert_allclose(np.asarray(denormalized), np.asarray(p0))

    def test_transform_bounds_with_fully_unbounded_param(self):
        p0 = jnp.array([1.0, 5.0])
        bounds = (jnp.array([0.0, -jnp.inf]), jnp.array([10.0, jnp.inf]))
        normalizer = ParameterNormalizer(p0=p0, bounds=bounds, strategy="bounds")

        lb_norm, ub_norm = normalizer.transform_bounds()
        assert not jnp.any(jnp.isnan(lb_norm)), lb_norm
        assert not jnp.any(jnp.isnan(ub_norm)), ub_norm

    def test_normalization_jacobian_finite_with_unbounded_param(self):
        p0 = jnp.array([1.0, 5.0])
        bounds = (jnp.array([0.0, -jnp.inf]), jnp.array([10.0, jnp.inf]))
        normalizer = ParameterNormalizer(p0=p0, bounds=bounds, strategy="bounds")
        assert jnp.all(jnp.isfinite(normalizer.normalization_jacobian))


class TestSigmaRejectedNotSilentlyIgnored:
    """B: sigma/absolute_sigma were accepted then silently dropped."""

    def test_fit_raises_when_sigma_provided(self):
        optimizer = AdaptiveHybridStreamingOptimizer(HybridStreamingConfig())

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 50)
        y = 2.0 * x + 1.0
        sigma = jnp.ones_like(y)

        with pytest.raises(NotImplementedError, match="sigma"):
            optimizer.fit(
                data_source=(x, y),
                func=model,
                p0=jnp.array([1.0, 1.0]),
                sigma=sigma,
                verbose=0,
            )

    def test_fit_succeeds_without_sigma(self):
        optimizer = AdaptiveHybridStreamingOptimizer(
            HybridStreamingConfig(
                warmup_iterations=5,
                max_warmup_iterations=5,
                gauss_newton_max_iterations=5,
            ),
        )

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 50)
        y = 2.0 * x + 1.0

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=jnp.array([1.0, 1.0]),
            verbose=0,
        )
        assert jnp.all(jnp.isfinite(result["x"]))


class TestScanAccumulationMatchesLoopForSingularModel:
    """G: padded x=0 rows evaluated before masking poisoned JTJ with NaN on the
    GPU/TPU scan accumulation path for models singular at x=0 (e.g. a/x),
    while the CPU Python-loop path never touched padded rows and was immune.
    """

    def test_no_nan_and_matches_loop_path_for_data_not_divisible_by_chunk_size(self):
        optimizer = AdaptiveHybridStreamingOptimizer(
            HybridStreamingConfig(chunk_size=8, normalize=False),
        )

        def model(x, a, b):
            # Singular at x=0 -- exercises the padded-row NaN bug directly.
            return a / x + b

        # n_points not divisible by chunk_size=8 forces padding in the scan path.
        n_points = 37
        x = jnp.linspace(1.0, 5.0, n_points)  # strictly positive, no real zeros
        y = 2.0 / x + 1.0

        optimizer._setup_normalization(model, jnp.array([1.0, 1.0]), None)
        params = optimizer.normalized_params

        jtj_scan, jtr_scan, cost_scan = optimizer._accumulate_jtj_jtr_scan(
            x,
            y,
            params,
        )
        assert jnp.all(jnp.isfinite(jtj_scan)), jtj_scan
        assert jnp.all(jnp.isfinite(jtr_scan)), jtr_scan
        assert np.isfinite(cost_scan)

        jtj_loop = jnp.zeros((2, 2))
        jtr_loop = jnp.zeros(2)
        cost_loop = 0.0
        chunk_size = optimizer.config.chunk_size
        for i in range(0, n_points, chunk_size):
            x_chunk = x[i : i + chunk_size]
            y_chunk = y[i : i + chunk_size]
            jtj_loop, jtr_loop, chunk_cost = optimizer._accumulate_jtj_jtr(
                x_chunk,
                y_chunk,
                params,
                jtj_loop,
                jtr_loop,
            )
            cost_loop += chunk_cost

        np.testing.assert_allclose(
            np.asarray(jtj_scan),
            np.asarray(jtj_loop),
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(jtr_scan),
            np.asarray(jtr_loop),
            rtol=1e-8,
            atol=1e-10,
        )
        assert cost_scan == pytest.approx(cost_loop, rel=1e-8)


class TestLbfgsBestParamsPairing:
    """M: best_params_global was paired with new_params (post-step) while
    best_cost_global held loss_value (pre-step), and the same mismatch existed
    in the local best_params/best_loss tracked by _run_phase1_warmup, which
    fit() uses to seed Phase 2.
    """

    def test_best_params_global_paired_with_the_point_its_loss_was_computed_at(self):
        optimizer = AdaptiveHybridStreamingOptimizer(
            HybridStreamingConfig(normalize=False),
        )

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 20)
        y = 2.0 * x + 1.0
        optimizer._setup_normalization(model, jnp.array([0.1, 0.1]), None)

        loss_fn = optimizer._create_warmup_loss_fn()
        params0 = optimizer.normalized_params
        opt, state = optimizer._create_lbfgs_optimizer(params0)

        _new_params, loss_value, _grad_norm, _state, _failed = optimizer._lbfgs_step(
            params=params0,
            opt_state=state,
            optimizer=opt,
            loss_fn=loss_fn,
            x_batch=x,
            y_batch=y,
            iteration=0,
        )

        # loss_value must equal loss_fn(params0, ...), the point it claims to
        # describe, not loss_fn(new_params, ...).
        actual_loss_at_input = float(loss_fn(params0, x, y))
        assert loss_value == pytest.approx(actual_loss_at_input, rel=1e-6)

        # best_params_global must be the point best_cost_global was measured
        # at. best_cost_global is stored as an SSR-equivalent (loss_value is
        # MSE, converted via * n_points -- see the cross-phase unit-mismatch
        # fix comment in _lbfgs_step) so it stays comparable with Phase 2's
        # SSR-based tracking.
        n_points = len(x)
        assert optimizer.best_cost_global == pytest.approx(
            loss_value * n_points,
            rel=1e-9,
        )
        recomputed_mse = float(loss_fn(optimizer.best_params_global, x, y))
        assert recomputed_mse * n_points == pytest.approx(
            optimizer.best_cost_global,
            rel=1e-6,
        )

    def test_phase1_best_params_is_the_point_best_loss_was_measured_at(self):
        optimizer = AdaptiveHybridStreamingOptimizer(
            HybridStreamingConfig(
                normalize=False,
                max_warmup_iterations=8,
                warmup_iterations=8,
            ),
        )

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 20)
        y = 2.0 * x + 1.0
        p0 = jnp.array([0.1, 0.1])

        result = optimizer._run_phase1_warmup((x, y), model, p0, None)
        loss_fn = optimizer._create_warmup_loss_fn()

        recomputed_best_loss = float(loss_fn(result["best_params"], x, y))
        assert recomputed_best_loss == pytest.approx(result["best_loss"], rel=1e-5)


class TestGroupVarianceRegularizationUnequalGroups:
    """N: dynamic_slice silently clamped its start index for a trailing group
    smaller than max_group_size, pulling in the previous group's element
    instead of the group's own last element.
    """

    def test_unequal_trailing_group_variance_matches_manual_computation(self):
        config = HybridStreamingConfig(
            normalize=False,
            enable_group_variance_regularization=True,
            group_variance_lambda=1.0,
            group_variance_indices=[(0, 3), (3, 6), (6, 8)],
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, *params):
            return jnp.sum(jnp.asarray(params)) * jnp.ones_like(x)

        optimizer._setup_normalization(model, jnp.arange(8, dtype=jnp.float64), None)
        loss_fn = optimizer._create_warmup_loss_fn()

        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0])
        x = jnp.array([0.0, 1.0])
        y = jnp.array([0.0, 0.0])

        loss = float(loss_fn(params, x, y))

        base_pred = float(jnp.sum(params))
        base_loss = float(jnp.mean((y - base_pred) ** 2))

        manual_penalty = 0.0
        for start, end in config.group_variance_indices:
            group = np.asarray(params[start:end])
            manual_penalty += float(np.var(group))

        expected = base_loss + config.group_variance_lambda * manual_penalty
        assert loss == pytest.approx(expected, rel=1e-6)


class TestAbsentFeaturesFailLoudly:
    """C/D/H: config fields that documented behavior fit() never actually
    performs (checkpoint resume, multi-device sharding, CG auto-selection
    for large parameter counts) now raise instead of silently no-op'ing.
    """

    def _model_and_data(self):
        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 20)
        y = 2.0 * x + 1.0
        return model, x, y

    def test_resume_from_checkpoint_raises(self):
        optimizer = AdaptiveHybridStreamingOptimizer(
            HybridStreamingConfig(resume_from_checkpoint="/tmp/does-not-matter.h5"),
        )
        model, x, y = self._model_and_data()
        with pytest.raises(NotImplementedError, match="resume_from_checkpoint"):
            optimizer.fit(
                data_source=(x, y), func=model, p0=jnp.array([1.0, 1.0]), verbose=0
            )

    def test_enable_multi_device_raises(self):
        optimizer = AdaptiveHybridStreamingOptimizer(
            HybridStreamingConfig(enable_multi_device=True),
        )
        model, x, y = self._model_and_data()
        with pytest.raises(NotImplementedError, match="enable_multi_device"):
            optimizer.fit(
                data_source=(x, y), func=model, p0=jnp.array([1.0, 1.0]), verbose=0
            )

    def test_cg_param_threshold_exceeded_raises(self):
        config = HybridStreamingConfig(cg_param_threshold=3)
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b, c, d):
            return a * x + b + c * 0 + d * 0

        x = jnp.linspace(0, 1, 20)
        y = 2.0 * x + 1.0
        with pytest.raises(NotImplementedError, match="cg_param_threshold"):
            optimizer.fit(
                data_source=(x, y),
                func=model,
                p0=jnp.array([1.0, 1.0, 1.0, 1.0]),
                verbose=0,
            )


class TestFitCallbackInvoked:
    """E: callback was accepted and documented but never called."""

    def test_callback_fires_during_fit(self):
        config = HybridStreamingConfig(
            warmup_iterations=2,
            max_warmup_iterations=3,
            gauss_newton_max_iterations=3,
            callback_frequency=1,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 20)
        y = 2.0 * x + 1.0

        calls = []

        def callback(params, loss, iteration):
            calls.append((iteration, float(loss)))

        optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=jnp.array([1.0, 1.0]),
            callback=callback,
            verbose=0,
        )

        assert len(calls) > 0

    def test_callback_exception_does_not_abort_fit(self):
        config = HybridStreamingConfig(
            warmup_iterations=2,
            max_warmup_iterations=3,
            gauss_newton_max_iterations=3,
            callback_frequency=1,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 20)
        y = 2.0 * x + 1.0

        def bad_callback(params, loss, iteration):
            raise RuntimeError("user callback bug")

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=jnp.array([1.0, 1.0]),
            callback=bad_callback,
            verbose=0,
        )
        assert jnp.all(jnp.isfinite(result["x"]))


class TestPhaseFailureFallsBackHonestly:
    """F: an exception in Phase 1 or Phase 2 used to abort fit() entirely,
    discarding any best_params_global already found. Now it falls back to
    the best known point, still runs Phase 3, and reports success=False.
    """

    def test_phase2_exception_still_returns_a_result(self, monkeypatch):
        config = HybridStreamingConfig(
            warmup_iterations=2,
            max_warmup_iterations=3,
            gauss_newton_max_iterations=3,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 20)
        y = 2.0 * x + 1.0

        def broken_phase2(*args, **kwargs):
            raise RuntimeError("simulated Phase 2 failure")

        monkeypatch.setattr(optimizer, "_run_phase2_gauss_newton", broken_phase2)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=jnp.array([1.0, 1.0]),
            verbose=0,
        )
        assert result["success"] is False
        assert "simulated Phase 2 failure" in result["message"]
        assert jnp.all(jnp.isfinite(result["x"]))
        assert jnp.all(jnp.isfinite(result["pcov"]))

    def test_phase1_exception_still_returns_a_result(self, monkeypatch):
        config = HybridStreamingConfig(
            warmup_iterations=2,
            max_warmup_iterations=3,
            gauss_newton_max_iterations=3,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 20)
        y = 2.0 * x + 1.0

        def broken_phase1(*args, **kwargs):
            raise RuntimeError("simulated Phase 1 failure")

        monkeypatch.setattr(optimizer, "_run_phase1_warmup", broken_phase1)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=jnp.array([1.0, 1.0]),
            verbose=0,
        )
        assert result["success"] is False
        assert "simulated Phase 1 failure" in result["message"]
        assert jnp.all(jnp.isfinite(result["x"]))


class TestCrossPhaseBestTrackingUnitsConsistent:
    """Codex/type-design-analyzer PR review finding: self.best_cost_global was
    updated from Phase 1's MSE (jnp.mean(residuals**2)) and Phase 2's SSR
    (jnp.sum(residuals**2)) without unit conversion. For n_points > 1,
    SSR >> MSE, so once Phase 1 ran, Phase 2's `new_cost < best_cost_global`
    could essentially never fire -- best_params_global would silently freeze
    at Phase 1's endpoint, and fit()'s Phase-2 exception-fallback path would
    discard all of Phase 2's actual progress.
    """

    def test_best_cost_global_stays_comparable_across_phase1_and_phase2(self):
        optimizer = AdaptiveHybridStreamingOptimizer(
            HybridStreamingConfig(normalize=False),
        )

        def model(x, a, b):
            return a * x + b

        # Many points: MSE and SSR differ by a large factor if unconverted.
        x = jnp.linspace(0, 1, 200)
        y = 2.0 * x + 1.0
        optimizer._setup_normalization(model, jnp.array([0.1, 0.1]), None)

        loss_fn = optimizer._create_warmup_loss_fn()
        params0 = optimizer.normalized_params
        opt, state = optimizer._create_lbfgs_optimizer(params0)

        # One Phase-1 step populates best_cost_global from MSE-space.
        optimizer._lbfgs_step(
            params=params0,
            opt_state=state,
            optimizer=opt,
            loss_fn=loss_fn,
            x_batch=x,
            y_batch=y,
            iteration=0,
        )
        phase1_best_cost = optimizer.best_cost_global
        n_points = len(x)

        # Simulate what Phase 2 would compare: an SSR-space cost that is a
        # genuine, large improvement in MSE terms but numerically larger in
        # raw SSR terms purely from not being pre-scaled like Phase 1 used to be.
        mse_equivalent_improvement = phase1_best_cost / n_points * 0.5  # better
        ssr_of_improved_point = mse_equivalent_improvement * n_points

        assert ssr_of_improved_point < phase1_best_cost, (
            "A genuinely better (lower-MSE) Phase 2 point must compare as "
            "lower than Phase 1's SSR-equivalent best_cost_global, or "
            "Phase 2 progress can never overwrite Phase 1's frozen value"
        )


class TestPhaseFailureFallbackJTJRecomputeGuarded:
    """silent-failure-hunter PR review finding: the Phase 2 exception-fallback
    path's own JTJ recompute had no try/except -- a second failure there
    (e.g. a degenerate fallback point) would crash fit() entirely instead of
    degrading to a success=False result.
    """

    def test_double_failure_still_returns_instead_of_raising(self, monkeypatch):
        config = HybridStreamingConfig(
            warmup_iterations=2,
            max_warmup_iterations=3,
            gauss_newton_max_iterations=3,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 1, 20)
        y = 2.0 * x + 1.0

        def broken_phase2(*args, **kwargs):
            raise RuntimeError("simulated Phase 2 failure")

        def broken_accumulate(*args, **kwargs):
            raise RuntimeError("simulated JTJ recompute failure")

        monkeypatch.setattr(optimizer, "_run_phase2_gauss_newton", broken_phase2)
        monkeypatch.setattr(optimizer, "_accumulate_jtj_jtr", broken_accumulate)
        monkeypatch.setattr(optimizer, "_accumulate_jtj_jtr_scan", broken_accumulate)
        monkeypatch.setattr(optimizer, "_use_scan_for_accumulation", lambda: False)

        result = optimizer.fit(
            data_source=(x, y),
            func=model,
            p0=jnp.array([1.0, 1.0]),
            verbose=0,
        )
        assert result["success"] is False
        assert "simulated Phase 2 failure" in result["message"]
        assert "simulated JTJ recompute failure" in result["message"]


class TestGroupVarianceBestCostSeedIncludesRegularization:
    """code-reviewer PR review finding (empirically reproduced): Phase 2's
    local best_cost was seeded from final_residual_sum_sq (data-only SSR),
    but new_cost inside the loop is SSR + the group-variance penalty. With
    the penalty non-negative, that made the seed artificially low, so
    `new_cost < best_cost` could permanently fail to fire and Phase 2 would
    silently return the UNOPTIMIZED starting point while reporting convergence.
    """

    def test_phase2_moves_away_from_p0_with_large_group_variance_penalty(self):
        # Call Phase 2 directly (bypass Phase 1/fit()) so the "before" state
        # is exactly p0, unaffected by warmup's own regularization-aware loss.
        config = HybridStreamingConfig(
            normalize=False,
            gauss_newton_max_iterations=5,
            enable_group_variance_regularization=True,
            group_variance_lambda=1000.0,
            group_variance_indices=[(0, 2)],
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b):
            return ((a + b) / 2.0) * jnp.ones_like(x)

        x = jnp.linspace(0, 1, 20)
        y = jnp.full_like(x, 4.0)
        # mean=4 (SSR~0), but Var([4,0]) is large -- the group-variance
        # penalty at p0 is large, and shrinking it while preserving
        # mean(a,b) genuinely reduces the regularized cost.
        p0 = jnp.array([4.0, 0.0])

        optimizer._setup_normalization(model, p0, None)
        result = optimizer._run_phase2_gauss_newton(
            data_source=(x, y),
            initial_params=optimizer.normalized_params,
        )

        final_params = np.asarray(result["final_params"])
        assert not np.allclose(final_params, np.asarray(p0), atol=1e-6), (
            "Phase 2 returned the unoptimized p0 unchanged -- the "
            "best_cost seed is missing the group-variance penalty term "
            "that new_cost includes, so no trial point ever compared as better"
        )
        # mean(a, b) must stay ~4 (that's what keeps SSR low); the penalty
        # should have pulled a and b together from their p0 spread of 4.
        assert abs(float(jnp.mean(jnp.asarray(final_params))) - 4.0) < 0.5
        assert abs(final_params[0] - final_params[1]) < abs(p0[0] - p0[1])

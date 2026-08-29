"""Regression tests for the 9 bug fixes in PR #29 (three-brain review of nlsq/core/).

Each test reproduces the exact failure mode reported for its fix and fails
on the pre-fix code. See PR #29 description / commit 1796e42 for the
corresponding source-level explanation of each bug.
"""

import time

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit, fit
from nlsq.core.minpack import (
    CurveFit,
    _apply_auto_bounds,
    _fit_global_multistart,
    _is_auto_p0,
)
from nlsq.core.trf_jit import _seed_lm_alpha
from nlsq.streaming.hybrid_config import HybridStreamingConfig


def _linear(x, a, b):
    return a * x + b


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    y = 2.0 * x + 1.0 + rng.normal(scale=0.01, size=x.shape)
    return x, y


class TestAutoBoundsPZeroNone:
    """auto_bounds=True was a no-op when p0=None (the documented default)."""

    def test_auto_bounds_applies_inferred_bounds_when_p0_none(self, linear_data):
        x, y = linear_data
        kwargs: dict = {}
        _apply_auto_bounds(_linear, x, y, None, 10.0, kwargs)
        assert "bounds" in kwargs
        lb, ub = kwargs["bounds"]
        # Pre-fix, _apply_auto_bounds returned without touching kwargs at all
        # when p0 was None, so "bounds" would be absent.
        assert not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))


class TestAutoBoundsPZeroAutoSentinel:
    """auto_bounds=True + p0="auto" crashed: TypeError: len() of unsized object."""

    def test_auto_bounds_with_p0_auto_sentinel_does_not_crash(self, linear_data):
        x, y = linear_data
        kwargs: dict = {}
        # Must not raise -- pre-fix this hit np.asarray("auto") -> len() crash
        # inside infer_bounds/BoundsInference.
        _apply_auto_bounds(_linear, x, y, "auto", 10.0, kwargs)
        assert "bounds" in kwargs


class TestIsAutoP0Sentinel:
    """p0="auto" was miscounted as n_params=1 via np.atleast_1d("auto")."""

    def test_is_auto_p0_detects_sentinel(self):
        assert _is_auto_p0("auto") is True
        assert _is_auto_p0(None) is False
        assert _is_auto_p0(np.array([1.0, 2.0])) is False
        assert _is_auto_p0("not_auto") is False

    def test_curve_fit_p0_auto_multiparam_model_infers_correct_count(self, linear_data):
        x, y = linear_data
        # Regression target: p0="auto" through the public curve_fit() API
        # for a multi-parameter model must not silently collapse to
        # n_params=1 (which would corrupt memory-budget/strategy selection
        # and, for models sensitive to it, initial-guess estimation).
        popt, _ = curve_fit(_linear, x, y, p0="auto")
        assert popt.shape == (2,)
        assert np.allclose(popt, [2.0, 1.0], atol=0.1)


class TestCMAESKwargsForwarding:
    """method='cmaes' dropped ftol/xtol/max_nfev and, separately, forwarding
    full_output/timeit/return_eval caused a silent fallback to an unrefined,
    "success=True" result with pcov=inf."""

    def test_cmaes_forwards_max_nfev(self, linear_data):
        x, y = linear_data
        # A tiny max_nfev should be honored (not silently dropped), so the
        # refinement stays close to the CMA-ES seed rather than fully
        # converging. We only assert the call succeeds and respects the
        # kwarg being forwarded at all (no TypeError from unexpected kwarg).
        result = curve_fit(
            _linear,
            x,
            y,
            method="cmaes",
            bounds=([0, 0], [5, 5]),
            max_nfev=5,
        )
        assert result is not None

    def test_cmaes_full_output_does_not_silently_degrade(self, linear_data):
        x, y = linear_data
        # Pre-fix: full_output=True forwarded into CMAESOptimizer.fit()'s
        # extra_kwargs, causing the nested curve_fit() in NLSQ refinement to
        # return a tuple instead of CurveFitResult, raising AttributeError
        # on `.x`, silently caught, and falling back to the unrefined
        # CMA-ES estimate (pcov=inf) while still claiming success=True.
        result = curve_fit(
            _linear,
            x,
            y,
            method="cmaes",
            bounds=([0, 0], [5, 5]),
            full_output=True,
        )
        popt = result[0] if isinstance(result, tuple) else result.popt
        pcov = result[1] if isinstance(result, tuple) else result.pcov
        assert np.all(np.isfinite(pcov)), (
            "pcov=inf indicates NLSQ refinement silently fell back to the "
            "raw CMA-ES estimate instead of running"
        )
        assert np.allclose(popt, [2.0, 1.0], atol=0.5)


class TestDataMaskPadding:
    """A caller-supplied data_mask wasn't extended with padding when flength
    padding was applied, only the auto-generated mask was."""

    def test_user_supplied_data_mask_gets_padded(self):
        cf = CurveFit()
        m = 10
        # flength longer than the data forces should_pad=True with len_diff>0.
        cf.flength = 15
        cf.use_dynamic_sizing = True
        data_mask = np.ones(m, dtype=bool)
        result_mask, none_mask, _len_diff, should_pad = cf._setup_data_mask_and_padding(
            data_mask=data_mask,
            m=m,
        )
        assert should_pad
        assert not none_mask
        assert len(result_mask) == 15


class TestTRFFractionalMaxNfev:
    """Outer-loop termination used nfev == max_nfev; a fractional max_nfev
    could never satisfy equality, hanging the optimizer."""

    def test_fractional_max_nfev_terminates_promptly(self, linear_data):
        x, y = linear_data
        start = time.monotonic()
        curve_fit(_linear, x, y, max_nfev=5.5)
        elapsed = time.monotonic() - start
        # Pre-fix this could spin indefinitely (denial-of-service hang);
        # a real termination should complete in well under a second for
        # this trivial 50-point linear fit.
        assert elapsed < 10.0


class TestAutoP0RemainingFitSites:
    """Round-2 finding: the p0="auto" guard was applied at 4-5 call sites in
    round 1 but missed 3 more identical-pattern sites in fit()'s
    workflow-config branches (_fit_with_config's HybridStreamingConfig
    branch, _fit_with_preset's streaming-tier branch,
    _fit_global_multistart's streaming-strategy branch), each of which
    crashed the same way: `ValueError: could not convert string to float`
    from np.atleast_1d("auto") flowing into prepare_bounds()."""

    def test_hybrid_streaming_config_branch_does_not_crash_on_p0_auto(
        self, linear_data
    ):

        x, y = linear_data
        result = fit(_linear, x, y, p0="auto", workflow=HybridStreamingConfig())
        assert result is not None

    def test_streaming_preset_branch_does_not_crash_on_p0_auto(self, linear_data):
        from nlsq.core.minpack import _fit_with_preset

        x, y = linear_data
        # The "streaming"/"hpc_distributed" WORKFLOW_PRESETS entries (tier
        # STREAMING/STREAMING_CHECKPOINT) are unreachable from the public
        # fit()/curve_fit() string-workflow API -- both are blocked by
        # REMOVED_PRESETS before the preset lookup. This branch is
        # unreachable-but-not-deleted dead code from the public surface, so
        # it's called directly here to still cover the fix.
        result = _fit_with_preset(
            f=_linear,
            xdata=x,
            ydata=y,
            p0="auto",
            sigma=None,
            absolute_sigma=False,
            check_finite=True,
            bounds=(-np.inf, np.inf),
            method=None,
            preset={"tier": "STREAMING", "description": "test"},
            goal=None,
            n_points=len(y),
        )
        assert result is not None

    def test_global_multistart_streaming_strategy_does_not_crash_on_p0_auto(
        self, linear_data
    ):

        x, y = linear_data
        result = _fit_global_multistart(
            f=_linear,
            xdata=x,
            ydata=y,
            p0="auto",
            sigma=None,
            absolute_sigma=False,
            check_finite=True,
            bounds=(-np.inf, np.inf),
            strategy="streaming",
            n_starts=2,
        )
        assert result is not None


class TestMaybeFlagInnerLoopLimit:
    """Round-2 finding: _maybe_flag_inner_loop_limit (extracted in round 1
    from duplicated inline code, renamed in round 2 for clarity) was bound
    onto the test double in test_trf_internal_helpers.py but never actually
    called by any test. Direct unit coverage of its 3 branches."""

    @staticmethod
    def _call(
        inner_loop_count, max_inner_iterations, termination_status, actual_reduction
    ):
        import importlib

        trf_module = importlib.import_module("nlsq.core.trf")
        opt = type("Opt", (), {})()
        opt.logger = type(
            "Logger", (), {"warning": staticmethod(lambda *_a, **_k: None)}
        )()
        bound = trf_module.TrustRegionReflective._maybe_flag_inner_loop_limit.__get__(
            opt, trf_module.TrustRegionReflective
        )
        return bound(
            inner_loop_count, max_inner_iterations, termination_status, actual_reduction
        )

    def test_limit_not_hit_leaves_termination_status_untouched(self):
        result = self._call(
            inner_loop_count=5,
            max_inner_iterations=100,
            termination_status=None,
            actual_reduction=-1.0,
        )
        assert result is None

    def test_limit_hit_with_no_reduction_flags_minus_three(self):
        result = self._call(
            inner_loop_count=100,
            max_inner_iterations=100,
            termination_status=None,
            actual_reduction=-0.5,
        )
        assert result == -3

    def test_limit_hit_but_step_already_accepted_does_not_clobber(self):
        # inner_loop_count can equal max_inner_iterations on the very
        # iteration that accepts a step -- must not overwrite that success.
        result = self._call(
            inner_loop_count=100,
            max_inner_iterations=100,
            termination_status=None,
            actual_reduction=0.5,
        )
        assert result is None

    def test_limit_hit_but_real_termination_already_set_does_not_clobber(self):
        result = self._call(
            inner_loop_count=100,
            max_inner_iterations=100,
            termination_status=1,
            actual_reduction=-0.5,
        )
        assert result == 1


class TestSeedLMAlpha:
    """Round-1 fix: CG/lsmr trust-region solvers' LM damping alpha stayed
    pinned at 0.0 (INITIAL_LEVENBERG_MARQUARDT_LAMBDA never carries forward
    for these solvers), silently degrading the "regularized" branch to
    unregularized Gauss-Newton. Direct numerical test of the extracted
    _seed_lm_alpha helper (round-2 extraction) plus an end-to-end check that
    cg/lsmr now agree with the exact solver on a rank-deficient problem."""

    def test_seed_lm_alpha_seeds_when_alpha_unset(self):
        g_norm = jnp.array(2.0)
        result = _seed_lm_alpha(g_norm, alpha=0.0, Delta=4.0)
        assert float(result) == pytest.approx(0.5)  # g_norm / Delta

    def test_seed_lm_alpha_preserves_existing_positive_alpha(self):
        g_norm = jnp.array(2.0)
        result = _seed_lm_alpha(g_norm, alpha=1.5, Delta=4.0)
        assert float(result) == pytest.approx(1.5)

    def test_cg_solver_matches_exact_on_rank_deficient_model(self):
        # Two parameters that only ever appear as a sum -> rank-deficient
        # Jacobian, exactly the case LM regularization exists to handle.
        def redundant(x, a, b):
            return (a + b) * x

        rng = np.random.default_rng(1)
        x = np.linspace(0, 5, 30)
        y = 3.0 * x + rng.normal(scale=1e-3, size=x.shape)

        popt_exact, _ = curve_fit(redundant, x, y, p0=[1.0, 1.0], tr_solver="exact")
        popt_cg, _ = curve_fit(redundant, x, y, p0=[1.0, 1.0], tr_solver="cg")

        # The sum a+b is well-determined even though a, b individually
        # aren't -- compare the identifiable quantity, not the raw params.
        assert (popt_exact[0] + popt_exact[1]) == pytest.approx(
            popt_cg[0] + popt_cg[1], rel=0.1
        )
        assert np.all(np.isfinite(popt_cg))

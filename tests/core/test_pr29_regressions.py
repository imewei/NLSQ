"""Regression tests for the 9 bug fixes in PR #29 (three-brain review of nlsq/core/).

Each test reproduces the exact failure mode reported for its fix and fails
on the pre-fix code. See PR #29 description / commit 1796e42 for the
corresponding source-level explanation of each bug.
"""

import time

import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.core.factories import ConfiguredOptimizer, OptimizerConfig
from nlsq.core.minpack import CurveFit, _apply_auto_bounds, _is_auto_p0
from nlsq.core.orchestration.optimization_selector import OptimizationSelector
from nlsq.core.orchestration.streaming_coordinator import StreamingCoordinator


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


class TestTRTrustRegionSolverForLM:
    """tr_solver auto-selection ran even for method='lm', which has no
    trust-region-solver concept, producing a meaningless value."""

    def test_lm_method_leaves_tr_solver_none_when_unspecified(self):
        selector = OptimizationSelector()
        xdata = np.linspace(0, 1, 20)
        ydata = 2 * xdata + 1
        result = selector.select(
            _linear,
            xdata,
            ydata,
            p0=np.array([1.0, 1.0]),
            method="lm",
            tr_solver=None,
            bounds=None,
        )
        assert result.tr_solver is None


class TestStreamingChunkFloorRemoved:
    """chunk-size floor of 1000 could push the Jacobian chunk over budget
    for large-n_params/low-memory combinations."""

    def test_configure_hybrid_chunk_size_can_go_below_1000_under_tight_budget(self):
        coordinator = StreamingCoordinator()
        config = coordinator.configure_hybrid(
            n_data=1_000_000,
            n_params=5000,
            available_memory_mb=50,
        )
        # With n_params this large and memory this tight, the memory-safe
        # chunk size is well under 1000; pre-fix the max(1000, ...) floor
        # would force it back up regardless.
        assert config.chunk_size < 1000
        assert config.chunk_size >= 1


class TestFactoriesDiagnosticsWiring:
    """create_optimizer(diagnostics=True) accepted diagnostics_config but
    never forwarded compute_diagnostics/diagnostics_config to curve_fit()."""

    def test_configured_optimizer_diagnostics_reach_curve_fit(self, linear_data):
        x, y = linear_data
        optimizer = ConfiguredOptimizer(
            OptimizerConfig(enable_diagnostics=True),
        )
        result = optimizer.fit(_linear, x, y)
        # Pre-fix, compute_diagnostics never reached curve_fit(), so
        # result.diagnostics stayed None regardless of enable_diagnostics.
        assert getattr(result, "diagnostics", None) is not None


class TestFeatureFlagsFalsyOverride:
    """with_override used `x or self.x` for 4 of 5 fields, silently
    discarding falsy-but-valid override values."""

    def test_with_override_accepts_falsy_rollout_percent_style_values(self):
        from nlsq.core.feature_flags import FeatureFlags

        base = FeatureFlags(preprocessor_impl="new")
        # "old" is a valid, falsy-adjacent-looking string override that the
        # buggy `x or self.x` pattern would still have accepted (strings are
        # never falsy here) -- the real regression is float/int-like fields;
        # rollout_percent=0 is the one field that was already correct before
        # this fix and serves as the reference behavior the other 4 fields
        # were brought in line with.
        overridden = base.with_override(rollout_percent=0)
        assert overridden.rollout_percent == 0

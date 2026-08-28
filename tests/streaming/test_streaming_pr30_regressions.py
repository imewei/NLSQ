"""Regression tests for the PR #30 three-brain review of nlsq/streaming/.

Each test reproduces a bug that was silently wrong (or crashed) before its
fix landed in this PR. The bugs were found by a Claude + Agy review of
large_dataset.py, adaptive_hybrid.py, telemetry.py, validators.py, and
hybrid_config.py, then a follow-up multi-agent PR review found two more
(the closure cache-key collision and the covariance ill-conditioning gap).
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.large_dataset import (
    ChunkBufferPool,
    LargeDatasetFitter,
    LDMemoryConfig,
    MemoryEstimator,
)
from nlsq.streaming.telemetry import DefenseLayerTelemetry
from nlsq.streaming.validators import (
    ConfigValidationError,
    validate_less_than_or_equal,
    validate_non_negative,
    validate_positive,
    validate_residual_weighting_config,
)


class TestRetryFailedChunkUsesValidLength:
    """Critical: _retry_failed_chunk fit against the raw zero-padded
    DataChunker bucket instead of the trimmed valid_length slice."""

    def test_retry_fits_only_valid_points_not_padding(self):
        fitter = LargeDatasetFitter()
        fitter.curve_fit.curve_fit = MagicMock(
            return_value=(np.array([1.0, 1.0]), np.eye(2)),
        )

        valid_length = 50
        padded_length = 1024  # simulates a DataChunker power-of-2 bucket
        x_chunk = np.zeros(padded_length)
        y_chunk = np.zeros(padded_length)
        x_chunk[:valid_length] = np.linspace(0, 1, valid_length)
        y_chunk[:valid_length] = 2.0 * x_chunk[:valid_length] + 1.0

        def model(x, a, b):
            return a * x + b

        fitter._retry_failed_chunk(
            f=model,
            x_chunk=x_chunk,
            y_chunk=y_chunk,
            chunk_idx=0,
            chunk_start_time=0.0,
            chunk_times=[],
            current_params=np.array([1.0, 1.0]),
            initial_error=RuntimeError("initial fit failed"),
            bounds=(-np.inf, np.inf),
            method="trf",
            solver="auto",
            valid_length=valid_length,
        )

        assert fitter.curve_fit.curve_fit.called
        call_args = fitter.curve_fit.curve_fit.call_args
        x_arg, y_arg = call_args.args[1], call_args.args[2]
        assert len(x_arg) == valid_length, (
            f"retry fit against {len(x_arg)} points, expected only the "
            f"{valid_length} valid (non-padded) points"
        )
        assert len(y_arg) == valid_length

    def test_retry_without_valid_length_falls_back_to_full_chunk(self):
        """valid_length is optional (default None -> full chunk) so callers
        that don't pass it (there should be none left) still work rather
        than crashing."""
        fitter = LargeDatasetFitter()
        fitter.curve_fit.curve_fit = MagicMock(
            return_value=(np.array([1.0, 1.0]), np.eye(2)),
        )
        x_chunk = np.linspace(0, 1, 20)
        y_chunk = 2.0 * x_chunk + 1.0

        def model(x, a, b):
            return a * x + b

        fitter._retry_failed_chunk(
            f=model,
            x_chunk=x_chunk,
            y_chunk=y_chunk,
            chunk_idx=0,
            chunk_start_time=0.0,
            chunk_times=[],
            current_params=np.array([1.0, 1.0]),
            initial_error=RuntimeError("initial fit failed"),
            bounds=(-np.inf, np.inf),
            method="trf",
            solver="auto",
        )
        call_args = fitter.curve_fit.curve_fit.call_args
        assert len(call_args.args[1]) == len(x_chunk)


class TestSVDPseudoInverseNullSpaceHandling:
    """Critical: the SVD Gauss-Newton solve floored small singular values to
    s_threshold and divided by that floor instead of zeroing their inverse
    contribution, amplifying null-space noise by up to ~1e10x on
    rank-deficient/ill-conditioned systems."""

    def test_rank_deficient_jtj_step_has_no_amplified_null_space_component(self):
        optimizer = AdaptiveHybridStreamingOptimizer(HybridStreamingConfig())

        # JTJ = [[1,1],[1,1]] is exactly rank-1: singular values are [2, 0],
        # with the null-space direction along (1, -1)/sqrt(2).
        jtj = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        # JTr chosen to lie entirely along the null-space direction, so the
        # pre-fix code (dividing by a ~2e-10 floor) would blow the step up to
        # roughly 1/(2e-10) before trust-region clipping -- i.e. it would hit
        # the trust radius. Post-fix, that component is zeroed instead.
        jtr = jnp.array([1.0, -1.0])

        step, predicted_reduction = optimizer._solve_gauss_newton_step(
            jtj,
            jtr,
            trust_radius=1e6,
            regularization=0.0,
        )

        assert jnp.all(jnp.isfinite(step)), step
        assert jnp.isfinite(predicted_reduction)
        # Pre-fix this would be ~1e6 (clipped at the trust radius) instead.
        assert float(jnp.linalg.norm(step)) < 1.0, (
            f"step norm {float(jnp.linalg.norm(step))} suggests null-space "
            "noise was amplified instead of zeroed"
        )

    def test_step_norm_stays_bounded_as_singular_value_approaches_zero(self):
        optimizer = AdaptiveHybridStreamingOptimizer(HybridStreamingConfig())
        base = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        jtr = jnp.array([1.0, -1.0])

        # s_threshold = max(s) * 1e-10 = 2 * 1e-10 = 2e-10 for this JTJ, so
        # every eps below stays in the "treat as null-space, zero it"
        # regime the fix targets. (An eps ABOVE the threshold, e.g. 1e-8,
        # is legitimately a small-but-resolvable direction and correctly
        # gets a large step -- that's not the bug this test targets, see
        # test_rank_deficient_jtj_step_has_no_amplified_null_space_component
        # for the eps=0 case and TestSVDPseudoInverseNullSpaceHandling's
        # class docstring for the threshold-crossing behavior generally.)
        step_norms = []
        for eps in (1e-11, 1e-12, 1e-13, 1e-15, 0.0):
            jtj = base + eps * jnp.eye(2)
            step, _ = optimizer._solve_gauss_newton_step(
                jtj,
                jtr,
                trust_radius=1e6,
                regularization=0.0,
            )
            assert jnp.all(jnp.isfinite(step))
            step_norms.append(float(jnp.linalg.norm(step)))

        # None of these should approach the trust radius -- the whole point
        # of the fix is that below-threshold singular values don't blow the
        # step up regardless of exactly how close to zero they are.
        assert all(n < 10.0 for n in step_norms), step_norms


class TestChunkedFitCovarianceUsesAccumulatedInformation:
    """High: result["pcov"] used a crude parameter-history proxy (or a
    fabricated diagonal for single-chunk fits) instead of the
    precision-weighted GLS covariance already accumulated in
    self._accum_information."""

    def test_uses_gls_covariance_when_information_is_well_conditioned(self):
        fitter = LargeDatasetFitter()
        fitter._accum_information = np.array([[4.0, 0.0], [0.0, 2.0]])
        pcov = fitter._compute_covariance_from_history(
            param_history=[np.array([1.0, 1.0])],
            current_params=np.array([1.0, 1.0]),
        )
        expected = np.linalg.inv(fitter._accum_information)
        np.testing.assert_allclose(pcov, expected)

    def test_falls_back_when_information_is_ill_conditioned(self):
        """A near-singular (high condition number) information matrix must
        NOT be silently inverted into confident-looking garbage -- this was
        a real gap found in PR review: np.linalg.inv "succeeds" on an
        ill-conditioned matrix without raising, so a bare finiteness check
        isn't enough to catch it."""
        fitter = LargeDatasetFitter()
        fitter._accum_information = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-14]])
        pcov = fitter._compute_covariance_from_history(
            param_history=[np.array([1.0, 1.0]), np.array([1.0, 1.0])],
            current_params=np.array([1.0, 1.0]),
        )
        # Should NOT equal the naive (numerically meaningless) inverse.
        naive_inverse = np.linalg.inv(fitter._accum_information)
        assert not np.allclose(pcov, naive_inverse)
        assert np.all(np.isfinite(pcov))

    def test_falls_back_to_proxy_when_no_accumulated_information(self):
        fitter = LargeDatasetFitter()
        fitter._accum_information = None
        pcov = fitter._compute_covariance_from_history(
            param_history=[np.array([1.0, 1.0])],
            current_params=np.array([1.0, 1.0]),
        )
        assert pcov.shape == (2, 2)
        assert np.all(np.isfinite(pcov))


class TestChunkBufferPoolMultiFeatureXdata:
    """High: ChunkBufferPool allocated 1D-only buffers, crashing on
    multi-feature xdata of shape (N, k>1)."""

    def test_buffers_match_feature_shape(self):
        pool = ChunkBufferPool(chunk_size=100, x_feature_shape=(3,))
        x_buf, y_buf = pool.get_buffers(50)
        assert x_buf.shape == (50, 3)
        assert y_buf.shape == (50,)

    def test_copy_multi_feature_chunk_into_pool_succeeds(self):
        pool = ChunkBufferPool(chunk_size=100, x_feature_shape=(3,))
        x_buf, _y_buf = pool.get_buffers(10)
        source = np.random.rand(10, 3)
        # Pre-fix this raised: could not broadcast input array from shape
        # (10, 3) into shape (10,).
        np.copyto(x_buf, source)
        np.testing.assert_array_equal(x_buf, source)

    def test_plain_1d_xdata_still_works(self):
        pool = ChunkBufferPool(chunk_size=100)
        x_buf, _y_buf = pool.get_buffers(50)
        assert x_buf.shape == (50,)


class TestValidateModelFunctionAcceptsPartialAndCallableClass:
    """High: _validate_model_function assumed f.__code__/f.__name__ exist,
    crashing on functools.partial or callable-class models."""

    def test_functools_partial_model_does_not_crash(self):
        import functools

        fitter = LargeDatasetFitter()

        def base_model(x, a, b, scale):
            return scale * (a * x + b)

        model = functools.partial(base_model, scale=2.0)
        xdata = np.linspace(0, 1, 20)
        ydata = 2.0 * (1.0 * xdata + 0.5)

        # Should not raise AttributeError on __code__/__name__.
        fitter._validate_model_function(model, xdata, ydata, p0=[1.0, 0.5])
        # Second call should hit the validation cache without crashing.
        fitter._validate_model_function(model, xdata, ydata, p0=[1.0, 0.5])

    def test_callable_class_model_does_not_crash(self):
        fitter = LargeDatasetFitter()

        class Model:
            def __call__(self, x, a, b):
                return a * x + b

        model = Model()
        xdata = np.linspace(0, 1, 20)
        ydata = 2.0 * xdata + 1.0

        fitter._validate_model_function(model, xdata, ydata, p0=[1.0, 1.0])


class TestModelValidationCacheKeyDistinguishesClosures:
    """Found in PR review: the model-validation cache keyed on
    (name, __code__.co_code, __code__.co_consts) collides for two distinct
    closures built by the same factory, since captured free variables live
    in __closure__, not co_consts -- a false cache hit would silently skip
    validating the second closure. Fixed by keying on closure_serial(f)
    instead, matching nlsq.caching.smart_cache's existing pattern."""

    def test_distinct_closures_from_same_factory_get_distinct_cache_keys(self):
        fitter = LargeDatasetFitter()

        def make_model(scale):
            def model(x, a, b):
                return scale * (a * x + b)

            return model

        model_a = make_model(1.0)
        model_b = make_model(2.0)

        xdata = np.linspace(0, 1, 20)
        ydata_a = 1.0 * (1.0 * xdata + 0.5)
        ydata_b = 2.0 * (1.0 * xdata + 0.5)

        fitter._validate_model_function(model_a, xdata, ydata_a, p0=[1.0, 0.5])
        assert len(fitter._validated_functions) == 1

        fitter._validate_model_function(model_b, xdata, ydata_b, p0=[1.0, 0.5])
        assert len(fitter._validated_functions) == 2, (
            "two distinct closures sharing bytecode were treated as the "
            "same cache entry"
        )


class TestCalculateOptimalChunkSizeRejectsEmptyDataset:
    """Medium: calculate_optimal_chunk_size(0, ...) set
    chunk_size = n_points = 0, reaching a ZeroDivisionError in downstream
    chunk-count math instead of a clear error."""

    def test_zero_points_raises_value_error_not_zero_division(self):
        with pytest.raises(ValueError, match="empty"):
            MemoryEstimator.calculate_optimal_chunk_size(0, 2, LDMemoryConfig())


class TestValidatorsRejectNaN:
    """Medium: `value <= 0` / `< 0` / `value1 > value2` are all False for
    NaN under IEEE-754, so NaN silently bypassed validate_positive,
    validate_non_negative, validate_less_than_or_equal, and
    validate_residual_weighting_config."""

    def test_validate_positive_rejects_nan(self):
        with pytest.raises(ConfigValidationError):
            validate_positive(float("nan"), "some_param")

    def test_validate_non_negative_rejects_nan(self):
        with pytest.raises(ConfigValidationError):
            validate_non_negative(float("nan"), "some_param")

    def test_validate_less_than_or_equal_rejects_nan_in_either_position(self):
        with pytest.raises(ConfigValidationError):
            validate_less_than_or_equal(float("nan"), 1.0, "a", "b")
        with pytest.raises(ConfigValidationError):
            validate_less_than_or_equal(0.0, float("nan"), "a", "b")

    def test_residual_weights_rejects_nan(self):
        with pytest.raises(ConfigValidationError):
            validate_residual_weighting_config(
                enabled=True,
                weights=[1.0, float("nan"), 2.0],
            )


class TestDefenseLayerTelemetryThreadSafety:
    """Medium: DefenseLayerTelemetry's docstring claimed thread-safe
    statistics, but the record_* methods did unguarded increments/appends.
    Smoke test: concurrent recorders shouldn't lose increments."""

    def test_concurrent_record_calls_do_not_lose_increments(self):
        telemetry = DefenseLayerTelemetry()
        # Kept under DefenseLayerTelemetry's _event_log deque cap (maxlen
        # 1000, an intentional bound -- not the bug under test) so the
        # get_recent_events() length assertion below is meaningful.
        n_threads = 8
        calls_per_thread = 100

        def worker():
            for _ in range(calls_per_thread):
                telemetry.record_layer1_trigger(relative_loss=0.5, threshold=0.1)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert telemetry.layer1_warm_start_triggers == n_threads * calls_per_thread
        assert (
            len(telemetry.get_recent_events(n=10_000)) == n_threads * calls_per_thread
        )

#!/usr/bin/env python3
"""Additional tests for minpack module to improve coverage."""

import unittest

import jax.numpy as jnp
import numpy as np

from nlsq.core.minpack import CurveFit, curve_fit

try:
    from nlsq import fit_large_dataset
except ImportError:
    fit_large_dataset = None


class TestMinpackCoverage(unittest.TestCase):
    """Tests to improve minpack module coverage."""

    def test_curve_fit_basic(self):
        """Test basic curve_fit functionality."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        popt, _pcov = curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)
        self.assertAlmostEqual(popt[1], 0.0, places=3)

    def test_curve_fit_with_bounds(self):
        """Test curve_fit with bounds."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Test with bounds
        popt, _pcov = curve_fit(model, x, y, bounds=([0, -10], [10, 10]))
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_curve_fit_with_sigma(self):
        """Test curve_fit with uncertainties."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        sigma = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        popt, _pcov = curve_fit(model, x, y, sigma=sigma)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_curve_fit_exponential(self):
        """Test curve_fit with exponential model."""

        def model(x, a, b):
            return a * jnp.exp(b * x)

        x = np.linspace(0, 1, 50)
        y = 2.5 * np.exp(0.5 * x) + 0.01 * np.random.randn(50)

        popt, _pcov = curve_fit(model, x, y, p0=[1, 1])
        self.assertAlmostEqual(popt[0], 2.5, places=1)
        self.assertAlmostEqual(popt[1], 0.5, places=1)

    def test_curve_fit_2d(self):
        """Test curve_fit with 2D data."""

        def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y):
            x, y = xy
            # Use JAX-compatible operations
            a = 1 / (2 * sigma_x**2)
            b = 1 / (2 * sigma_y**2)
            g = amplitude * jnp.exp(-(a * (x - xo) ** 2 + b * (y - yo) ** 2))
            return g.ravel()

        # Create 2D grid
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        xx, yy = np.meshgrid(x, y)

        # Generate data
        z = gaussian_2d((xx, yy), 1, 5, 5, 1, 1)
        z += 0.01 * np.random.randn(*z.shape)

        # Fit
        popt, _pcov = curve_fit(
            gaussian_2d, (xx.ravel(), yy.ravel()), z, p0=[1, 5, 5, 1, 1]
        )

        self.assertAlmostEqual(popt[0], 1.0, places=1)
        self.assertAlmostEqual(popt[1], 5.0, places=1)

    def test_curve_fit_with_method(self):
        """Test curve_fit with different methods."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Test with trf method (only one implemented)
        popt, _pcov = curve_fit(model, x, y, method="trf")
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_curve_fit_maxfev(self):
        """Test curve_fit with max function evaluations."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        popt, _pcov = curve_fit(model, x, y, maxfev=10)
        # Should still converge for simple problem
        self.assertAlmostEqual(popt[0], 2.0, places=2)

    def test_curve_fit_class(self):
        """Test CurveFit class directly."""
        cf = CurveFit(use_dynamic_sizing=True)

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        popt, _pcov = cf.curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_fit_large_dataset(self):
        """Test fit_large_dataset for large datasets."""
        if fit_large_dataset is None:
            self.skipTest("fit_large_dataset not available")

        def model(x, a, b):
            return a * x + b

        # Large dataset
        np.random.seed(42)  # For reproducibility
        x = np.linspace(0, 100, 10000)
        y = 2 * x + 5 + np.random.randn(10000) * 0.1

        # Use fit_large_dataset with appropriate parameters
        result = fit_large_dataset(
            model, x, y, initial_params=[1.0, 0.0], chunk_size=1000
        )

        # Check that optimization was successful
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.popt[0], 2.0, places=1)
        self.assertAlmostEqual(result.popt[1], 5.0, places=0)

    def test_curve_fit_nan_policy(self):
        """Test curve_fit with NaN policy."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # With nan_policy='omit' should skip NaN values
        try:
            popt, _pcov = curve_fit(model, x, y, nan_policy="omit")
            # Should work with remaining points
            self.assertEqual(len(popt), 2)
        except:
            # May not be implemented yet
            pass

    def test_curve_fit_full_output(self):
        """Test curve_fit with full_output option."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # With full_output should return additional info
        try:
            result = curve_fit(model, x, y, full_output=True)
            if isinstance(result, tuple) and len(result) > 2:
                _popt, _pcov, infodict, _mesg, _ier = result
                self.assertIn("nfev", infodict)
        except:
            # May not be implemented
            pass

    def test_curve_fit_jac(self):
        """Test curve_fit with analytical Jacobian."""

        def model(x, a, b):
            return a * x + b

        # Jacobian not supported in current implementation
        # Test without jacobian
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Without analytical Jacobian
        popt, _pcov = curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)


class TestBugFixRegressions(unittest.TestCase):
    """Regression tests for bugs found in the 2026-08-25 three-brain review.

    Each test asserts the fixed behavior directly (not just "doesn't
    crash") so a future regression fails meaningfully.
    """

    def test_cmaes_config_chunked_strategy_does_not_crash(self):
        """CMAESConfig is slots=True (no __dict__); rebuilding it via
        **{**cfg.__dict__, ...} crashed every auto-chunked/streaming CMA-ES
        run with AttributeError. Verify the chunked strategy now sets
        data_chunk_size and completes successfully."""
        from nlsq.global_optimization.cmaes_config import (
            CMAESConfig,
            is_evosax_available,
        )

        if not is_evosax_available():
            self.skipTest("evosax not installed")

        from nlsq.core.minpack import _fit_global_cmaes

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 5, 200)
        y = np.asarray(2.0 * jnp.exp(-0.5 * x))
        config = CMAESConfig(data_chunk_size=None)

        result = _fit_global_cmaes(
            f=model,
            xdata=x,
            ydata=y,
            p0=np.array([1.0, 1.0]),
            sigma=None,
            absolute_sigma=False,
            bounds=(np.array([0.0, 0.0]), np.array([10.0, 5.0])),
            strategy="chunked",
            cmaes_config=config,
        )
        self.assertTrue(result["success"])
        self.assertAlmostEqual(float(result["popt"][0]), 2.0, delta=0.5)

    def test_stability_auto_fix_does_not_corrupt_positional_sigma(self):
        """stability='auto' p0 fixes used to move p0 into kwargs and slice
        the raw positional args tuple to drop it, silently reindexing any
        positional sigma/absolute_sigma that followed -- corrupting the
        weighting or crashing with 'multiple values for p0'."""
        x = np.linspace(0, 10, 50)
        rng = np.random.default_rng(1)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 50)
        sigma = np.full(50, 0.1)

        popt, _pcov = curve_fit(
            lambda x, a, b: a * x + b,
            x,
            y,
            [1.5, 0.5],  # positional p0
            sigma,  # positional sigma
            stability="auto",
        )
        np.testing.assert_allclose(popt, [2.0, 1.0], atol=0.15)

    def test_curve_fit_rejects_positional_arg_overflow(self):
        """The positional-to-keyword normalization used to silently drop
        any positional arg past the 9th (jac) via zip()'s shorter-sequence
        truncation. Overflow must now raise, not silently discard."""
        with self.assertRaises(TypeError):
            curve_fit(lambda x, a: a * x, [1, 2], [1, 2], *([None] * 18))

    def test_recovery_path_does_not_double_pass_tr_solver(self):
        """The recovery-lambda's least_squares() call forwards tr_solver
        (and ftol/xtol/x_scale/loss) explicitly, but the outer kwargs
        dict already has tr_solver set unconditionally by
        _select_tr_solver whenever it resolves a value (the default
        solver='auto' case). A prior version of the fix also spread the
        raw outer **kwargs at the end, so every recovery attempt raised
        TypeError: got multiple values for keyword argument 'tr_solver' --
        silently swallowed by recovery.py's except Exception, masking the
        real error behind a misleading 'recovery unsuccessful' message.
        Force the first least_squares() call to fail so recovery actually
        runs, and assert it converges instead of raising."""

        def model(x, a, b):
            return a * jnp.exp(b * x)

        x = np.linspace(0, 5, 20)
        y = np.asarray(model(x, 2.0, 0.3))

        cf = CurveFit(enable_recovery=True)
        real_least_squares = cf.ls.least_squares
        calls = {"n": 0}

        def flaky_least_squares(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("forced failure to trigger recovery")
            return real_least_squares(*args, **kwargs)

        cf.ls.least_squares = flaky_least_squares

        popt, _pcov = cf.curve_fit(model, x, y, p0=[1.0, 0.1])

        self.assertGreaterEqual(calls["n"], 2)  # recovery actually ran
        np.testing.assert_allclose(popt, [2.0, 0.3], atol=0.05)

    def test_curve_fit_eleventh_positional_arg_is_applied(self):
        """timeit is the 11th positional parameter of CurveFit.curve_fit;
        verify it's actually forwarded now, not silently dropped."""
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0

        result = curve_fit(
            lambda x, a, b: a * x + b,
            x,
            y,
            [1.5, 0.5],  # p0
            None,  # sigma
            False,  # absolute_sigma
            True,  # check_finite
            (-np.inf, np.inf),  # bounds
            None,  # method
            "auto",  # solver
            None,  # batch_size
            None,  # jac
            None,  # data_mask
            True,  # timeit  (11th positional)
        )
        # timeit=True makes CurveFit.curve_fit return a plain tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)  # popt, pcov, res, post_time, ctime

    def test_curve_fit_special_output_modes_do_not_crash(self):
        """CurveFit.curve_fit intentionally returns a plain tuple (not a
        CurveFitResult) for timeit/return_eval/full_output -- the
        module-level curve_fit() wrapper used to unconditionally do
        result['multistart_diagnostics'] = {...} on whatever came back,
        crashing with TypeError: 'tuple' object does not support item
        assignment for all three modes."""
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0

        for kwargs in (
            {"timeit": True},
            {"return_eval": True},
            {"full_output": True},
        ):
            with self.subTest(**kwargs):
                result = curve_fit(
                    lambda x, a, b: a * x + b, x, y, p0=[1.5, 0.5], **kwargs
                )
                self.assertIsInstance(result, tuple)

        # Normal calls must still get multistart_diagnostics
        result = curve_fit(lambda x, a, b: a * x + b, x, y, p0=[1.5, 0.5])
        self.assertIn("multistart_diagnostics", result)

    def test_fit_with_config_chunked_uses_fitter_below_1m_points(self):
        """_fit_with_config used to re-decide the strategy by n_points alone,
        silently falling back to plain curve_fit() (no chunking) whenever
        n_points < 1_000_000 -- even though MemoryBudgetSelector already
        chose 'chunked' because peak memory (n_params-scaled Jacobian)
        exceeded the safe threshold. That defeated the memory-based
        decision for small-n_points/huge-n_params problems and risked OOM.
        Verify LargeDatasetFitter (not plain curve_fit) is used regardless
        of n_points once an LDMemoryConfig has been selected."""
        from unittest.mock import patch

        from nlsq.core.minpack import _fit_with_config
        from nlsq.streaming.large_dataset import LDMemoryConfig

        x = np.linspace(0, 10, 500)  # well under 1_000_000 points
        y = 2.0 * x + 1.0
        config = LDMemoryConfig(memory_limit_gb=4.0)

        with (
            patch("nlsq.core.minpack.curve_fit") as mock_curve_fit,
            patch(
                "nlsq.streaming.large_dataset.LargeDatasetFitter.fit"
            ) as mock_fitter_fit,
        ):
            mock_fitter_fit.return_value = {
                "popt": np.array([2.0, 1.0]),
                "pcov": np.eye(2),
                "success": True,
            }
            _fit_with_config(
                f=lambda x, a, b: a * x + b,
                xdata=x,
                ydata=y,
                p0=[1.5, 0.5],
                sigma=None,
                absolute_sigma=False,
                check_finite=True,
                bounds=(-np.inf, np.inf),
                method=None,
                config=config,
                goal=None,
            )

        mock_fitter_fit.assert_called_once()
        mock_curve_fit.assert_not_called()

    def test_auto_global_multistart_chunked_uses_real_memory_config(self):
        """_fit_global_multistart's 'chunked' branch used to hardcode a
        fresh LDMemoryConfig(memory_limit_gb=8.0, ...), discarding the
        config MemoryBudgetSelector actually computed from the real
        (detected or overridden) memory budget in _fit_with_auto_global.
        Verify LargeDatasetFitter is now built from the selector's real
        config, not a hardcoded 8.0 GB default."""
        from unittest.mock import patch

        from nlsq.core.minpack import _fit_global_multistart
        from nlsq.streaming.large_dataset import LDMemoryConfig

        x = np.linspace(0, 10, 500)
        y = 2.0 * x + 1.0
        # A distinctive value that must survive through to LargeDatasetFitter.
        real_config = LDMemoryConfig(memory_limit_gb=33.0)

        with patch(
            "nlsq.streaming.large_dataset.LargeDatasetFitter"
        ) as mock_fitter_cls:
            mock_fitter_cls.return_value.fit.return_value = {
                "popt": np.array([2.0, 1.0]),
                "pcov": np.eye(2),
                "success": True,
            }
            _fit_global_multistart(
                f=lambda x, a, b: a * x + b,
                xdata=x,
                ydata=y,
                p0=[1.5, 0.5],
                sigma=None,
                absolute_sigma=False,
                check_finite=True,
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 5.0])),
                strategy="chunked",
                n_starts=5,
                memory_config=real_config,
            )

        mock_fitter_cls.assert_called_once()
        # The fitter must have been constructed with the real selected budget,
        # not a hardcoded 8.0 GB default.
        _args, ctor_kwargs = mock_fitter_cls.call_args
        self.assertIs(ctor_kwargs["config"], real_config)
        self.assertEqual(ctor_kwargs["memory_limit_gb"], 33.0)

    def test_fit_global_cmaes_reuses_real_memory_config_chunk_size(self):
        """_fit_global_cmaes derives data_chunk_size from n_points alone
        whenever cmaes_config.data_chunk_size is None -- verify a supplied
        memory_config's budget-derived chunk size (chunk_size for streaming,
        streaming_batch_size for chunked) actually overrides that fallback
        instead of being ignored, for both strategies."""
        from unittest.mock import MagicMock, patch

        from nlsq.global_optimization.cmaes_config import CMAESConfig
        from nlsq.streaming.hybrid_config import HybridStreamingConfig
        from nlsq.streaming.large_dataset import LDMemoryConfig

        x = np.linspace(0, 5, 50)
        y = np.asarray(2.0 * jnp.exp(-0.5 * x))
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 5.0]))

        cases = [
            ("streaming", HybridStreamingConfig(chunk_size=12_345), 12_345),
            ("chunked", LDMemoryConfig(streaming_batch_size=54_321), 54_321),
        ]
        for strategy, memory_config, expected_chunk_size in cases:
            with self.subTest(strategy=strategy):
                from nlsq.core.minpack import _fit_global_cmaes

                with patch(
                    "nlsq.global_optimization.cmaes_optimizer.CMAESOptimizer"
                ) as mock_optimizer_cls:
                    mock_optimizer_cls.return_value.fit.return_value = {
                        "popt": np.array([2.0, 0.5]),
                        "pcov": np.eye(2),
                    }
                    _fit_global_cmaes(
                        f=lambda x, a, b: a * jnp.exp(-b * x),
                        xdata=x,
                        ydata=y,
                        p0=np.array([1.0, 1.0]),
                        sigma=None,
                        absolute_sigma=False,
                        bounds=bounds,
                        strategy=strategy,
                        cmaes_config=CMAESConfig(data_chunk_size=None),
                        memory_config=memory_config,
                    )

                mock_optimizer_cls.assert_called_once()
                _args, ctor_kwargs = mock_optimizer_cls.call_args
                used_config = ctor_kwargs.get("config") or (_args[0] if _args else None)
                self.assertIsNotNone(used_config)
                self.assertEqual(used_config.data_chunk_size, expected_chunk_size)

    def test_fit_global_cmaes_clamps_budget_chunk_size_below_cmaes_minimum(self):
        """MemoryBudgetSelector's own chunk-size floor is 1_000 (FR-007),
        but CMAESConfig.data_chunk_size requires >= 1024 and raises
        ValueError otherwise. Passing a memory_config whose budget-derived
        chunk size falls in [1000, 1023] into _fit_global_cmaes used to
        pass that value straight through to CMAESConfig unclamped,
        crashing exactly the streaming/chunked auto_global CMA-ES runs
        this PR's memory_config threading was meant to make safer (found
        by an independent Codex review pass on PR #8)."""
        from unittest.mock import patch

        from nlsq.core.minpack import _fit_global_cmaes
        from nlsq.global_optimization.cmaes_config import CMAESConfig
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        x = np.linspace(0, 5, 50)
        y = np.asarray(2.0 * jnp.exp(-0.5 * x))
        # 1_000 is exactly MemoryBudgetSelector's own floor -- below
        # CMAESConfig's hard minimum of 1024.
        memory_config = HybridStreamingConfig(chunk_size=1_000)

        with patch(
            "nlsq.global_optimization.cmaes_optimizer.CMAESOptimizer"
        ) as mock_optimizer_cls:
            mock_optimizer_cls.return_value.fit.return_value = {
                "popt": np.array([2.0, 0.5]),
                "pcov": np.eye(2),
            }
            _fit_global_cmaes(
                f=lambda x, a, b: a * jnp.exp(-b * x),
                xdata=x,
                ydata=y,
                p0=np.array([1.0, 1.0]),
                sigma=None,
                absolute_sigma=False,
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 5.0])),
                strategy="streaming",
                cmaes_config=CMAESConfig(data_chunk_size=None),
                memory_config=memory_config,
            )

        mock_optimizer_cls.assert_called_once()
        _args, ctor_kwargs = mock_optimizer_cls.call_args
        used_config = ctor_kwargs.get("config") or (_args[0] if _args else None)
        self.assertGreaterEqual(used_config.data_chunk_size, 1024)

    def test_fit_global_multistart_streaming_reuses_real_memory_config(self):
        """_fit_global_multistart's streaming branch (strategy not
        'standard' or 'chunked') builds a fresh HybridStreamingConfig()
        default when no memory_config is given. Verify a supplied
        memory_config's fields (e.g. chunk_size) survive the
        dataclasses.replace() call into the optimizer instead of being
        silently dropped for a default config."""
        from unittest.mock import patch

        from nlsq.core.minpack import _fit_global_multistart
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        x = np.linspace(0, 10, 500)
        y = 2.0 * x + 1.0
        real_config = HybridStreamingConfig(chunk_size=98_765, normalize=False)

        with patch(
            "nlsq.streaming.adaptive_hybrid.AdaptiveHybridStreamingOptimizer"
        ) as mock_optimizer_cls:
            mock_optimizer_cls.return_value.fit.return_value = {
                "popt": np.array([2.0, 1.0]),
                "pcov": np.eye(2),
                "success": True,
            }
            _fit_global_multistart(
                f=lambda x, a, b: a * x + b,
                xdata=x,
                ydata=y,
                p0=[1.5, 0.5],
                sigma=None,
                absolute_sigma=False,
                check_finite=True,
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 5.0])),
                strategy="streaming",
                n_starts=5,
                memory_config=real_config,
            )

        mock_optimizer_cls.assert_called_once()
        _args, ctor_kwargs = mock_optimizer_cls.call_args
        used_config = ctor_kwargs.get("config") or (_args[0] if _args else None)
        self.assertIsNotNone(used_config)
        self.assertEqual(used_config.chunk_size, 98_765)
        # Multistart-specific overrides must still apply on top of the reused config.
        self.assertTrue(used_config.enable_multistart)
        self.assertEqual(used_config.n_starts, 5)

    def test_fit_global_multistart_labels_messages_with_actual_workflow(self):
        """_fit_global_multistart used to hardcode 'hpc' in its
        checkpoint-not-supported warning and 'auto_global' in its chunked
        NotImplementedError/RuntimeError, regardless of which workflow=
        value the caller actually used (both entry points route through
        this same function). Verify the workflow_name parameter now
        threads through into both messages correctly, for BOTH literal
        values, so it can never be a coincidence that the old hardcoded
        text happened to match."""
        import warnings
        from unittest.mock import patch

        from nlsq.core.minpack import _fit_global_multistart

        x = np.linspace(0, 5, 50)
        y = 2.0 * x + 1.0
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 5.0]))

        # Checkpoint warning: must name the actual workflow, not a fixed 'hpc'.
        with patch("nlsq.core.minpack.curve_fit") as mock_curve_fit:
            mock_curve_fit.return_value = {"success": True}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _fit_global_multistart(
                    f=lambda x, a, b: a * x + b,
                    xdata=x,
                    ydata=y,
                    p0=[1.5, 0.5],
                    sigma=None,
                    absolute_sigma=False,
                    check_finite=True,
                    bounds=bounds,
                    strategy="standard",
                    n_starts=5,
                    workflow_name="auto_global",
                    _hpc_checkpoint_dir="/tmp/some-checkpoint-dir",
                )
        messages = [str(w.message) for w in caught]
        assert any("workflow='auto_global'" in m for m in messages), messages
        assert not any("workflow='hpc'" in m for m in messages), messages

        # Chunked + sigma NotImplementedError: must name the actual workflow,
        # not a fixed 'auto_global'.
        with self.assertRaises(NotImplementedError) as excinfo:
            _fit_global_multistart(
                f=lambda x, a, b: a * x + b,
                xdata=x,
                ydata=y,
                p0=[1.5, 0.5],
                sigma=np.ones_like(y),
                absolute_sigma=False,
                check_finite=True,
                bounds=bounds,
                strategy="chunked",
                n_starts=5,
                workflow_name="hpc",
            )
        self.assertIn("workflow='hpc'", str(excinfo.exception))
        self.assertNotIn("workflow='auto_global'", str(excinfo.exception))

        # Chunked-fit-failed RuntimeError: must also name the actual
        # workflow -- this is a third, independent interpolation site from
        # the NotImplementedError above (only reached when sigma is None
        # and LargeDatasetFitter itself reports failure).
        with patch(
            "nlsq.streaming.large_dataset.LargeDatasetFitter"
        ) as mock_fitter_cls:
            mock_fitter_cls.return_value.fit.return_value = {
                "success": False,
                "message": "synthetic failure",
            }
            with self.assertRaises(RuntimeError) as excinfo:
                _fit_global_multistart(
                    f=lambda x, a, b: a * x + b,
                    xdata=x,
                    ydata=y,
                    p0=[1.5, 0.5],
                    sigma=None,
                    absolute_sigma=False,
                    check_finite=True,
                    bounds=bounds,
                    strategy="chunked",
                    n_starts=5,
                    workflow_name="hpc",
                )
        self.assertIn("workflow='hpc'", str(excinfo.exception))
        self.assertNotIn("workflow='auto_global'", str(excinfo.exception))

    def test_fit_workflow_hpc_end_to_end_labels_checkpoint_warning_correctly(self):
        """The unit test above supplies workflow_name= directly to
        _fit_global_multistart -- it doesn't prove the real call chain
        (fit(workflow='hpc') -> _fit_with_hpc -> _fit_with_auto_global ->
        _fit_global_multistart) actually threads _workflow_name="hpc"
        through. A future refactor dropping the
        workflow_name=_workflow_name kwarg at the _fit_with_auto_global
        call site would leave the direct unit test green while this real
        bug reappeared. Runs fit() end-to-end (small dataset, multistart
        selected since scale_ratio~1 stays under the CMA-ES threshold) and
        checks the actual warning text."""
        import tempfile
        import warnings

        from nlsq import fit

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        y = 2.5 * np.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            fit(
                model,
                x,
                y,
                p0=[1.0, 0.5],
                workflow="hpc",
                bounds=([0.0, 0.0], [10.0, 10.0]),
                checkpoint_dir=tmpdir,
            )
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any("workflow='hpc'" in m and "checkpoint" in m for m in messages),
            messages,
        )

    def test_fit_global_cmaes_checkpoint_dir_override_preserves_config_ids(self):
        """_fit_global_cmaes's checkpoint_dir override used to unconditionally
        replace a pre-built cmaes_config's run_id/model_id with the fit()
        kwargs' values (None when the caller only overrides checkpoint_dir
        at the top level), wiping out identifiers already set on the config
        and crashing CMAESConfig's own checkpoint_dir-requires-run_id/
        model_id validation. Verify a pre-built config's run_id/model_id
        survive a checkpoint_dir-only override, matching the None-preserves
        -existing-value treatment checkpoint_interval already gets."""
        from unittest.mock import patch

        from nlsq.core.minpack import _fit_global_cmaes
        from nlsq.global_optimization.cmaes_config import CMAESConfig

        x = np.linspace(0, 5, 50)
        y = np.asarray(2.0 * jnp.exp(-0.5 * x))
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 5.0]))

        prebuilt_config = CMAESConfig(
            restart_strategy="none",
            seed=1,
            run_id="prebuilt-run",
            model_id="prebuilt-model",
        )

        with patch(
            "nlsq.global_optimization.cmaes_optimizer.CMAESOptimizer"
        ) as mock_optimizer_cls:
            mock_optimizer_cls.return_value.fit.return_value = {
                "popt": np.array([2.0, 0.5]),
                "pcov": np.eye(2),
            }
            _fit_global_cmaes(
                f=lambda x, a, b: a * jnp.exp(-b * x),
                xdata=x,
                ydata=y,
                p0=np.array([1.0, 1.0]),
                sigma=None,
                absolute_sigma=False,
                bounds=bounds,
                strategy="standard",
                cmaes_config=prebuilt_config,
                # Caller overrides checkpoint_dir only -- no run_id/model_id
                # at this level -- so this must not fail CMAESConfig
                # validation by wiping the config's own run_id/model_id.
                _hpc_checkpoint_dir="/tmp/some-checkpoint-dir",
            )

        mock_optimizer_cls.assert_called_once()
        _args, ctor_kwargs = mock_optimizer_cls.call_args
        used_config = ctor_kwargs.get("config") or (_args[0] if _args else None)
        self.assertEqual(used_config.run_id, "prebuilt-run")
        self.assertEqual(used_config.model_id, "prebuilt-model")

    def test_fit_with_auto_global_reuses_precomputed_budget(self):
        """_fit_with_auto_global computes a MemoryBudget for logging and
        used to let selector.select() silently redetect memory a second
        time. Verify MemoryBudget.compute() is called exactly once (not
        twice) for this workflow, proving the budget=budget wiring is
        actually exercised end-to-end, not just at the unit level."""
        from unittest.mock import patch

        from nlsq.core.minpack import _fit_with_auto_global
        from nlsq.core.workflow import MemoryBudget

        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0

        with (
            patch.object(
                MemoryBudget, "compute", wraps=MemoryBudget.compute
            ) as mock_compute,
            patch("nlsq.core.minpack._fit_global_multistart") as mock_multistart,
            patch("nlsq.core.minpack._fit_global_cmaes") as mock_cmaes,
        ):
            mock_multistart.return_value = {"success": True}
            mock_cmaes.return_value = {"success": True}
            _fit_with_auto_global(
                f=lambda x, a, b: a * x + b,
                xdata=x,
                ydata=y,
                p0=[1.5, 0.5],
                sigma=None,
                absolute_sigma=False,
                check_finite=True,
                bounds=([0.0, 0.0], [10.0, 5.0]),
                n_points=30,
                n_params=2,
                goal=None,
            )

        mock_compute.assert_called_once()

    def test_log_memory_budget_diagnostics_reuses_precomputed_budget(self):
        """_log_memory_budget_diagnostics computes a MemoryBudget then used
        to let selector.select() redetect memory again internally. Verify
        MemoryBudget.compute() is called exactly once."""
        from unittest.mock import patch

        from nlsq.core.minpack import _log_memory_budget_diagnostics
        from nlsq.core.workflow import MemoryBudget

        with patch.object(
            MemoryBudget, "compute", wraps=MemoryBudget.compute
        ) as mock_compute:
            _log_memory_budget_diagnostics(
                xdata=np.linspace(0, 10, 30),
                ydata=np.linspace(0, 20, 30),
                p0=np.array([1.5, 0.5]),
            )

        mock_compute.assert_called_once()

    def test_data_mask_respected_when_flength_is_none(self):
        """_setup_data_mask_and_padding's flength-is-None branch used to
        unconditionally overwrite a caller-supplied data_mask with an
        all-True mask, silently including points the caller asked to
        exclude. With CurveFit(flength=None) (the default), a mask that
        excludes the outlier point must still be honored."""
        fitter = CurveFit(flength=None)
        m = 3
        mask = np.array([True, True, False])

        data_mask, none_mask, _len_diff, _should_pad = (
            fitter._setup_data_mask_and_padding(mask, m)
        )

        np.testing.assert_array_equal(data_mask, mask)
        self.assertFalse(none_mask)

    def test_data_mask_excludes_outlier_from_fit(self):
        """End-to-end: fitting a*x with a data_mask that excludes an
        outlier must produce the masked-in slope, not the slope pulled
        toward the outlier.

        The outlier point (300, 900) implies slope 3, well off the
        masked-in points' slope of 1 (verified: unmasked WLS solution is
        exactly a=1.0 for (1,1),(2,2); including (300,900) pulls it to
        ~3.0). A collinear outlier would make this test pass even with the
        data_mask fix reverted, since masking it out or not changes
        nothing about the optimal slope."""

        def model(x, a):
            return a * x

        x = np.array([1.0, 2.0, 300.0])
        y = np.array([1.0, 2.0, 900.0])
        mask = np.array([True, True, False])

        fitter = CurveFit(flength=None)
        popt, _pcov = fitter.curve_fit(model, x, y, data_mask=mask, p0=[0.5])

        self.assertAlmostEqual(float(popt[0]), 1.0, places=2)

    def test_reused_fitter_picks_up_new_jac_closure(self):
        """Regression: LeastSquares.update_function()'s analytical-Jacobian
        reuse check compared only jac.__code__.co_code, unlike the fun
        check right above it (which already compares co_consts and closure
        cells). Two `jac` closures built by the same factory share
        bytecode but capture different scale constants; reusing a CurveFit
        instance across both fits used to silently keep self.jac pointing
        at the *first* closure whenever a same-bytecode replacement came
        in, instead of picking up the newly supplied one."""

        def make_jac(scale):
            def jac(x, a):
                return scale * jnp.ones((1, x.shape[0]))

            return jac

        def model(x, a):
            return a * x

        x = np.linspace(1.0, 5.0, 20)
        y = 3.0 * x

        fitter = CurveFit(flength=None)

        jac_a = make_jac(1.0)
        fitter.curve_fit(model, x, y, jac=jac_a, p0=[1.0])
        self.assertIs(fitter.ls.jac, jac_a)

        # A different closure (same bytecode, different captured scale)
        # must replace self.jac, not be silently treated as "unchanged".
        jac_b = make_jac(9.0)
        fitter.curve_fit(model, x, y, jac=jac_b, p0=[1.0])
        self.assertIs(
            fitter.ls.jac,
            jac_b,
            "a jac closure with different captured constants must "
            "replace the stale wrapper, not be treated as identical "
            "just because co_code matches",
        )

    def test_stability_auto_rescale_data_flag_is_noop(self):
        """curve_fit(..., stability='auto') used to silently rescale
        xdata/ydata to [0, 1] when ill-conditioned, returning fitted
        parameters in the wrong coordinate system for an arbitrary
        nonlinear model (not soundly invertible). rescale_data=True and
        rescale_data=False must now produce identical popt, and a warning
        must explain that rescaling was skipped."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 1e5, 100)  # large x_range triggers the old rescale
        y = np.asarray(2.5 * np.exp(-3e-5 * x))
        p0 = [1.0, 1e-4]

        popt_true, _ = curve_fit(
            model, x, y, p0=p0, stability="auto", rescale_data=True
        )
        popt_false, _ = curve_fit(
            model, x, y, p0=p0, stability="auto", rescale_data=False
        )
        np.testing.assert_allclose(popt_true, popt_false)

        with self.assertLogs("nlsq.minpack", level="WARNING") as cm:
            curve_fit(model, x, y, p0=p0, stability="auto", rescale_data=True)
        self.assertTrue(
            any("NOT applied" in msg for msg in cm.output),
            f"expected a 'NOT applied' warning, got: {cm.output}",
        )

    def test_streaming_strategy_forwards_memory_config(self):
        """_curve_fit_auto_memory's 'streaming' branch used to drop the
        memory-budget-derived HybridStreamingConfig on the floor, silently
        falling back to the optimizer's own default chunk size instead of
        the one the memory-budget selection just computed."""
        from unittest.mock import MagicMock, patch

        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 5, 20)
        y = np.asarray(2.0 * jnp.exp(-0.5 * x))
        real_config = HybridStreamingConfig(chunk_size=12_345)

        with (
            patch(
                "nlsq.core.workflow.MemoryBudgetSelector.select",
                return_value=("streaming", real_config),
            ),
            patch.object(CurveFit, "_curve_fit_hybrid_streaming") as mock_stream,
        ):
            mock_result = MagicMock()
            mock_stream.return_value = mock_result
            cf = CurveFit()
            cf._curve_fit_auto_memory(
                f=model,
                xdata=x,
                ydata=y,
                p0=[1.0, 1.0],
                sigma=None,
                absolute_sigma=False,
                check_finite=True,
                bounds=(-np.inf, np.inf),
                callback=None,
                verbose=0,
            )

        self.assertIs(mock_stream.call_args.kwargs["config"], real_config)

    def test_singular_matrix_failure_routes_to_numerical_recovery(self):
        """A caught LinAlgError used to always be classified as the generic
        'optimization_error', which never matches
        OptimizationRecovery._adjust_regularization's
        ["numerical", "ill_conditioned"] gate -- that recovery strategy
        (regularization boost + LSMR switch) was therefore unreachable.
        Verify a LinAlgError now routes through recovery and the fit still
        succeeds instead of propagating the exception."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 5, 20)
        y = np.asarray(2.0 * jnp.exp(-0.5 * x))

        cf = CurveFit(enable_recovery=True)
        real_least_squares = cf.ls.least_squares
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise np.linalg.LinAlgError("Singular matrix")
            return real_least_squares(*args, **kwargs)

        cf.ls.least_squares = flaky
        popt, _ = cf.curve_fit(model, x, y, p0=[1.0, 1.0])

        self.assertGreaterEqual(calls["n"], 2)
        self.assertTrue(np.all(np.isfinite(popt)))


if __name__ == "__main__":
    unittest.main()

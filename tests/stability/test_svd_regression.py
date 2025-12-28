"""Regression tests for SVD accuracy and determinism in optimization.

These tests verify that:
1. No randomized/approximate SVD code paths remain in the codebase
2. SVD results are fully deterministic across runs
3. Large dataset optimization converges correctly with full SVD

Historical context:
  Prior to v0.3.5, compute_svd_adaptive used randomized SVD for large Jacobians
  (>500K elements). The approximation error accumulated across iterations,
  causing early termination at worse local minima.

  Evidence from homodyne XPCS fitting:
    - Full SVD (v0.3.0): D0 error 9.74%, alpha error 0.59%, 15 iterations
    - Randomized SVD (v0.3.1-v0.3.4): D0 error 30.18%, alpha error 14.66%, 6 iterations

  Resolution (v0.3.5): Randomized SVD completely removed from codebase.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import LeastSquares
from nlsq.stability.svd_fallback import compute_svd_adaptive, compute_svd_with_fallback


class TestNoRandomizedSVD:
    """Tests to verify randomized SVD has been completely removed."""

    def test_no_randomized_svd_function(self):
        """Verify randomized_svd function no longer exists."""
        from nlsq.stability import svd_fallback

        assert not hasattr(svd_fallback, "randomized_svd"), (
            "randomized_svd function should be removed from svd_fallback module"
        )

    def test_no_randomized_svd_threshold(self):
        """Verify RANDOMIZED_SVD_THRESHOLD constant no longer exists."""
        from nlsq.stability import svd_fallback

        assert not hasattr(svd_fallback, "RANDOMIZED_SVD_THRESHOLD"), (
            "RANDOMIZED_SVD_THRESHOLD should be removed from svd_fallback module"
        )

    def test_no_random_import(self):
        """Verify jax.random is not imported in svd_fallback."""
        import inspect

        from nlsq.stability import svd_fallback

        source = inspect.getsource(svd_fallback)
        assert "from jax import random" not in source, (
            "jax.random should not be imported in svd_fallback module"
        )
        assert "jax.random" not in source, (
            "jax.random should not be used in svd_fallback module"
        )

    def test_compute_svd_adaptive_warns_on_use_randomized_true(self):
        """Verify compute_svd_adaptive warns if use_randomized=True is passed."""
        m, n = 100, 10
        A = jnp.ones((m, n))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, _, _ = compute_svd_adaptive(A, use_randomized=True)

            # Should issue a deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Randomized SVD was removed" in str(w[0].message)

    def test_compute_svd_adaptive_uses_full_svd(self):
        """Verify compute_svd_adaptive always uses full deterministic SVD."""
        np.random.seed(42)
        m, n = 1000, 10
        A = jnp.array(np.random.randn(m, n))

        # Get results from compute_svd_adaptive
        U_adaptive, s_adaptive, V_adaptive = compute_svd_adaptive(A)

        # Get results from compute_svd_with_fallback (known full SVD)
        U_full, s_full, V_full = compute_svd_with_fallback(A)

        # They should be identical
        np.testing.assert_array_almost_equal(s_adaptive, s_full, decimal=14)
        np.testing.assert_array_almost_equal(
            np.abs(U_adaptive), np.abs(U_full), decimal=14
        )
        np.testing.assert_array_almost_equal(
            np.abs(V_adaptive), np.abs(V_full), decimal=14
        )


class TestSVDDeterminism:
    """Tests to verify SVD results are fully deterministic."""

    def test_svd_deterministic_across_runs(self):
        """Verify SVD produces identical results across multiple runs."""
        np.random.seed(42)
        m, n = 500, 20
        A = jnp.array(np.random.randn(m, n))

        # Run SVD multiple times
        results = []
        for _ in range(5):
            U, s, V = compute_svd_with_fallback(A)
            results.append((np.array(U), np.array(s), np.array(V)))

        # All results should be bitwise identical
        U0, s0, V0 = results[0]
        for i, (U, s, V) in enumerate(results[1:], 1):
            np.testing.assert_array_equal(
                s, s0, err_msg=f"Run {i}: singular values differ"
            )
            np.testing.assert_array_equal(
                U, U0, err_msg=f"Run {i}: left singular vectors differ"
            )
            np.testing.assert_array_equal(
                V, V0, err_msg=f"Run {i}: right singular vectors differ"
            )

    def test_svd_reconstruction_accuracy(self):
        """Verify full SVD achieves machine precision reconstruction."""
        np.random.seed(42)
        m, n = 1000, 15
        A = jnp.array(np.random.randn(m, n).astype(np.float64))

        U, s, V = compute_svd_with_fallback(A, full_matrices=False)

        # Reconstruct matrix
        A_reconstructed = U @ jnp.diag(s) @ V.T

        # Should achieve machine precision (< 1e-10 relative error)
        relative_error = jnp.linalg.norm(A - A_reconstructed) / jnp.linalg.norm(A)
        assert float(relative_error) < 1e-10, (
            f"Full SVD reconstruction error too high: {relative_error}"
        )

    def test_svd_singular_values_ordering(self):
        """Verify singular values are in descending order."""
        np.random.seed(42)
        m, n = 200, 30
        A = jnp.array(np.random.randn(m, n))

        _, s, _ = compute_svd_with_fallback(A)

        # Singular values should be in descending order
        s_np = np.array(s)
        assert np.all(np.diff(s_np) <= 0), "Singular values not in descending order"


class TestOptimizationConvergence:
    """Tests for optimization convergence with full deterministic SVD."""

    def test_large_dataset_convergence(self):
        """Test convergence for large dataset (>500K Jacobian elements).

        This test verifies that with full deterministic SVD, optimization
        converges to the correct solution. The old randomized SVD would
        converge to a worse local minimum with 3-6x worse errors.
        """
        # 50000 points * 13 params = 650,000 Jacobian elements
        n_points = 50000
        np.random.seed(42)

        n_phi = 5
        n_params = 2 * n_phi + 3
        q = 0.01

        # Ground truth
        ground_truth = {
            "D0": 100.0,
            "alpha": 0.5,
            "D_offset": 1.0,
            "contrast": 0.5,
            "offset": 1.0,
        }

        # Generate data
        n_per_phi = n_points // n_phi
        n_t = int(np.sqrt(n_per_phi))

        t1 = np.linspace(0.1, 1.0, n_t)
        t2 = np.linspace(0.1, 1.0, n_t)
        T1, T2 = np.meshgrid(t1, t2, indexing="ij")

        xdata_list = []
        ydata_list = []

        for i in range(n_phi):
            t1_flat = T1.ravel()
            t2_flat = T2.ravel()
            tau = np.sqrt(t1_flat * t2_flat)
            D_eff = (
                ground_truth["D0"] * np.power(tau + 0.01, ground_truth["alpha"] - 1.0)
                + ground_truth["D_offset"]
            )
            decay = np.exp(-D_eff * (t1_flat + t2_flat) * q * q)
            g2_theoretical = ground_truth["offset"] + ground_truth["contrast"] * decay
            sigma = np.abs(g2_theoretical) * 0.02 + 1e-6
            noise = np.random.normal(0, 1, len(t1_flat)) * sigma
            g2_noisy = g2_theoretical + noise
            xdata_list.append(
                np.column_stack(
                    [
                        t1_flat,
                        t2_flat,
                        np.full(len(t1_flat), i),
                        np.full(len(t1_flat), i),
                    ]
                )
            )
            ydata_list.append(g2_noisy)

        xdata = jnp.array(np.vstack(xdata_list))
        ydata = jnp.array(np.hstack(ydata_list))

        # Initial guess
        p0 = np.array([0.55] * n_phi + [1.05] * n_phi + [115.0, 0.45, 1.2])
        bounds = (
            np.array([0.1] * n_phi + [0.5] * n_phi + [10.0, 0.1, 0.1]),
            np.array([0.9] * n_phi + [1.5] * n_phi + [1000.0, 1.0, 10.0]),
        )

        def model_func(xdata_chunk, *params):
            params_arr = jnp.array(params)
            contrasts = params_arr[:n_phi]
            offsets = params_arr[n_phi : 2 * n_phi]
            D0 = params_arr[2 * n_phi]
            alpha = params_arr[2 * n_phi + 1]
            D_offset = params_arr[2 * n_phi + 2]
            t1 = xdata_chunk[:, 0]
            t2 = xdata_chunk[:, 1]
            phi_idx = xdata_chunk[:, 3].astype(jnp.int32)
            contrast = contrasts[phi_idx]
            offset = offsets[phi_idx]
            tau = jnp.sqrt(t1 * t2)
            D_eff = D0 * jnp.power(tau + 0.01, alpha - 1.0) + D_offset
            decay = jnp.exp(-D_eff * (t1 + t2) * q * q)
            return offset + contrast * decay

        # Run optimization
        ls = LeastSquares(enable_stability=True, enable_diagnostics=False)
        result = ls.least_squares(
            fun=model_func,
            x0=p0,
            xdata=xdata,
            ydata=ydata,
            bounds=bounds,
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=500,
            verbose=0,
        )

        popt = np.array(result["x"])
        D0_recovered = popt[2 * n_phi]
        alpha_recovered = popt[2 * n_phi + 1]

        # D0 should be recovered within 20% (allowing for noise)
        D0_error = abs(D0_recovered - ground_truth["D0"]) / ground_truth["D0"]
        assert D0_error < 0.20, f"D0 recovery error too high: {D0_error * 100:.1f}%"

        # Alpha should be recovered within 30% (more sensitive to noise)
        alpha_error = (
            abs(alpha_recovered - ground_truth["alpha"]) / ground_truth["alpha"]
        )
        assert alpha_error < 0.30, (
            f"alpha recovery error too high: {alpha_error * 100:.1f}%"
        )

        # Should converge (not hit max iterations)
        assert result.get("success", True), "Optimization should converge"

    def test_optimization_determinism(self):
        """Verify optimization produces identical results across runs."""
        np.random.seed(42)
        n_points = 1000
        n_phi = 3
        q = 0.01

        # Generate deterministic data
        t1 = np.linspace(0.1, 1.0, 10)
        t2 = np.linspace(0.1, 1.0, 10)
        T1, T2 = np.meshgrid(t1, t2, indexing="ij")

        xdata_list = []
        ydata_list = []

        for i in range(n_phi):
            t1_flat = T1.ravel()
            t2_flat = T2.ravel()
            tau = np.sqrt(t1_flat * t2_flat)
            D_eff = 100.0 * np.power(tau + 0.01, -0.5) + 1.0
            decay = np.exp(-D_eff * (t1_flat + t2_flat) * q * q)
            g2 = 1.0 + 0.5 * decay
            xdata_list.append(
                np.column_stack(
                    [
                        t1_flat,
                        t2_flat,
                        np.full(len(t1_flat), i),
                        np.full(len(t1_flat), i),
                    ]
                )
            )
            ydata_list.append(g2)

        xdata = jnp.array(np.vstack(xdata_list))
        ydata = jnp.array(np.hstack(ydata_list))

        p0 = np.array([0.55] * n_phi + [1.05] * n_phi + [115.0, 0.45, 1.2])
        bounds = (
            np.array([0.1] * n_phi + [0.5] * n_phi + [10.0, 0.1, 0.1]),
            np.array([0.9] * n_phi + [1.5] * n_phi + [1000.0, 1.0, 10.0]),
        )

        def model_func(xdata_chunk, *params):
            params_arr = jnp.array(params)
            contrasts = params_arr[:n_phi]
            offsets = params_arr[n_phi : 2 * n_phi]
            D0 = params_arr[2 * n_phi]
            alpha = params_arr[2 * n_phi + 1]
            D_offset = params_arr[2 * n_phi + 2]
            t1 = xdata_chunk[:, 0]
            t2 = xdata_chunk[:, 1]
            phi_idx = xdata_chunk[:, 3].astype(jnp.int32)
            contrast = contrasts[phi_idx]
            offset = offsets[phi_idx]
            tau = jnp.sqrt(t1 * t2)
            D_eff = D0 * jnp.power(tau + 0.01, alpha - 1.0) + D_offset
            decay = jnp.exp(-D_eff * (t1 + t2) * q * q)
            return offset + contrast * decay

        # Run optimization twice
        results = []
        for _ in range(2):
            ls = LeastSquares(enable_stability=True, enable_diagnostics=False)
            result = ls.least_squares(
                fun=model_func,
                x0=p0.copy(),
                xdata=xdata,
                ydata=ydata,
                bounds=bounds,
                method="trf",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=500,
                verbose=0,
            )
            results.append(np.array(result["x"]))

        # Results should be identical
        np.testing.assert_array_almost_equal(
            results[0],
            results[1],
            decimal=10,
            err_msg="Optimization results differ between runs - not deterministic",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

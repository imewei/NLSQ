"""
Tests for Global Optimization Sampling Module
==============================================

Tests for LHS, Sobol, Halton samplers and bounds transformation utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization.sampling import (
    center_samples_around_p0,
    get_sampler,
    halton_sample,
    latin_hypercube_sample,
    scale_samples_to_bounds,
    sobol_sample,
)


@pytest.mark.parametrize("sampler", [sobol_sample, halton_sample])
def test_quasi_random_sampler_rejects_negative_skip(sampler):
    with pytest.raises(ValueError, match="skip"):
        sampler(4, 1, skip=-2)


def test_scale_samples_handles_wide_finite_bounds():
    samples = jnp.array([[0.0], [0.5], [1.0]])
    scaled = scale_samples_to_bounds(samples, jnp.array([-1e308]), jnp.array([1e308]))
    np.testing.assert_allclose(scaled[:, 0], [-1e308, 0.0, 1e308])


def test_centered_samples_cover_wide_finite_bounds():
    samples = jnp.array([[0.0], [0.5], [1.0]])
    scaled = center_samples_around_p0(
        samples, jnp.array([0.0]), 1.0, jnp.array([-1e308]), jnp.array([1e308])
    )
    np.testing.assert_allclose(scaled[:, 0], [-1e308, 0.0, 1e308])


class TestLatinHypercubeSample:
    """Test Latin Hypercube Sampling implementation."""

    def test_lhs_correct_shape(self):
        """Test LHS generates correct shape (N, n_params) arrays."""
        n_samples = 20
        n_dims = 5

        samples = latin_hypercube_sample(n_samples, n_dims)

        assert samples.shape == (n_samples, n_dims)
        assert isinstance(samples, jnp.ndarray)

    def test_lhs_stratification_property(self):
        """Test LHS stratification property (one sample per stratum per dimension)."""
        n_samples = 10
        n_dims = 3
        rng_key = jax.random.PRNGKey(42)

        samples = latin_hypercube_sample(n_samples, n_dims, rng_key=rng_key)

        # For each dimension, samples should be stratified:
        # One sample in each of N equal strata [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
        for dim in range(n_dims):
            dim_samples = np.asarray(samples[:, dim])
            # Sort samples and check they fall in appropriate strata
            sorted_samples = np.sort(dim_samples)
            for i, sample in enumerate(sorted_samples):
                # Sample i should be in stratum [i/N, (i+1)/N)
                lower_bound = i / n_samples
                upper_bound = (i + 1) / n_samples
                assert lower_bound <= sample < upper_bound, (
                    f"Sample {sample} at index {i} should be in [{lower_bound}, {upper_bound})"
                )

    def test_lhs_draws_independent_keys_per_dimension(self, monkeypatch):
        """Each dimension must get its own (key_uniform, key_perm) pair.

        Regression test: the previous implementation split n_dims + 1 keys
        and reused keys[i] as key_perm for dimension i-1 AND key_uniform for
        dimension i, correlating adjacent dimensions. The fix draws
        2 * n_dims keys so no index is shared across dimensions.
        """
        n_samples = 8
        n_dims = 4
        rng_key = jax.random.PRNGKey(0)

        requested_counts = []
        real_split = jax.random.split

        def spy_split(key, num=2):
            requested_counts.append(num)
            return real_split(key, num)

        monkeypatch.setattr(jax.random, "split", spy_split)

        latin_hypercube_sample(n_samples, n_dims, rng_key=rng_key)

        # 2 keys per dimension (key_uniform + key_perm), none shared across
        # dimensions -- the buggy version requested only n_dims + 1 keys.
        assert requested_counts[0] == 2 * n_dims


class TestSobolSample:
    """Test Sobol sequence generation."""

    def test_sobol_correct_shape(self):
        """Test Sobol sequence generation with correct shape."""
        n_samples = 16
        n_dims = 4

        samples = sobol_sample(n_samples, n_dims)

        assert samples.shape == (n_samples, n_dims)
        assert isinstance(samples, jnp.ndarray)

    def test_sobol_quasi_random_properties(self):
        """Test Sobol sequence has correct quasi-random properties."""
        n_samples = 64
        n_dims = 2

        samples = sobol_sample(n_samples, n_dims)
        samples_np = np.asarray(samples)

        # All samples should be in [0, 1]
        assert np.all(samples_np >= 0.0)
        assert np.all(samples_np <= 1.0)

        # Sobol sequences should have good space-filling properties
        # Check that samples are relatively well-distributed by verifying
        # the mean is close to 0.5 (center of [0, 1])
        mean_per_dim = np.mean(samples_np, axis=0)
        assert np.allclose(mean_per_dim, 0.5, atol=0.15), (
            f"Mean per dimension {mean_per_dim} should be close to 0.5"
        )

    def test_sobol_deterministic(self):
        """Test Sobol sequence is deterministic."""
        samples1 = sobol_sample(10, 3, skip=0)
        samples2 = sobol_sample(10, 3, skip=0)

        np.testing.assert_array_equal(np.asarray(samples1), np.asarray(samples2))


class TestHaltonSample:
    """Test Halton sequence generation."""

    def test_halton_correct_shape(self):
        """Test Halton sequence generation with correct shape."""
        n_samples = 20
        n_dims = 5

        samples = halton_sample(n_samples, n_dims)

        assert samples.shape == (n_samples, n_dims)
        assert isinstance(samples, jnp.ndarray)

    def test_halton_uses_prime_bases(self):
        """Test Halton sequence uses prime bases for different dimensions."""
        n_samples = 100
        n_dims = 3

        samples = halton_sample(n_samples, n_dims)
        samples_np = np.asarray(samples)

        # All samples should be in [0, 1]
        assert np.all(samples_np >= 0.0)
        assert np.all(samples_np <= 1.0)

        # Check space-filling: samples should cover the space reasonably well
        mean_per_dim = np.mean(samples_np, axis=0)
        assert np.allclose(mean_per_dim, 0.5, atol=0.1), (
            f"Mean per dimension {mean_per_dim} should be close to 0.5"
        )

    def test_halton_deterministic(self):
        """Test Halton sequence is deterministic."""
        samples1 = halton_sample(10, 3, skip=0)
        samples2 = halton_sample(10, 3, skip=0)

        np.testing.assert_array_equal(np.asarray(samples1), np.asarray(samples2))


class TestBoundsTransformation:
    """Test bounds-aware scaling utilities."""

    def test_scale_samples_to_bounds(self):
        """Test bounds-aware scaling transforms samples to [lb, ub]."""
        # Samples in [0, 1]
        samples = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            ]
        )
        lb = jnp.array([0.0, -10.0, 100.0])
        ub = jnp.array([10.0, 10.0, 200.0])

        scaled = scale_samples_to_bounds(samples, lb, ub)
        scaled_np = np.asarray(scaled)

        # First row: all at lower bounds
        np.testing.assert_array_almost_equal(scaled_np[0], [0.0, -10.0, 100.0])
        # Second row: all at midpoints
        np.testing.assert_array_almost_equal(scaled_np[1], [5.0, 0.0, 150.0])
        # Third row: all at upper bounds
        np.testing.assert_array_almost_equal(scaled_np[2], [10.0, 10.0, 200.0])

    def test_center_samples_around_p0(self):
        """Test centering samples around p0 with scale factor."""
        # Samples in [0, 1]
        samples = jnp.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
            ]
        )
        p0 = jnp.array([5.0, 10.0])
        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([10.0, 20.0])
        scale_factor = 0.5  # Explore within 50% of range around p0

        centered = center_samples_around_p0(samples, p0, scale_factor, lb, ub)
        centered_np = np.asarray(centered)

        # Check that samples are centered around p0
        # With scale_factor=0.5, the range is 50% of (ub - lb) centered at p0
        # For dim 0: range = 10 * 0.5 = 5, so [p0 - 2.5, p0 + 2.5] = [2.5, 7.5]
        # For dim 1: range = 20 * 0.5 = 10, so [p0 - 5, p0 + 5] = [5, 15]

        # Sample at 0.0: should be at lower end of centered range
        # Sample at 0.5: should be at p0
        # Sample at 1.0: should be at upper end of centered range

        # Check middle sample is at p0
        np.testing.assert_array_almost_equal(centered_np[1], [5.0, 10.0])

        # Check all samples are within bounds
        assert np.all(centered_np >= np.asarray(lb))
        assert np.all(centered_np <= np.asarray(ub))


class TestSamplerFactory:
    """Test sampler factory function."""

    def test_get_sampler_lhs(self):
        """Test get_sampler returns LHS for 'lhs'."""
        sampler = get_sampler("lhs")
        samples = sampler(10, 3)

        assert samples.shape == (10, 3)
        # LHS samples should be in [0, 1]
        assert np.all(np.asarray(samples) >= 0.0)
        assert np.all(np.asarray(samples) <= 1.0)

    def test_get_sampler_sobol(self):
        """Test get_sampler returns Sobol for 'sobol'."""
        sampler = get_sampler("sobol")
        samples = sampler(10, 3)

        assert samples.shape == (10, 3)

    def test_get_sampler_halton(self):
        """Test get_sampler returns Halton for 'halton'."""
        sampler = get_sampler("halton")
        samples = sampler(10, 3)

        assert samples.shape == (10, 3)

    def test_get_sampler_invalid_type(self):
        """Test get_sampler raises error for invalid type."""
        with pytest.raises(ValueError, match="Unknown sampler type"):
            get_sampler("invalid_sampler")

    def test_get_sampler_case_insensitive(self):
        """Test get_sampler is case insensitive."""
        sampler1 = get_sampler("LHS")
        sampler2 = get_sampler("lhs")

        # Both should work and return similar samplers
        samples1 = sampler1(5, 2)
        samples2 = sampler2(5, 2)

        assert samples1.shape == samples2.shape


class TestCenterSamplesStayInBounds:
    """center_samples_around_p0 promises points inside [lb, ub] (no 1-ulp excess)."""

    @pytest.mark.parametrize("scale_factor", [0.5, 1.0, 2.0])
    def test_random_boxes_with_corner_samples(self, scale_factor):
        from nlsq.global_optimization.sampling import center_samples_around_p0

        rng = np.random.default_rng(0)
        n = 20000
        lb = rng.uniform(-10, 10, size=(n, 1))
        ub = lb + rng.uniform(1e-3, 20, size=(n, 1)) * rng.uniform(0.1, 3, (n, 1))
        p0 = lb + rng.uniform(0, 1, size=(n, 1)) * (ub - lb)
        samples = np.array([[0.0], [1.0], [0.5]])

        out = jax.vmap(
            lambda p, lo, hi: center_samples_around_p0(samples, p, scale_factor, lo, hi)
        )(jnp.asarray(p0), jnp.asarray(lb), jnp.asarray(ub))

        assert bool(jnp.all(out >= jnp.asarray(lb)[:, None, :]))
        assert bool(jnp.all(out <= jnp.asarray(ub)[:, None, :]))

    def test_infinite_bounds_still_work(self):
        from nlsq.global_optimization.sampling import center_samples_around_p0

        out = center_samples_around_p0(
            jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            jnp.array([1.0, 2.0]),
            1.0,
            jnp.array([-jnp.inf, 0.0]),
            jnp.array([jnp.inf, jnp.inf]),
        )
        assert bool(jnp.all(jnp.isfinite(out)))
        assert bool(out[0, 1] >= 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

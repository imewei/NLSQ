"""nlsq.fit(workflow="auto_global") forwards the sampler options."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["lhs", "sobol", "halton"])
def test_auto_global_forwards_sampler(sampler):
    x = np.linspace(0, 4, 80)
    y = 2 * np.exp(-1.3 * x) + 0.01 * np.random.default_rng(0).standard_normal(80)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = nlsq.fit(
            lambda x, a, b: a * jnp.exp(-b * x),
            x,
            y,
            bounds=([0, 0], [10, 5]),
            workflow="auto_global",
            n_starts=4,
            sampler=sampler,
        )
    assert r.multistart_diagnostics["sampler"] == sampler

"""Factory probes still need two observations for single-point chunks."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.exact_lm import exact_lm_fit


def test_single_point_factory_chunks_reject_batch_dependent_model():
    x = np.arange(4.0)
    y = np.ones(4)

    def factory(chunk):
        for i in range(len(x)):
            yield x[i : i + 1], y[i : i + 1]

    def batch_centered(x, a):
        return a * (x - jnp.mean(x))

    with pytest.raises(ValueError, match="not pointwise"):
        exact_lm_fit(batch_centered, factory, None, [1.0], chunk_size=1)

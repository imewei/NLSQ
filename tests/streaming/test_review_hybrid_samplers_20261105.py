"""All documented hybrid multistart samplers must reach the fitting phases."""

import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig


@pytest.mark.parametrize("sampler", ["lhs", "sobol", "halton"])
def test_hybrid_supports_each_configured_multistart_sampler(sampler):
    optimizer = AdaptiveHybridStreamingOptimizer(
        HybridStreamingConfig(
            enable_multistart=True,
            multistart_sampler=sampler,
            n_starts=4,
            elimination_rounds=0,
            batches_per_round=1,
            warmup_iterations=0,
            max_warmup_iterations=1,
            gauss_newton_max_iterations=10,
            verbose=0,
        )
    )
    x = np.linspace(0, 2, 20)
    result = optimizer.fit((x, 2 * x), lambda x, a: a * x, [1.0], verbose=0)
    assert result["success"], result["message"]
    np.testing.assert_allclose(result["x"], [2.0], atol=1e-6)

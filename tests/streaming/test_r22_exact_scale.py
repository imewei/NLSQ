"""exact_streaming's relative step test must be unit-independent."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit_large


def _m(x, a, b, c):
    return a * jnp.exp(-b * x) + c


@pytest.mark.filterwarnings("ignore:normal-equation")
def test_exact_streaming_converges_with_large_y_scale():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 5, 5003)
    y = 3.0 * np.exp(-1.2 * x) + 0.5 + rng.normal(0, 0.05, x.size)
    s = 1e9
    popt, _ = curve_fit_large(_m, x, y * s, p0=[s, 0.5, 0.0], method="exact_streaming")
    np.testing.assert_allclose(np.asarray(popt) / [s, 1, s], [3.0, 1.2, 0.5], atol=0.03)


def test_hybrid_workflow_rejects_data_mask_and_omit():
    from nlsq import fit
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    x = np.linspace(0, 5, 200)
    y = 2 * np.exp(-x) + 0.1
    for kw in ({"data_mask": np.ones(200, bool)}, {"nan_policy": "omit"}):
        with pytest.raises(ValueError, match="data_mask"):
            fit(_m, x, y, p0=[1, 1, 0], workflow=HybridStreamingConfig(), **kw)

"""Workflow memory estimates follow the public SciPy coordinate layout."""

import warnings
from unittest.mock import patch

import numpy as np

from nlsq.core.minpack import fit
from nlsq.streaming.large_dataset import LDMemoryConfig


def test_workflow_2d_memory_counts_features_not_observations():
    x = np.zeros((2, 10_000))
    y = np.zeros(10_000)
    with patch("nlsq.core.minpack._fit_with_config") as run:
        fit(
            lambda x, a: a * x[0] + x[1],
            x,
            y,
            p0=[1.0],
            workflow="auto",
            memory_limit_gb=0.5,
        )
    # These arrays and their one-column Jacobian fit in <1 MB; only the fixed
    # 0.1 GB solver overhead contributes materially to the 0.375 GB budget.
    assert run.call_args.kwargs["config"] is None


def test_explicit_auto_guess_survives_chunked_workflow_routing():
    x = np.arange(10.0)
    result = fit(lambda x, a: a * x, x, 2.0 * x, p0="auto", workflow=LDMemoryConfig())
    np.testing.assert_allclose(result.popt, [2.0], atol=1e-6)


def test_public_chunked_workflow_does_not_warn_about_internal_memory_option():
    from nlsq import fit as public_fit

    x = np.arange(10.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        public_fit(lambda x, a: a * x, x, 2.0 * x, p0=[1.0], workflow=LDMemoryConfig())
    assert not any(
        "Ignoring unknown" in str(w.message) and "memory_limit_gb" in str(w.message)
        for w in caught
    )

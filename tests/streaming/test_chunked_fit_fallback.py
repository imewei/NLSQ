"""Regression tests for the chunked-fit precision-weighted fallback.

Adversarial-review finding (high): when every successful chunk has unusable
covariance (``pcov`` is None / singular / all-inf), the precision-weighted
combiner must fall back to the *latest fitted* chunk parameters. The previous
code instead preserved ``current_params``, so a user-provided ``p0`` was
returned verbatim as a "successful" fit even though every chunk produced real
fitted parameters.
"""

import numpy as np

from nlsq.streaming.large_dataset import LargeDatasetFitter


def _make_fitter() -> LargeDatasetFitter:
    fitter = LargeDatasetFitter(memory_limit_gb=8.0)
    # The chunked-fit loop resets these accumulators once per fit (see
    # large_dataset.py); a direct unit call must mirror that reset.
    fitter._accum_information = None
    fitter._accum_info_vector = None
    return fitter


def test_unusable_covariance_uses_latest_fit_not_p0():
    """A successful chunk with unusable covariance must contribute its fitted
    params, never silently fall back to the user's initial guess."""
    fitter = _make_fitter()
    p0 = np.array([1.0, 2.0, 3.0])  # user-provided initial guess
    popt_chunk = np.array([10.0, 20.0, 30.0])  # what the chunk actually fitted

    combined, _history, _metric, stop = fitter._update_parameters_convergence(
        current_params=p0,
        popt_chunk=popt_chunk,
        pcov_chunk=None,  # unusable covariance -> no information contributed
        param_history=[p0.copy()],
        convergence_metric=np.inf,
        chunk_idx=1,
        n_chunks=3,
    )

    # Must reflect the fitted chunk, NOT the initial guess.
    np.testing.assert_allclose(combined, popt_chunk)
    assert not np.allclose(combined, p0)
    assert stop is False


def test_multichunk_unusable_covariance_threads_latest_fit():
    """Across many unusable-covariance chunks the running estimate tracks the
    latest fit, so the final result is never frozen at p0."""
    fitter = _make_fitter()
    p0 = np.array([0.0, 0.0])
    current = p0
    history = [p0.copy()]
    metric = np.inf
    last = None
    for i, popt in enumerate(
        [np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([3.0, 3.0])]
    ):
        current, history, metric, _stop = fitter._update_parameters_convergence(
            current, popt, None, history, metric, i, 3
        )
        last = popt

    np.testing.assert_allclose(current, last)  # last chunk's fit, not p0
    assert not np.allclose(current, p0)


def test_first_chunk_without_p0_still_uses_fit():
    """No-p0 path (current_params is None) is unchanged: the chunk fit is used."""
    fitter = _make_fitter()
    popt_chunk = np.array([5.0, 6.0])

    combined, history, metric, stop = fitter._update_parameters_convergence(
        current_params=None,
        popt_chunk=popt_chunk,
        pcov_chunk=None,
        param_history=[],
        convergence_metric=np.inf,
        chunk_idx=0,
        n_chunks=2,
    )

    np.testing.assert_allclose(combined, popt_chunk)
    assert metric == np.inf
    assert stop is False
    assert len(history) == 1


def test_usable_covariance_precision_weights_kick_in():
    """Once usable covariance arrives, the precision-weighted solve is used
    instead of the latest-fit fallback."""
    fitter = _make_fitter()
    p0 = np.array([0.0, 0.0])

    # Chunk 0: no covariance -> latest-fit fallback.
    c0, h0, m0, _ = fitter._update_parameters_convergence(
        p0, np.array([10.0, 10.0]), None, [p0.copy()], np.inf, 0, 2
    )
    np.testing.assert_allclose(c0, [10.0, 10.0])

    # Chunk 1: a usable (identity) covariance -> precision-weighted combination.
    cov = np.eye(2)
    c1, _h1, _m1, _ = fitter._update_parameters_convergence(
        c0, np.array([2.0, 2.0]), cov, h0, m0, 1, 2
    )
    # With a single usable chunk, the BLUE estimate is exactly that chunk's fit.
    np.testing.assert_allclose(c1, [2.0, 2.0])

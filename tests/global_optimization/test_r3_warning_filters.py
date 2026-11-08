"""GS-1: concurrent multi-start fits must not leak a global OptimizeWarning filter."""

import threading
import warnings

import jax.numpy as jnp
import numpy as np

import nlsq.global_optimization.multi_start as ms
from nlsq import OptimizeWarning, curve_fit
from nlsq.global_optimization.config import GlobalOptimizationConfig


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def test_overlapping_multistart_threads_do_not_leak_filters(monkeypatch):
    """Force the interleaving enter(A) enter(B) exit(A) exit(B)."""
    inside = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    idx = threading.local()

    def fake_start(f, x, y, p0, bounds, kw, cfkw):
        inside[idx.i].set()
        assert release[idx.i].wait(30)
        return np.asarray(p0), 0.0, None

    monkeypatch.setattr(ms, "_fit_single_start", fake_start)
    # Sequential path (one worker) so the fake runs on the calling thread.
    monkeypatch.setattr(ms, "_select_worker_count", lambda n: 1)

    # conftest ignores OptimizeWarning globally; start from a clean filter list.
    with warnings.catch_warnings():
        warnings.resetwarnings()
        x, y = np.linspace(0, 1, 5), np.ones(5)
        pts = np.array([[1.0, 1.0]])
        bounds = (np.zeros(2), np.full(2, 5.0))
        before = list(warnings.filters)

        def job(i):
            idx.i = i
            ms.MultiStartOrchestrator(
                GlobalOptimizationConfig(n_starts=1)
            ).evaluate_starting_points(_model, x, y, pts, bounds)

        threads = [threading.Thread(target=job, args=(i,)) for i in range(2)]
        threads[0].start()
        assert inside[0].wait(30)
        threads[1].start()
        assert inside[1].wait(30)
        release[0].set()
        threads[0].join(30)
        release[1].set()
        threads[1].join(30)

        assert warnings.filters == before

        # A plain fit that cannot converge must still emit OptimizeWarning.
        xs = np.linspace(0, 1, 20)
        with warnings.catch_warnings(record=True) as rec:
            curve_fit(_model, xs, np.zeros_like(xs) + 1.0, p0=[1.0, 1.0], max_nfev=1)
        assert any(issubclass(w.category, OptimizeWarning) for w in rec)

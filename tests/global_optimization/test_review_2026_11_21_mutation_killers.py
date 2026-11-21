"""Tests that kill mutants the earlier suites let survive (caching, callbacks,
global_optimization, diagnostics); each was verified red on its mutant."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


# --- caching/memory_manager.py: adaptive TTL tiers -----------------------
def _ttl(stamps, base=3.0):
    from nlsq.caching.memory_manager import MemoryManager

    m = MemoryManager(memory_cache_ttl=base, adaptive_ttl=True)
    m._call_frequency_tracker.clear()
    m._call_frequency_tracker.extend(stamps)
    return m._get_effective_ttl()


def test_adaptive_ttl_tiers():
    assert _ttl([5.0, 5.0]) == 15.0  # zero time span
    assert _ttl([i * 0.001 for i in range(100)]) == 15.0  # ~1000 calls/s
    assert _ttl([i * 0.05 for i in range(20)]) == 10.0  # ~21 calls/s
    assert _ttl([i * 0.5 for i in range(5)]) == 3.0  # 2.5 calls/s -> base


# --- caching/unified_cache.py: compile-time bookkeeping bounded by LRU ---
def test_compile_times_dropped_on_eviction():
    from nlsq.caching.unified_cache import UnifiedCache

    cache = UnifiedCache(maxsize=1, enable_stats=True)
    x = jnp.array([1.0])
    for k in range(4):
        cache.get_or_compile(lambda a, k=k: a + k, (x,), {}, static_argnums=())
    assert len(cache._cache) == 1
    assert len(cache._compile_times) == 1


# --- global_optimization/method_selector.py: fixed params are ignored ----
def test_scale_ratio_ignores_zero_width_ranges():
    from nlsq.global_optimization.method_selector import MethodSelector

    r = MethodSelector().compute_scale_ratio([0, 1, 0], [1, 1, 1e4])
    assert r == pytest.approx(1e4)


def test_scale_threshold_is_strict():
    from nlsq.global_optimization.method_selector import MethodSelector

    sel = MethodSelector(scale_threshold=100.0)
    assert sel.compute_scale_ratio([0, 0], [1, 100]) == pytest.approx(100.0)
    assert sel.select("auto", [0, 0], [1, 100]) == "multi-start"


# --- global_optimization/bipop.py ----------------------------------------
def test_bipop_large_popsize_capped_at_32x():
    from nlsq.global_optimization.bipop import BIPOPRestarter

    r = BIPOPRestarter(base_popsize=10, n_params=2, max_restarts=40, seed=0)
    r.restart_count = 20  # large regime, k=10 -> uncapped 2**11
    r._use_large_pop = True
    assert r.get_next_popsize() == 320


def test_bipop_stagnation_is_strict():
    from nlsq.global_optimization.bipop import BIPOPRestarter

    r = BIPOPRestarter(
        base_popsize=10, n_params=2, max_restarts=3, min_fitness_spread=1e-6, seed=0
    )
    assert not r.check_stagnation(1e-6)
    assert r.check_stagnation(9e-7)


# --- global_optimization/config.py ---------------------------------------
def test_config_rejects_boundary_values():
    from nlsq.global_optimization.config import GlobalOptimizationConfig as C

    with pytest.raises(ValueError):
        C(elimination_fraction=1.0)
    with pytest.raises(ValueError):
        C(seed=2**63)
    C(seed=2**63 - 1)


# --- global_optimization/checkpoint.py -----------------------------------
def test_checkpoint_version_mismatch_rejected(tmp_path):
    import h5py

    from nlsq.global_optimization.checkpoint import HPCCheckpointManager
    from tests.global_optimization.test_checkpoint import (
        _sample_fingerprint,
        _sample_state,
    )

    m = HPCCheckpointManager()
    fp = _sample_fingerprint()
    p = tmp_path / "run.h5"
    m.save(p, _sample_state(), fp)
    with h5py.File(p, "r+") as f:
        f.attrs["version"] = "9.9"
    with pytest.raises(ValueError, match="version"):
        m.load(p, fp)


# --- global_optimization/tournament.py -----------------------------------
def test_prior_failure_is_ranked_worst_even_if_finite_this_round():
    from nlsq.global_optimization.config import GlobalOptimizationConfig
    from nlsq.global_optimization.tournament import TournamentSelector

    cfg = GlobalOptimizationConfig(
        n_starts=4, elimination_rounds=1, elimination_fraction=0.5, batches_per_round=1
    )
    sel = TournamentSelector(np.array([[0.0], [1.0], [2.0], [3.0]]), cfg)
    sel.loss_sums[1] = np.inf  # candidate 1 failed earlier; it is exact (a=1) now
    sel._run_single_round(
        iter([(np.ones(2), np.ones(2))]), lambda x, a: a * x, 0, top_m=1
    )
    assert not sel.survival_mask[1]


# --- callbacks.py --------------------------------------------------------
def test_early_stopping_ignores_nonfinite_cost():
    from nlsq.callbacks import EarlyStopping

    cb = EarlyStopping(patience=2, verbose=False)
    cb(0, 1.0, None, {})
    for i in (1, 2, 3, 4):
        cb(i, float("nan"), None, {})  # must not consume patience
    assert cb.wait == 0


def test_callback_chain_stop_is_or_combined():
    from nlsq.callbacks import CallbackChain

    chain = CallbackChain(lambda i, c, p, info: True, lambda i, c, p, info: None)
    assert chain(0, 1.0, None, {}) is True


def test_run_end_reason_ftol_wins_after_5_stagnant_gens():
    from nlsq.global_optimization.cmaes_optimizer import _run_end_reason

    assert _run_end_reason(1e-12, 1e-8, 5) == "ftol"
    assert _run_end_reason(1e-12, 1e-8, 0) == "xtol"
    assert _run_end_reason(1.0, 1e-8, 0) is None


def _score(pairs=0, sloppy=None):
    from types import SimpleNamespace as NS

    from nlsq.diagnostics.health_report import _compute_health_score
    from nlsq.diagnostics.types import DiagnosticLevel, DiagnosticsConfig

    ident = NS(
        available=True,
        numerical_rank=3,
        n_params=3,
        condition_number=1.0,
        highly_correlated_pairs=[(0, 1, 0.99)] * pairs,
    )
    sm = None if sloppy is None else NS(available=True, is_sloppy=sloppy)
    return _compute_health_score(
        ident, None, sm, DiagnosticsConfig(level=DiagnosticLevel.FULL)
    )


def test_health_score_correlation_deduction_capped_at_0_2():
    assert _score(pairs=4) == pytest.approx(0.8)
    assert _score(pairs=9) == pytest.approx(0.8)


def test_health_score_sloppy_deducts_0_1():
    assert _score(sloppy=True) == pytest.approx(0.9)
    assert _score(sloppy=False) == pytest.approx(1.0)

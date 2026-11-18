"""Regression tests for the 2026-11-18 round-2 global_optimization review (fixer F4)."""

import json
import os
import subprocess
import sys
import textwrap
import warnings

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.config import MemoryConfig, get_memory_config, memory_context
from nlsq.global_optimization.bounds_transform import (
    transform_from_bounds,
    transform_to_bounds,
)
from nlsq.global_optimization.checkpoint import (
    CMAESCheckpointState,
    HPCCheckpointManager,
    compute_fingerprint,
)
from nlsq.global_optimization.cmaes_config import CMAESConfig, estimate_cmaes_memory_gb
from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.global_optimization.method_selector import MethodSelector
from nlsq.global_optimization.tournament import TournamentSelector


def _linear(x, a):
    return a * x


def _cmaes(**cfg):
    pytest.importorskip("evosax")
    from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

    return CMAESOptimizer(
        CMAESConfig(restart_strategy="none", seed=4, **{"max_generations": 3, **cfg})
    )


def test_cmaes_rejects_column_predictions():
    """Codex P06 HIGH: an (n, 1) prediction broadcast against (n,) data into (n, n)."""
    x = jnp.arange(1.0, 9.0)

    def column_model(x, a):
        return (a * x)[:, None]

    with pytest.raises(ValueError, match="one prediction per observation"):
        _cmaes().fit(column_model, x, 2 * x, p0=[1.0], bounds=([0.0], [4.0]))


def test_cmaes_integer_residuals_do_not_overflow():
    """Codex P06 HIGH: int32 residuals wrapped before the squared sum."""

    def count_model(x, a):
        return jnp.round(a * x).astype(jnp.int32)

    opt = _cmaes(max_generations=20, refine_with_nlsq=False)
    x = np.array([1], dtype=np.int32)
    y = np.array([50000], dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = opt.fit(count_model, x, y, p0=[1000.0], bounds=([0.0], [60000.0]))
    assert res["cmaes_diagnostics"]["best_fitness"] >= 0.0


def test_cmaes_bounds_length_must_match_parameters():
    """Codex P06 MEDIUM: bounds=([0.], [4., 4.]) gave a covariance of the wrong size."""
    x = jnp.arange(1.0, 9.0)
    with pytest.raises(ValueError, match="one entry per parameter"):
        _cmaes(refine_with_nlsq=False).fit(
            lambda x, a, b: a * x + b,
            x,
            2 * x + 1,
            p0=[1.0, 1.0],
            bounds=([0.0], [4.0, 4.0]),
        )


def test_cmaes_refinement_handles_points_first_x():
    """Codex P06 MEDIUM: refinement rejected (N, k) x that the global search accepted."""
    x = jnp.stack([jnp.arange(8.0), jnp.arange(8.0) ** 2], axis=1)

    def model(x, a, b):
        return a * x[:, 0] + b * x[:, 1]

    res = _cmaes().fit(model, x, model(x, 2.0, 3.0), p0=[1.0, 1.0], bounds=(0.0, 5.0))
    np.testing.assert_allclose(res["popt"], [2.0, 3.0], rtol=1e-6)


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_cmaes_config_rejects_unusable_seed(seed):
    """Codex P06 MEDIUM: seeds both backends fail on were accepted at construction."""
    with pytest.raises(ValueError, match="seed"):
        CMAESConfig(seed=seed)


def test_cmaes_memory_estimate_counts_resident_data_when_chunked():
    """Codex P06 MEDIUM / Claude C06 LOW-2: the chunked estimate omitted x and y."""
    n = 10_000_000
    chunked = estimate_cmaes_memory_gb(n, popsize=16, data_chunk_size=50_000)
    assert chunked >= n * 16 / 1024**3
    assert estimate_cmaes_memory_gb(n, popsize=16) == pytest.approx(3.725290298461914)


def test_bounds_transform_widens_integer_bounds():
    """Codex P06 MEDIUM: int64 spans wrapped before the float transform."""
    lb = jnp.array([-5 * 10**18], dtype=jnp.int64)
    ub = jnp.array([5 * 10**18], dtype=jnp.int64)
    mid = transform_to_bounds(jnp.zeros(1), lb, ub)
    assert float(mid[0]) == pytest.approx(0.0, abs=1e6)
    back = transform_from_bounds(jnp.zeros(1, dtype=jnp.float64), lb, ub)
    assert float(back[0]) == pytest.approx(0.0, abs=1e-6)


def test_method_selector_widens_integer_bounds():
    """Codex P06 LOW: int16 endpoint subtraction wrapped (ratio 1 instead of 60000)."""
    ratio = MethodSelector().compute_scale_ratio(
        np.array([-30000, 0], dtype=np.int16), np.array([30000, 1], dtype=np.int16)
    )
    assert ratio == pytest.approx(60000.0)


def test_global_config_to_dict_is_json_ready():
    """Codex P06 LOW: np.float32 scalars survived validation and broke json.dumps."""
    cfg = GlobalOptimizationConfig(
        scale_factor=np.float32(0.5), elimination_fraction=np.float32(0.25)
    )
    json.dumps(cfg.to_dict())


def _checkpoint_state(generation):
    return CMAESCheckpointState(
        generation_counter=generation,
        mean=jnp.zeros(3),
        std=jnp.array(0.5),
        p_std=jnp.zeros(3),
        p_c=jnp.zeros(3),
        C=jnp.eye(3),
        B=jnp.eye(3),
        D=jnp.ones(3),
        best_solution=jnp.array([1.0, 2.0, 3.0]),
        best_fitness=0.01,
        key_data=np.array([1, 2], dtype=np.uint32),
        fitness_history=[1.0, 0.5],
        popsize=8,
    )


def test_checkpoint_with_wrong_key_length_falls_back_to_backup(tmp_path):
    """Codex P06 HIGH: a 3-element key passed load() and failed on resume, past the .bak."""
    manager = HPCCheckpointManager()
    fp = compute_fingerprint(
        model_id="m",
        xdata=np.linspace(0, 1, 10),
        ydata=np.linspace(0, 1, 10),
        sigma=None,
        bounds=(np.zeros(3), np.ones(3)),
        config_fields={
            "popsize": 8,
            "sigma": 0.5,
            "tol_fun": 1e-8,
            "tol_x": 1e-8,
            "seed": 1,
        },
    )
    path = tmp_path / "run.h5"
    manager.save(path, _checkpoint_state(2), fp)
    manager.save(path, _checkpoint_state(3), fp)  # generation 2 is now the .bak
    with h5py.File(path, "r+") as f:
        del f["state/key_data"]
        f["state"].create_dataset("key_data", data=np.array([1, 2, 3], dtype=np.uint32))
    assert manager.load(path, fp).generation_counter == 2


def test_tournament_disqualified_candidate_stays_eliminated():
    """Codex P06 HIGH: a candidate that failed round 1 beat the only valid one in round 2."""
    cfg = GlobalOptimizationConfig(
        n_starts=3, elimination_rounds=2, batches_per_round=1, elimination_fraction=0.5
    )
    selector = TournamentSelector(np.array([[0.0], [1.0], [2.0], [3.0]]), cfg)

    def model(x, a):
        # NaN for positive parameters while y == 0 (round 1), finite afterwards
        return jnp.where((a > 0) & (jnp.max(x) < 1.5), jnp.nan, a + 0.0 * x)

    batches = iter(
        [(np.array([1.0]), np.array([0.0])), (np.array([2.0]), np.array([3.0]))]
    )
    best = selector.run_tournament(batches, model, top_m=1)
    assert float(best[0][0]) == 0.0


def test_tournament_rejects_column_response():
    """Codex P06 MEDIUM: y of shape (n, 1) broadcast into an (n, n) residual matrix."""
    cfg = GlobalOptimizationConfig(
        n_starts=1, elimination_rounds=1, batches_per_round=1
    )
    selector = TournamentSelector(np.array([[1.8], [2.0]]), cfg)
    batches = iter([(np.array([1.0, 2.0]), np.array([[2.0], [4.0]]))])
    with pytest.raises(ValueError, match="does not match"):
        selector.run_tournament(batches, lambda x, a: a * x, top_m=1)


def test_multistart_return_eval_matches_standard_route():
    """Codex P06 MEDIUM: sigma-NaN omission and scalar models disagreed with curve_fit."""
    x = np.arange(1.0, 7.0)
    y = 2 * x
    sigma = np.array([1, 1, 1, 1, 1, np.nan])
    multi = {"multistart": True, "n_starts": 3, "bounds": ([0.0], [5.0])}
    for kw in ({}, multi):
        out = curve_fit(
            _linear,
            x,
            y,
            p0=[1.0],
            sigma=sigma,
            nan_policy="omit",
            data_mask=np.ones(6, bool),
            return_eval=True,
            **kw,
        )
        assert np.shape(out[2]) == (5,)

        def constant(x, a):
            return a + 0 * jnp.sum(x)

        out = curve_fit(constant, x, y, p0=[1.0], return_eval=True, **kw)
        assert np.shape(out[2]) == (6,)


def test_multistart_workers_inherit_memory_context():
    """Codex P06 MEDIUM: pool threads saw the global memory config, not the caller's."""
    from nlsq.global_optimization import multi_start as ms

    seen = []
    real = ms._fit_single_start

    def spy(*args):
        seen.append(get_memory_config().memory_limit_gb)
        return real(*args)

    x = np.arange(1.0, 9.0)
    ms._fit_single_start = spy
    try:
        with memory_context(MemoryConfig(memory_limit_gb=0.125)):
            curve_fit(
                _linear,
                x,
                2 * x,
                p0=[1.0],
                bounds=([0.0], [5.0]),
                multistart=True,
                n_starts=3,
            )
    finally:
        ms._fit_single_start = real
    assert seen
    assert set(seen) == {0.125}


@pytest.mark.serial
def test_multistart_workers_use_distinct_devices():
    """Codex P06 MEDIUM: all workers allocated on device 0 despite several devices."""
    code = textwrap.dedent(
        """
        import warnings; warnings.simplefilter("ignore")
        import numpy as np, jax.numpy as jnp
        from nlsq import curve_fit
        from nlsq.global_optimization import multi_start as ms
        seen = set()
        real = ms._fit_single_start
        def spy(*a):
            seen.add(next(iter(jnp.zeros(1).devices())).id)
            return real(*a)
        ms._fit_single_start = spy
        x = np.arange(1.0, 9.0)
        curve_fit(lambda x, a: a * x, x, 2 * x, p0=[1.0], bounds=([0.0], [5.0]),
                  multistart=True, n_starts=2)
        print(sorted(seen))
        """
    )
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "NLSQ_SKIP_GPU_CHECK": "1",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "[0, 1]"
    assert jax.devices()  # the parent process is unaffected


@pytest.mark.parametrize(
    "kw",
    [
        {"nan_policy": "omit"},
        {"nan_policy": "omit", "sigma": np.array([np.nan, 1, 1, 1, 1, 1, 1, 1.0])},
        {"nan_policy": "omit", "data_mask": np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)},
        {},
    ],
    ids=["omit", "omit+sigma", "omit+mask", "plain"],
)
def test_multistart_return_eval_equals_the_standard_route(kw):
    """Wave 2: the multistart eval is the same array (rows and values) as the TRF route's."""
    x = np.arange(1.0, 9.0)
    y = 2 * x
    if kw.get("nan_policy"):
        y[7] = np.nan
    std = curve_fit(_linear, x, y, p0=[1.0], return_eval=True, **kw)
    multi = curve_fit(
        _linear,
        x,
        y,
        p0=[1.0],
        return_eval=True,
        multistart=True,
        n_starts=3,
        bounds=([0.0], [5.0]),
        **kw,
    )
    assert np.shape(multi[2]) == np.shape(std[2])
    np.testing.assert_allclose(multi[2], std[2], rtol=1e-6)

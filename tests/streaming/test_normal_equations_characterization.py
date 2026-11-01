"""Characterization tests for streaming normal-equation (J^T J / J^T r) accumulation.

These pin the CURRENT behavior of the exact-LM solver and of the hybrid
optimizer's two accumulation implementations so that a later refactor that moves
the accumulation into a shared module can prove it is behavior-preserving.

Golden constants were captured once by running the unmodified code on the fixed
seed dataset below (``_dataset``) and pasting ``repr`` output here; see
``GOLDEN``. Do not regenerate them to make a failing test pass: a mismatch means
behavior changed.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.exact_lm import exact_lm_fit
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.checkpoint import CheckpointManager
from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase

RTOL = 1e-8
ATOL = 1e-10

N_POINTS = 20_000
CHUNK = 5_000
P0 = np.array([1.0, 0.5])


def model(x, a, b):
    return a * jnp.exp(-b * x)


def _dataset():
    rng = np.random.default_rng(12345)
    x = np.linspace(0.0, 5.0, N_POINTS)
    y = 2.0 * np.exp(-1.3 * x) + rng.normal(0.0, 0.05, x.shape)
    return x, y


def _config(strategy="loop", penalty=False, chunk=CHUNK, **extra):
    extra.setdefault("enable_checkpoints", False)
    return HybridStreamingConfig(
        chunk_size=chunk,
        loop_strategy=strategy,
        enable_group_variance_regularization=penalty,
        group_variance_lambda=0.01,
        group_variance_indices=[(0, 2)] if penalty else None,
        verbose=0,
        **extra,
    )


def _setup(config):
    """Optimizer after Phase 0, plus device data and normalized p0."""
    x, y = _dataset()
    opt = AdaptiveHybridStreamingOptimizer(config=config)
    opt._setup_normalization(model, jnp.asarray(P0), None)
    p0_norm = opt.normalizer.normalize(jnp.asarray(P0))
    return opt, (jnp.asarray(x), jnp.asarray(y)), p0_norm


def run_exact():
    x, y = _dataset()
    # chunk_size < n so the accumulation really spans several chunks.
    res = exact_lm_fit(model, x, y, P0, chunk_size=CHUNK)
    return np.asarray(res.x), float(res.cost), np.asarray(res.pcov)


def run_hybrid_phase2(penalty, strategy, chunk=CHUNK):
    """Phase 0 -> Phase 2 -> Phase 3 (Phase 1 warmup skipped for determinism)."""
    opt, data, p0_norm = _setup(_config(strategy, penalty, chunk))
    r2 = opt._run_phase2_gauss_newton(data, p0_norm)
    r3 = opt._run_phase3_finalize(
        r2["final_params"], r2["JTJ_final"], r2["residual_sum_sq"], N_POINTS
    )
    return (
        np.asarray(r3["popt"]),
        float(r2["residual_sum_sq"]),
        np.asarray(r3["pcov"]),
    )


def _chunk_inputs(opt, p0_norm):
    x, y = _dataset()
    n_p = len(P0)
    return (
        jnp.asarray(x[:1000]),
        jnp.asarray(y[:1000]),
        p0_norm,
        jnp.zeros((n_p, n_p)),
        jnp.zeros(n_p),
    )


def run_hybrid_accumulate_chunk():
    opt, _, p0_norm = _setup(_config())
    jtj, jtr, cost = opt._accumulate_jtj_jtr(*_chunk_inputs(opt, p0_norm))
    return np.asarray(jtj), np.asarray(jtr), float(cost)


def run_phases_iteration(penalty):
    """One GaussNewtonPhase iteration (includes the penalty).

    Starts off p0 by an uneven shift: at the normalized p0 both parameters
    coincide, so the group variance, and hence its gradient, would be zero.
    """
    opt, (x, y), p0_norm = _setup(_config("loop", penalty))
    phase = GaussNewtonPhase(_config("loop", penalty), opt.normalized_model)
    start = p0_norm + jnp.array([0.3, -0.2])
    r = phase._gauss_newton_iteration((x, y), start, 1.0)
    return (
        float(r["new_cost"]),
        float(r["gradient_norm"]),
        np.asarray(r["step"]),
    )


def run_phases_accumulate_chunk():
    opt, _, p0_norm = _setup(_config())
    phase = GaussNewtonPhase(_config(), opt.normalized_model)
    jtj, jtr, cost = phase._accumulate_jtj_jtr(*_chunk_inputs(opt, p0_norm))
    return np.asarray(jtj), np.asarray(jtr), float(cost)


# Captured from the unmodified code (see module docstring).
GOLDEN = {
    "exact_lm": {
        "x": [2.0011127994390256, 1.3003529583685232],
        "cost": 25.051368401882545,
        "pcov": [
            [3.256465004542377e-06, 2.116502423163314e-06],
            [2.116502423163314e-06, 2.7516447345475737e-06],
        ],
    },
    "hybrid_nopen_scan": {
        "x": [2.0011127893458007, 1.3003529394349982],
        "rss": 50.10273680376472,
        "pcov": [
            [3.2564649572421217e-06, 2.1165023723439565e-06],
            [2.1165023723439565e-06, 2.751644642292871e-06],
        ],
    },
    "hybrid_nopen_loop": {
        "x": [2.0011127893458007, 1.3003529394349982],
        "rss": 50.10273680376473,
        "pcov": [
            [3.256464957242112e-06, 2.1165023723439417e-06],
            [2.1165023723439417e-06, 2.751644642292854e-06],
        ],
    },
    "hybrid_pen_scan": {
        "x": [1.9914201296045526, 1.2659764128226814],
        "rss": 51.66824287612398,
        "pcov": [
            [3.2696704640810668e-06, 2.0791088435790158e-06],
            [2.0791088435790158e-06, 2.6443861162447973e-06],
        ],
    },
    "hybrid_pen_loop": {
        "x": [1.9914201296045526, 1.2659764128226814],
        "rss": 51.66824287612398,
        "pcov": [
            [3.2696704640810875e-06, 2.079108843579029e-06],
            [2.079108843579029e-06, 2.644386116244803e-06],
        ],
    },
    "hybrid_chunk": {
        "jtj": [
            [884.9021767516939, -52.95157880853472],
            [-52.95157880853472, 4.3173025558065525],
        ],
        "jtr": [726.3339744285704, -40.16728089847266],
        "cost": 608.218483150702,
    },
    "phases_chunk": {
        "jtj": [
            [884.9021767516939, -52.95157880853472],
            [-52.95157880853472, 4.3173025558065525],
        ],
        "jtr": [726.3339744285704, -40.16728089847266],
        "cost": 608.218483150702,
    },
    # chunk_size=6000 -> chunks of 6000, 6000, 6000, 2000 (ragged tail), penalty on.
    "hybrid_ragged_scan": {
        "x": [1.9914201296045526, 1.2659764128226814],
        "rss": 51.66824287612398,
        "pcov": [
            [3.2696704640810914e-06, 2.0791088435790344e-06],
            [2.0791088435790344e-06, 2.644386116244809e-06],
        ],
    },
    "hybrid_ragged_loop": {
        "x": [1.9914201296045526, 1.2659764128226814],
        "rss": 51.66824287612398,
        "pcov": [
            [3.269670464081091e-06, 2.07910884357903e-06],
            [2.07910884357903e-06, 2.644386116244803e-06],
        ],
    },
    # GaussNewtonPhase._gauss_newton_iteration from p0_norm + [0.3, -0.2], radius 1.0.
    "phases_iter_nopen": {
        "new_cost": 424.7557098754678,
        "gradient_norm": 3432.7554333242992,
        "step": [0.2393817160828744, 0.7716573655329735],
    },
    "phases_iter_pen": {
        "new_cost": 424.8078163421925,
        "gradient_norm": 3432.758834603918,
        "step": [0.2393817356061301, 0.7716573477717127],
    },
}


def _assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_exact_lm_golden():
    x, cost, pcov = run_exact()
    g = GOLDEN["exact_lm"]
    _assert_close(x, g["x"])
    _assert_close(cost, g["cost"])
    _assert_close(pcov, g["pcov"])


@pytest.mark.slow
@pytest.mark.parametrize("penalty", [False, True], ids=["no_penalty", "penalty"])
@pytest.mark.parametrize("strategy", ["scan", "loop"])
def test_hybrid_phase2_golden(monkeypatch, penalty, strategy):
    # Confirm the production switch (config.loop_strategy) really selects the path.
    calls = {"scan": 0}
    orig_scan = AdaptiveHybridStreamingOptimizer._accumulate_jtj_jtr_scan

    def spy_scan(self, *a, **k):
        calls["scan"] += 1
        return orig_scan(self, *a, **k)

    monkeypatch.setattr(
        AdaptiveHybridStreamingOptimizer, "_accumulate_jtj_jtr_scan", spy_scan
    )

    x, rss, pcov = run_hybrid_phase2(penalty, strategy)

    assert (calls["scan"] > 0) == (strategy == "scan")
    g = GOLDEN[f"hybrid_{'pen' if penalty else 'nopen'}_{strategy}"]
    _assert_close(x, g["x"])
    _assert_close(rss, g["rss"])
    _assert_close(pcov, g["pcov"])


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["scan", "loop"])
def test_hybrid_phase2_ragged_tail_golden(strategy):
    x, rss, pcov = run_hybrid_phase2(True, strategy, chunk=6000)
    g = GOLDEN[f"hybrid_ragged_{strategy}"]
    _assert_close(x, g["x"])
    _assert_close(rss, g["rss"])
    _assert_close(pcov, g["pcov"])


@pytest.mark.parametrize("penalty", [False, True], ids=["no_penalty", "penalty"])
def test_phases_iteration_golden(penalty):
    """phases/ scales the group-variance penalty differently from adaptive_hybrid
    (no n_points, no 0.5); pin that as it is today."""
    cost, gnorm, step = run_phases_iteration(penalty)
    g = GOLDEN[f"phases_iter_{'pen' if penalty else 'nopen'}"]
    _assert_close(cost, g["new_cost"])
    _assert_close(gnorm, g["gradient_norm"])
    _assert_close(step, g["step"])


def test_phases_iteration_penalty_is_not_vacuous():
    pen = run_phases_iteration(True)
    nopen = run_phases_iteration(False)
    assert abs(pen[0] - nopen[0]) > 1e-2
    assert abs(pen[1] - nopen[1]) > 1e-4


def test_hybrid_accumulate_chunk_golden():
    jtj, jtr, cost = run_hybrid_accumulate_chunk()
    g = GOLDEN["hybrid_chunk"]
    _assert_close(jtj, g["jtj"])
    _assert_close(jtr, g["jtr"])
    _assert_close(cost, g["cost"])


def test_phases_accumulate_chunk_golden():
    jtj, jtr, cost = run_phases_accumulate_chunk()
    g = GOLDEN["phases_chunk"]
    _assert_close(jtj, g["jtj"])
    _assert_close(jtr, g["jtr"])
    _assert_close(cost, g["cost"])


def _ckpt_config(tmp_path, **extra):
    return _config(
        "loop",
        enable_checkpoints=True,
        checkpoint_dir=str(tmp_path),
        checkpoint_frequency=1,
        **extra,
    )


def _record_saves(opt):
    """Wrap ``_save_checkpoint`` to snapshot the state it is about to serialise."""
    saved = []
    orig = opt._save_checkpoint

    def spy(path):
        saved.append(
            {
                "path": Path(path),
                "JTJ": np.array(opt.phase2_JTJ_accumulator),
                "JTr": np.array(opt.phase2_JTr_accumulator),
                "params": np.array(opt.normalized_params),
            }
        )
        return orig(path)

    opt._save_checkpoint = spy
    return saved


@pytest.mark.slow
def test_hybrid_checkpoint_roundtrip(tmp_path):
    x, y = _dataset()
    config = _ckpt_config(tmp_path, warmup_iterations=20, max_warmup_iterations=20)
    opt = AdaptiveHybridStreamingOptimizer(config=config)
    saved = _record_saves(opt)
    result = opt.fit(
        (jnp.asarray(x), jnp.asarray(y)), model, jnp.asarray(P0), verbose=0
    )
    assert result["success"]

    phase2 = [s for s in saved if "phase2" in s["path"].name]
    assert phase2, "fit() wrote no Phase 2 checkpoint"
    manager = CheckpointManager(config)
    for snap in phase2:
        state = manager.load(snap["path"])
        assert state.phase2_JTJ_accumulator is not None
        _assert_close(state.phase2_JTJ_accumulator, snap["JTJ"])
        _assert_close(state.phase2_JTr_accumulator, snap["JTr"])
        _assert_close(state.normalized_params, snap["params"])


@pytest.mark.slow
def test_hybrid_phase2_resume_continues_from_checkpoint(tmp_path):
    x, y = _dataset()
    data = (jnp.asarray(x), jnp.asarray(y))

    # Uninterrupted run, checkpointing after every Phase 2 iteration.
    opt, _, p0_norm = _setup(_ckpt_config(tmp_path))
    saved = _record_saves(opt)
    full = opt._run_phase2_gauss_newton(data, p0_norm)
    assert len(saved) >= 3
    ckpt = saved[1]  # checkpoint written during the second iteration

    # Fresh optimizer: Phase 0 + checkpoint restore, then Phase 2 directly (fit()
    # would restart normalization and warmup).
    opt2 = AdaptiveHybridStreamingOptimizer(config=_config("loop"))
    opt2._setup_normalization(model, jnp.asarray(P0), None)
    opt2._load_checkpoint(ckpt["path"])
    restored = np.array(opt2.normalized_params)
    _assert_close(restored, ckpt["params"])
    _assert_close(opt2.phase2_JTJ_accumulator, ckpt["JTJ"])
    _assert_close(opt2.phase2_JTr_accumulator, ckpt["JTr"])

    first_start = []
    orig = opt2._accumulate_jtj_jtr

    def spy(xc, yc, params, *a, **k):
        if not first_start:
            first_start.append(np.array(params))
        return orig(xc, yc, params, *a, **k)

    opt2._accumulate_jtj_jtr = spy
    resumed = opt2._run_phase2_gauss_newton(data, opt2.normalized_params)

    # Phase 2 started from exactly the restored parameters, not p0 or a warmup result.
    _assert_close(first_start[0], restored)
    assert not np.allclose(restored, np.array(p0_norm))

    r3_full = opt._run_phase3_finalize(
        full["final_params"], full["JTJ_final"], full["residual_sum_sq"], N_POINTS
    )
    r3_resumed = opt2._run_phase3_finalize(
        resumed["final_params"],
        resumed["JTJ_final"],
        resumed["residual_sum_sq"],
        N_POINTS,
    )
    _assert_close(np.asarray(r3_resumed["popt"]), np.asarray(r3_full["popt"]))


def test_hybrid_padded_tail_finite():
    def log_model(x, a, b):
        return a * jnp.log(x) + b

    phase = GaussNewtonPhase(_config(), log_model)
    n_valid, n_pad = 64, 100
    x_valid = jnp.linspace(1.0, 3.0, n_valid)
    y_valid = 0.7 * jnp.log(x_valid) + 0.2
    x_pad = jnp.concatenate([x_valid, jnp.zeros(n_pad - n_valid)])
    y_pad = jnp.concatenate([y_valid, jnp.zeros(n_pad - n_valid)])
    params = jnp.array([1.0, 0.5])
    jtj0, jtr0 = jnp.zeros((2, 2)), jnp.zeros(2)

    # Without the mask the padded zeros poison the accumulation (log(0) = -inf).
    bad = phase._accumulate_jtj_jtr(x_pad, y_pad, params, jtj0, jtr0)
    assert not np.all(np.isfinite(np.asarray(bad[0])))

    ref = phase._accumulate_jtj_jtr(x_valid, y_valid, params, jtj0, jtr0)
    got = phase._accumulate_jtj_jtr(
        x_pad, y_pad, params, jtj0, jtr0, valid_size=n_valid
    )
    for g, r in zip(got, ref, strict=True):
        assert np.all(np.isfinite(np.asarray(g)))
        np.testing.assert_allclose(g, r, rtol=RTOL, atol=ATOL)


def test_hybrid_chunk_cost_convention():
    """Hybrid chunk cost is sum(r^2), not the kernel's 0.5 * sum(r^2)."""
    opt, _, p0_norm = _setup(_config())
    x_chunk, y_chunk, params, jtj0, jtr0 = _chunk_inputs(opt, p0_norm)
    _, _, cost = opt._accumulate_jtj_jtr(x_chunk, y_chunk, params, jtj0, jtr0)
    expected = float(jnp.sum((y_chunk - opt.normalized_model(x_chunk, *params)) ** 2))
    assert float(cost) == pytest.approx(expected, rel=1e-12)


def test_hybrid_kernel_follows_reassigned_jacobian():
    """Dropping _jacobian_fn_compiled rebuilds the kernel on the inline fallback."""
    opt, _, p0_norm = _setup(_config())
    args = _chunk_inputs(opt, p0_norm)
    ref = opt._accumulate_jtj_jtr(*args)
    first = opt._get_normal_eq()
    opt._jacobian_fn_compiled = None
    assert opt._get_normal_eq() is not first
    got = opt._accumulate_jtj_jtr(*args)
    for g, r in zip(got, ref, strict=True):
        np.testing.assert_allclose(g, r, rtol=RTOL, atol=ATOL)

"""Regression tests for SWM2 / SWM3 / warmup best-pairing (2026-11-04 review)."""

import jax
import jax.numpy as jnp
import numpy as np

from nlsq.precision.parameter_normalizer import (
    NormalizedModelWrapper,
    ParameterNormalizer,
)
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase
from nlsq.streaming.phases.warmup import WarmupPhase

jax.config.update("jax_enable_x64", True)


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


def _exp3(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def test_all_rejected_steps_return_initial_point():
    """SWM2: best must be seeded from the start, not a rejected trial."""
    x = jnp.linspace(0, 5, 200)
    y = _exp(x, 2.0, 0.3)
    cfg = HybridStreamingConfig(
        chunk_size=256,
        gauss_newton_max_iterations=1,
        trust_region_initial=50.0,
        verbose=0,
    )
    gn = GaussNewtonPhase(cfg, _exp)
    p0 = jnp.array([2.0, 5.0])
    c0 = gn._compute_cost(p0, x, y)
    res = gn.run(
        (x, y), p0, [], {"best_params_global": p0, "best_cost_global": float("inf")}
    )
    assert float(res["best_cost"]) <= c0 * (1 + 1e-9)
    np.testing.assert_allclose(np.asarray(res["best_params"]), [2.0, 5.0])


def _penalized_phase(reg, p0, strategy="p0"):
    cfg = HybridStreamingConfig(
        chunk_size=1024,
        enable_group_variance_regularization=reg,
        group_variance_lambda=0.1,
        group_variance_indices=[(0, 3)],
        verbose=0,
    )
    normalizer = ParameterNormalizer(p0, strategy=strategy)
    return GaussNewtonPhase(cfg, NormalizedModelWrapper(_exp3, normalizer)), normalizer


def _capture_jtj_jtr(gn, q, x, y):
    seen = {}
    orig = gn._solve_gauss_newton_step

    def spy(JTJ, JTr, tr):
        seen["JTJ"], seen["JTr"] = JTJ, JTr
        return orig(JTJ, JTr, tr)

    gn._solve_gauss_newton_step = spy
    gn._gauss_newton_iteration((x, y), q, 1.0)
    return seen


def test_group_variance_terms_match_cost_scaling():
    """SWM3 + HYB2: J^T r / J^T J penalty terms are the derivatives of the
    n_points-scaled penalty over PHYSICAL parameters (jax.grad / jax.hessian)."""
    n = 1000
    lam = 0.1
    x = jnp.linspace(0, 5, n)
    y = _exp3(x, 2.0, 0.3, 1.0)
    p0 = jnp.array([1.5, 0.5, 0.7])
    gn, normalizer = _penalized_phase(True, p0)
    gn0, _ = _penalized_phase(False, p0)
    assert np.ptp(np.asarray(normalizer.scales)) > 0  # else the test is blind
    q = normalizer.normalize(jnp.array([1.5, 2.0, 0.2]))

    on = _capture_jtj_jtr(gn, q, x, y)
    off = _capture_jtj_jtr(gn0, q, x, y)

    def penalty(qq):
        return jnp.var(normalizer.denormalize(qq))

    np.testing.assert_allclose(
        np.asarray(on["JTr"] - off["JTr"]),
        -0.5 * lam * n * np.asarray(jax.grad(penalty)(q)),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(on["JTJ"] - off["JTJ"]),
        0.5 * lam * n * np.asarray(jax.hessian(penalty)(q)),
        rtol=1e-8,
    )

    # ...and J^T r agrees with -0.5 * finite-difference gradient of _compute_cost.
    eps = 1e-6
    fd = np.array(
        [
            (
                gn._compute_cost(q.at[i].add(eps), x, y)
                - gn._compute_cost(q.at[i].add(-eps), x, y)
            )
            / (2 * eps)
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(np.asarray(on["JTr"]), -0.5 * fd, rtol=1e-4)


def test_group_variance_cost_is_normalization_invariant():
    """HYB2: the penalty depends on physical params, not on the scaling used."""
    x = jnp.linspace(0, 5, 300)
    y = _exp3(x, 2.0, 0.3, 1.0)
    phys = jnp.array([1.5, 2.0, 0.2])
    costs = []
    for strategy in ("p0", "none"):
        gn, normalizer = _penalized_phase(True, jnp.array([1.5, 0.5, 0.7]), strategy)
        costs.append(gn._compute_cost(normalizer.normalize(phys), x, y))
    np.testing.assert_allclose(costs[0], costs[1], rtol=1e-10)


def test_warmup_best_loss_matches_best_params():
    x = jnp.linspace(0, 5, 200)
    y = _exp(x, 2.0, 0.3)
    cfg = HybridStreamingConfig(
        chunk_size=256,
        warmup_iterations=0,
        max_warmup_iterations=4,
        enable_warm_start_detection=False,
        enable_cost_guard=False,
        active_switching_criteria=[],
        verbose=0,
    )
    w = WarmupPhase(cfg, _exp)
    loss_fn = w._create_loss_fn()
    r = w.run(
        (x, y),
        jnp.array([1.0, 1.0]),
        [],
        {"best_params_global": None, "best_cost_global": float("inf")},
    )
    np.testing.assert_allclose(
        r["best_loss"], float(loss_fn(r["best_params"], x, y)), rtol=1e-12
    )

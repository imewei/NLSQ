"""Regression tests for the Codex root-module review, round 1 (2026-11-16)."""

import threading
import time
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize._lsq.common import solve_lsq_trust_region

from nlsq.common_jax import CommonJIT, solve_lsq_trust_region_jax
from nlsq.config import JAXConfig, MemoryConfig, get_memory_config, memory_context


@pytest.fixture
def restore_config():
    JAXConfig()
    saved = (JAXConfig.get_memory_config(), JAXConfig.get_large_dataset_config())
    yield
    JAXConfig.set_memory_config(saved[0])
    JAXConfig.set_large_dataset_config(saved[1])


# --- W1: rank-deficient interior minimizer ----------------------------------


def test_rank_deficient_interior_minimizer_is_taken():
    uf = jnp.array([0.1, 1.0])
    s = jnp.array([1.0, 0.0])
    V = jnp.eye(2)
    p, alpha, _ = solve_lsq_trust_region_jax(2, 2, uf, s, V, 1.0)
    np.testing.assert_allclose(np.asarray(p), [-0.1, 0.0], atol=1e-12)
    assert float(alpha) == 0.0


def test_rank_deficient_never_worse_than_boundary_step():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m, n = 6, 4
        J = rng.normal(size=(m, n))
        J[:, -1] = J[:, 0]  # rank n-1
        f = rng.normal(size=m)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        uf = U.T @ f
        Delta = float(rng.uniform(0.05, 5.0))
        p, _, _ = solve_lsq_trust_region_jax(
            n, m, jnp.asarray(uf), jnp.asarray(s), jnp.asarray(Vt.T), Delta
        )
        p = np.asarray(p)
        assert np.linalg.norm(p) <= Delta * (1 + 1e-9)
        # never uphill from the zero step
        assert 0.5 * np.sum((J @ p + f) ** 2) <= 0.5 * np.sum(f**2) + 1e-12


def test_least_squares_makes_progress_on_rank_deficient_problem():
    from nlsq.core.least_squares import LeastSquares

    res = LeastSquares().least_squares(
        lambda x: jnp.array([x[0] + 0.1, 1.0]), np.array([0.0, 0.0]), max_nfev=3
    )
    assert abs(float(res.x[0]) + 0.1) < 1e-6


def test_full_rank_still_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(60):
        m, n = 8, 4
        J = rng.normal(size=(m, n))
        f = rng.normal(size=m)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        uf = U.T @ f
        Delta = float(rng.uniform(0.05, 5.0))
        ref, _, _ = solve_lsq_trust_region(n, m, uf, s, Vt.T, Delta)
        p, _, _ = solve_lsq_trust_region_jax(
            n, m, jnp.asarray(uf), jnp.asarray(s), jnp.asarray(Vt.T), Delta
        )
        np.testing.assert_allclose(np.asarray(p), ref, rtol=1e-6, atol=1e-9)


# --- W2: overflow-safe column norms -----------------------------------------


@pytest.mark.parametrize("mag", [1e200, 1e-200])
def test_jac_scale_survives_extreme_magnitudes(mag):
    scale, inv = CommonJIT().compute_jac_scale(jnp.array([[mag], [mag]]))
    np.testing.assert_allclose(float(inv[0]), mag * np.sqrt(2.0), rtol=1e-12)
    np.testing.assert_allclose(float(scale[0]), 1 / (mag * np.sqrt(2.0)), rtol=1e-12)


def test_jac_scale_zero_column_is_one():
    _scale, inv = CommonJIT().compute_jac_scale(jnp.array([[0.0, 3.0], [0.0, 4.0]]))
    np.testing.assert_allclose(np.asarray(inv), [1.0, 5.0])


# --- W3: build_quadratic_1d composes with jit --------------------------------


def test_build_quadratic_1d_jittable_and_correct():
    cj = CommonJIT()
    J = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]])
    g = jnp.array([1.0, -2.0])
    s = jnp.array([0.3, 0.7])
    s0 = jnp.array([-0.2, 0.1])
    diag = jnp.array([0.5, 2.0])

    def run(J, g, s, s0, diag):
        return cj.build_quadratic_1d(J, g, s, diag=diag, s0=s0)

    got = jax.jit(run)(J, g, s, s0, diag)
    a, b, c = run(J, g, s, s0, diag)
    Jn, gn, sn, s0n, dn = map(np.asarray, (J, g, s, s0, diag))
    v, u = Jn @ sn, Jn @ s0n
    exp_a = 0.5 * (v @ v + (sn * dn) @ sn)
    exp_b = gn @ sn + u @ v + (s0n * dn) @ sn
    exp_c = 0.5 * u @ u + gn @ s0n + 0.5 * (s0n * dn) @ s0n
    np.testing.assert_allclose([float(x) for x in got], [exp_a, exp_b, exp_c])
    np.testing.assert_allclose([float(a), float(b), float(c)], [exp_a, exp_b, exp_c])


# --- W4: setters coordinate with contexts ------------------------------------


def test_setter_during_context_is_not_lost(restore_config):
    JAXConfig.set_memory_config(MemoryConfig(memory_limit_gb=1.0))
    inside, set_done = threading.Event(), threading.Event()
    seen = {}

    def holder():
        with memory_context(MemoryConfig(memory_limit_gb=2.0)):
            inside.set()
            set_done.wait(10)  # the setter runs while this context is open
            seen["inside"] = get_memory_config().memory_limit_gb

    def setter():
        inside.wait(10)
        JAXConfig.set_memory_config(MemoryConfig(memory_limit_gb=3.0))
        set_done.set()

    ts = [threading.Thread(target=holder), threading.Thread(target=setter)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    assert seen["inside"] == 2.0  # the holder never sees B's value
    assert get_memory_config().memory_limit_gb == 3.0  # and B's update survives


# --- W5: TPU is not a GPU ----------------------------------------------------


def _dev(platform):
    return types.SimpleNamespace(platform=platform, __str__=lambda self: platform)


def test_get_device_info_separates_gpu_and_tpu(monkeypatch):
    from nlsq.device import get_device_info

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(jax, "devices", lambda: [_dev("tpu")])
    info = get_device_info()
    assert info["gpu_count"] == 0
    assert info["using_gpu"] is False
    assert info["tpu_count"] == 1

    monkeypatch.setattr(jax, "devices", lambda: [_dev("gpu"), _dev("gpu")])
    info = get_device_info()
    assert info["gpu_count"] == 2 and info["using_gpu"] is True
    assert info["tpu_count"] == 0


def test_check_gpu_availability_ignores_tpu(monkeypatch):
    from nlsq.device import check_gpu_availability

    monkeypatch.delenv("NLSQ_SKIP_GPU_CHECK", raising=False)
    monkeypatch.setattr("nlsq.device.get_gpu_info", lambda: ("Fake GPU", 8.0))
    monkeypatch.setattr("nlsq.device.get_system_cuda_version", lambda: ("12.4", 12))
    monkeypatch.setattr(jax, "devices", lambda: [_dev("tpu")])
    assert check_gpu_availability(warn=False) is False


# --- context-local config (contextvars) --------------------------------------


def test_context_does_not_deadlock_with_setter_on_helper_thread(restore_config):
    JAXConfig.set_memory_config(MemoryConfig(memory_limit_gb=1.0))
    with memory_context(MemoryConfig(memory_limit_gb=4.0)):
        t = threading.Thread(
            target=lambda: JAXConfig.set_memory_config(
                MemoryConfig(memory_limit_gb=2.0)
            ),
            daemon=True,
        )
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "setter on another thread blocked by the context"
        # the context is local to this thread/task: the helper's value is global
        assert get_memory_config().memory_limit_gb == 4.0
    assert get_memory_config().memory_limit_gb == 2.0


def test_contexts_on_two_threads_do_not_serialize(restore_config):
    both_inside = threading.Barrier(2, timeout=5)
    seen = {}

    def work(name, value):
        with memory_context(MemoryConfig(memory_limit_gb=value)):
            both_inside.wait()  # blocks forever if the second context is excluded
            seen[name] = get_memory_config().memory_limit_gb

    ts = [
        threading.Thread(target=work, args=("a", 2.0), daemon=True),
        threading.Thread(target=work, args=("b", 3.0), daemon=True),
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert seen == {"a": 2.0, "b": 3.0}


def test_context_nested_restores_and_survives_exception(restore_config):
    JAXConfig.set_memory_config(MemoryConfig(memory_limit_gb=1.0))
    with memory_context(MemoryConfig(memory_limit_gb=2.0)):
        with (
            pytest.raises(RuntimeError),
            memory_context(MemoryConfig(memory_limit_gb=3.0)),
        ):
            assert get_memory_config().memory_limit_gb == 3.0
            raise RuntimeError
        assert get_memory_config().memory_limit_gb == 2.0
    assert get_memory_config().memory_limit_gb == 1.0


def test_setter_inside_context_stays_local(restore_config):
    from nlsq.config import set_memory_limits

    JAXConfig.set_memory_config(MemoryConfig(memory_limit_gb=1.0))
    with memory_context(MemoryConfig(memory_limit_gb=2.0)):
        set_memory_limits(5.0)
        assert get_memory_config().memory_limit_gb == 5.0
    assert get_memory_config().memory_limit_gb == 1.0


def test_large_dataset_context_is_context_local(restore_config):
    from nlsq.config import (
        LargeDatasetConfig,
        get_large_dataset_config,
        large_dataset_context,
    )

    seen = []
    cfg = LargeDatasetConfig(enable_automatic_solver_selection=False)
    with large_dataset_context(cfg):
        t = threading.Thread(
            target=lambda: seen.append(
                get_large_dataset_config().enable_automatic_solver_selection
            )
        )
        t.start()
        t.join(5)
        assert get_large_dataset_config().enable_automatic_solver_selection is False
    assert seen == [True]  # plain threads do not inherit the context
    assert get_large_dataset_config().enable_automatic_solver_selection is True


def test_jac_scale_infinite_column_stays_infinite():
    cj = CommonJIT()
    scale, inv = cj.compute_jac_scale(
        jnp.array([[1e200, 0.0], [1e200, 1e-200], [np.inf, 1.0]])
    )
    assert np.isinf(float(inv[0])) and float(scale[0]) == 0.0
    np.testing.assert_allclose(float(inv[1]), 1.0)


def test_chain_warns_when_stop_drops_earlier_error():
    from nlsq.callbacks import CallbackChain, StopOptimization

    def bad(iteration, cost, params, info):
        raise RuntimeError("boom")

    def stop(iteration, cost, params, info):
        raise StopOptimization

    with (
        pytest.warns(RuntimeWarning, match="boom"),
        pytest.raises(StopOptimization),
    ):
        CallbackChain(bad, stop)(1, 1.0, np.zeros(1), {})


# --- truncated GN step must not hide a resolvable decrease (Codex r2) --------


def _model_value(uf, s, p):
    return 0.5 * float(np.sum((np.asarray(uf) + np.asarray(s) * np.asarray(p)) ** 2))


@pytest.mark.parametrize(
    ("uf", "s", "Delta"),
    [
        ([0.0, 1.0], [1.0, 1e-16], 2e16),  # discarded direction has all the decrease
        ([0.0, 1e16], [1.0, 1e-16], 1.0),  # huge gradient along a tiny singular value
    ],
)
def test_truncated_step_not_accepted_when_discarded_direction_decreases(uf, s, Delta):
    p, _, _ = solve_lsq_trust_region_jax(
        2, 2, jnp.asarray(uf), jnp.asarray(s), jnp.eye(2), Delta
    )
    p = np.asarray(p)
    assert np.linalg.norm(p) <= Delta * (1 + 1e-9)
    assert np.linalg.norm(p) > 0  # the zero step was the bug
    ref, _, _ = solve_lsq_trust_region(
        2, 2, np.asarray(uf), np.asarray(s), np.eye(2), Delta
    )
    assert _model_value(uf, s, p) <= _model_value(uf, s, ref) * (1 + 1e-9)


def test_first_repro_reaches_model_minimum():
    p, _, _ = solve_lsq_trust_region_jax(
        2, 2, jnp.array([0.0, 1.0]), jnp.array([1.0, 1e-16]), jnp.eye(2), 2e16
    )
    assert _model_value([0.0, 1.0], [1.0, 1e-16], np.asarray(p)) < 1e-9


def test_exact_zero_singular_value_still_takes_interior_step():
    p, alpha, n_iter = solve_lsq_trust_region_jax(
        2, 2, jnp.array([0.1, 1.0]), jnp.array([1.0, 0.0]), jnp.eye(2), 1.0
    )
    np.testing.assert_allclose(np.asarray(p), [-0.1, 0.0], atol=1e-12)
    assert float(alpha) == 0.0 and int(n_iter) == 0


# --- Codex r3: prefer the finite full minimizer; never NaN --------------------


def test_full_minimizer_preferred_when_feasible():
    uf, s = jnp.array([1.0, 1e-4]), jnp.array([1.0, 1e-16])
    p, alpha, n_iter = solve_lsq_trust_region_jax(2, 2, uf, s, jnp.eye(2), 2e12)
    np.testing.assert_allclose(np.asarray(p), [-1.0, -1e12], rtol=1e-9)
    assert float(alpha) == 0.0 and int(n_iter) == 0


def test_feasibility_test_survives_squaring_overflow():
    uf, s = jnp.array([0.0, 1.0]), jnp.array([1.0, 1e-160])
    p, _, _ = solve_lsq_trust_region_jax(2, 2, uf, s, jnp.eye(2), 2e160)
    np.testing.assert_allclose(np.asarray(p), [0.0, -1e160], rtol=1e-9)


@pytest.mark.parametrize("Delta", [1e100, 1e-100, 1.0])
def test_secular_branch_never_returns_nan(Delta):
    uf, s = jnp.array([0.0, 1.0]), jnp.array([1.0, 1e-160])
    p, alpha, _ = solve_lsq_trust_region_jax(2, 2, uf, s, jnp.eye(2), Delta)
    assert np.all(np.isfinite(np.asarray(p))) and np.isfinite(float(alpha))
    assert np.linalg.norm(np.asarray(p)) <= Delta * (1 + 1e-9)


def test_representable_decrease_in_small_singular_direction_is_not_dropped():
    uf, s = np.array([1.0, 1e-5]), np.array([1.0, 1e-16])
    p, _, _ = solve_lsq_trust_region_jax(
        2, 2, jnp.asarray(uf), jnp.asarray(s), jnp.eye(2), 1e10
    )
    ref, _, _ = solve_lsq_trust_region(2, 2, uf, s, np.eye(2), 1e10)
    assert _model_value(uf, s, p) <= _model_value(uf, s, ref) * (1 + 1e-9)
    assert _model_value(uf, s, p) < 0.5e-10  # [-1, 0] gives 5e-11


def test_random_sweep_never_worse_than_scipy_or_outside_radius():
    rng = np.random.default_rng(5)
    checked = 0
    for k in range(600):
        m, n = int(rng.integers(2, 8)), int(rng.integers(1, 6))
        J = rng.normal(size=(m, n)) * 10 ** rng.uniform(-3, 3)
        if k % 3 == 1 and n > 1:
            J[:, -1] = J[:, 0]
        if k % 3 == 2:
            J = J * (rng.random(n) < 0.7)
        f = rng.normal(size=m) * 10 ** rng.uniform(-3, 3)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        uf, V, Delta = U.T @ f, Vt.T, float(10 ** rng.uniform(-2, 3))
        with np.errstate(all="ignore"):
            ref, _, _ = solve_lsq_trust_region(n, m, uf, s, V, Delta)
        if not np.all(np.isfinite(ref)):
            continue
        p, _, _ = solve_lsq_trust_region_jax(
            n, m, jnp.asarray(uf), jnp.asarray(s), jnp.asarray(V), Delta
        )
        p = np.asarray(p)
        checked += 1
        assert np.all(np.isfinite(p))
        assert np.linalg.norm(p) <= Delta * (1 + 1e-8)
        vp = 0.5 * np.sum((uf + s * (V.T @ p)) ** 2)
        vr = 0.5 * np.sum((uf + s * (V.T @ ref)) ** 2)
        assert vp <= vr * (1 + 1e-6) + 1e-12
    assert checked > 400


def test_nonfinite_full_step_is_not_accepted_as_interior():
    uf, s = jnp.array([0.0, 1e150]), jnp.array([1.0, 1e-160])
    p, alpha, _ = solve_lsq_trust_region_jax(2, 2, uf, s, jnp.eye(2), 1.0)
    p = np.asarray(p)
    assert np.all(np.isfinite(p)) and np.isfinite(float(alpha))
    np.testing.assert_allclose(p, [0.0, -1.0], atol=1e-9)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_safe_norm_propagates_nonfinite_and_solver_does_not_crash(bad):
    from nlsq.common_jax import _safe_norm

    assert not np.isfinite(float(_safe_norm(jnp.array([1.0, bad]))))
    assert float(_safe_norm(jnp.zeros(3))) == 0.0
    solve_lsq_trust_region_jax(
        2, 2, jnp.array([1.0, bad]), jnp.array([1.0, 0.5]), jnp.eye(2), 1.0
    )


def test_steepest_descent_fallback_survives_overflowing_product():
    uf, s = jnp.array([1e120, 1.0]), jnp.array([1e200, 1.0])  # s*uf overflows
    p, alpha, _ = solve_lsq_trust_region_jax(2, 2, uf, s, jnp.eye(2), 0.5)
    p = np.asarray(p)
    assert np.all(np.isfinite(p)) and np.isfinite(float(alpha))
    assert np.linalg.norm(p) <= 0.5 * (1 + 1e-9)


def test_last_resort_returns_zero_step_not_nan():
    uf, s = jnp.array([jnp.nan, 1.0]), jnp.array([1.0, 1.0])
    p, alpha, _ = solve_lsq_trust_region_jax(2, 2, uf, s, jnp.eye(2), 0.5)
    assert np.all(np.isfinite(np.asarray(p))) and np.isfinite(float(alpha))

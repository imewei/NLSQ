"""A3: small exact SVDs run on the host device under a GPU backend."""

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core import device_policy
from nlsq.core.covariance import pcov_from_jacobian
from nlsq.core.minpack import _covariance_svd
from nlsq.core.trf_jit import _svd_no_bounds

EPS = np.finfo(float).eps


@pytest.fixture
def force_host(monkeypatch):
    """Force the policy on and count host-path entries via ``host_device()``."""
    calls = []
    real = device_policy.host_device

    def spy():
        calls.append(1)
        return real()

    monkeypatch.setattr(device_policy, "use_host_for_small_svd", lambda shape, b: True)
    monkeypatch.setattr(device_policy, "host_device", spy)
    return calls


def _problem(rank_deficient):
    rng = np.random.default_rng(0)
    if rank_deficient:
        J = rng.standard_normal((1000, 3))
        J = np.column_stack([J, J[:, 0]])
    else:
        J = rng.standard_normal((1000, 3))
    return jnp.asarray(J), jnp.asarray(rng.standard_normal(1000)), jnp.ones(J.shape[1])


def test_policy_false_on_cpu_for_any_shape():
    for shape in [(1, 1), (1000, 3), (10**6, 10)]:
        assert device_policy.use_host_for_small_svd(shape, "cpu") is False


def test_policy_false_above_threshold():
    n = device_policy.HOST_SVD_MAX_ELEMENTS // 3 + 1
    assert device_policy.use_host_for_small_svd((n, 3), "gpu") is False
    assert device_policy.use_host_for_small_svd((n - 1, 3), "gpu") is True


@pytest.mark.parametrize("backend", ["gpu", "cuda", "rocm"])
def test_policy_true_for_small_shape_on_gpu(backend):
    assert device_policy.use_host_for_small_svd((1000, 3), backend) is True


def test_policy_false_for_other_backends():
    assert device_policy.use_host_for_small_svd((1000, 3), "tpu") is False


def test_host_device_is_cpu():
    assert device_policy.host_device() == jax.devices("cpu")[0]


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_host_svd_matches_device_svd(monkeypatch, rank_deficient):
    J, f, d = _problem(rank_deficient)
    monkeypatch.setattr(device_policy, "use_host_for_small_svd", lambda s, b: False)
    ref = [np.asarray(a) for a in _svd_no_bounds(J, d, f)]
    monkeypatch.setattr(device_policy, "use_host_for_small_svd", lambda s, b: True)
    _, U, s, V, uf = (np.asarray(a) for a in _svd_no_bounds(J, d, f))
    sr, Vr = ref[2], ref[3]
    n, p = J.shape
    cut = EPS * max(n, p) * sr[0]
    keep = sr > cut
    np.testing.assert_allclose(s[keep], sr[keep], rtol=1e-12)
    np.testing.assert_allclose(s[~keep], sr[~keep], rtol=0, atol=cut)
    Jh = np.asarray(J * d)
    np.testing.assert_allclose(
        U @ np.diag(s) @ V.T, Jh, rtol=0, atol=1e-12 * np.linalg.norm(Jh)
    )
    np.testing.assert_allclose(U.T @ U, np.eye(p), atol=1e-12)
    np.testing.assert_allclose(V.T @ V, np.eye(p), atol=1e-12)
    k = keep.sum()
    np.testing.assert_allclose(
        V[:, :k] @ V[:, :k].T, Vr[:, :k] @ Vr[:, :k].T, atol=1e-10
    )
    np.testing.assert_allclose(uf, U.T @ np.asarray(f), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_pcov_equal_between_host_and_device(monkeypatch, rank_deficient):
    J, _, _ = _problem(rank_deficient)
    args = (0.5, J.shape[0], J.shape[1], False, _covariance_svd)
    monkeypatch.setattr(device_policy, "use_host_for_small_svd", lambda s, b: False)
    ref, _ = pcov_from_jacobian(J, *args)
    calls = []
    real = device_policy.host_device
    monkeypatch.setattr(device_policy, "host_device", lambda: calls.append(1) or real())
    monkeypatch.setattr(device_policy, "use_host_for_small_svd", lambda s, b: True)
    got, _ = pcov_from_jacobian(J, *args)
    assert calls
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=0)


def test_host_path_entered_when_forced(force_host):
    J, f, d = _problem(False)
    _svd_no_bounds(J, d, f)
    assert force_host


def test_returned_placement(force_host):
    J, f, d = _problem(False)
    dev = next(iter(J.devices()))
    J_h, _U, s, V, uf = _svd_no_bounds(J, d, f)
    for a in (J_h, s, V, uf):
        assert a.devices() == {dev}


def test_cpu_backend_does_not_call_device_put(monkeypatch):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU default backend only")

    def boom(*a, **k):
        raise AssertionError("device_put called on CPU backend")

    monkeypatch.setattr(jax, "device_put", boom)
    J, f, d = _problem(False)
    _svd_no_bounds(J, d, f)
    _covariance_svd(J)


_EMULATION_SCRIPT = textwrap.dedent(
    """
    import jax, jax.numpy as jnp, numpy as np
    from nlsq import curve_fit
    from nlsq.core import device_policy
    from nlsq.core.trf_jit import _svd_no_bounds

    devs = jax.devices()
    assert len(devs) >= 2 and devs[0].platform == "cpu"
    rng = np.random.default_rng(0)
    J = jax.device_put(jnp.asarray(rng.standard_normal((200, 3))), devs[0])
    f = jax.device_put(jnp.asarray(rng.standard_normal(200)), devs[0])
    d = jax.device_put(jnp.ones(3), devs[0])
    x = np.linspace(0, 3, 50)
    y = 2 * np.exp(-1.3 * x) + 0.01 * np.sin(7 * x)
    model = lambda x, a, b: a * jnp.exp(-b * x)

    ref = [np.asarray(a) for a in _svd_no_bounds(J, d, f)]
    pref = curve_fit(model, x, y, p0=[1.5, 1.0])

    calls = []
    device_policy.use_host_for_small_svd = lambda shape, backend: True
    device_policy.host_device = lambda: calls.append(1) or jax.devices("cpu")[1]
    out = _svd_no_bounds(J, d, f)
    assert calls
    for i in (0, 2, 3, 4):  # J_h, s, V, uf (U stays on the host device)
        np.testing.assert_allclose(np.asarray(out[i]), ref[i], rtol=1e-10, atol=1e-12)
        assert out[i].devices() == {devs[0]}, out[i].devices()
    got = curve_fit(model, x, y, p0=[1.5, 1.0])
    np.testing.assert_allclose(got[0], pref[0], rtol=1e-10)
    np.testing.assert_allclose(got[1], pref[1], rtol=1e-8)
    print("OK")
    """
)


@pytest.mark.serial
def test_cross_device_placement_emulated():
    import os

    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run(
        [sys.executable, "-c", _EMULATION_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-3000:]


@pytest.mark.gpu
def test_host_svd_matches_device_svd_on_real_gpu(monkeypatch):
    if jax.default_backend() not in ("gpu", "cuda", "rocm"):
        pytest.skip("needs a GPU backend")
    J, f, d = _problem(False)
    monkeypatch.setattr(device_policy, "use_host_for_small_svd", lambda s, b: False)
    _, _, sr, _, ufr = (np.asarray(a) for a in _svd_no_bounds(J, d, f))
    monkeypatch.setattr(device_policy, "use_host_for_small_svd", lambda s, b: True)
    _, _, s, V, uf = _svd_no_bounds(J, d, f)
    dev = next(iter(J.devices()))
    assert s.devices() == V.devices() == uf.devices() == {dev}
    np.testing.assert_allclose(np.asarray(s), sr, rtol=1e-10)
    np.testing.assert_allclose(
        np.abs(np.asarray(uf)), np.abs(ufr), rtol=1e-8, atol=1e-10
    )

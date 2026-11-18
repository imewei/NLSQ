"""Regression tests for the 2026-11-18 three-brain review (fixer F2: core/)."""

import os
import subprocess
import sys
import textwrap
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.core.functions import exponential_decay, exponential_growth
from nlsq.core.least_squares import LeastSquares
from nlsq.core.loss_functions import LossFunctionsJIT
from nlsq.core.sparse_jacobian import (
    SparseJacobianComputer,
    detect_jacobian_sparsity,
)

_X = jnp.array([1.0, 2.0, 3.0])
_Y = jnp.array([2.0, 4.0, 6.0])


def test_c02_nonfinite_jacobian_repair_does_not_report_gtol_success():
    def model(x, a, b):
        return a * jnp.sqrt(b) * x

    x = np.linspace(1, 4, 20)
    y = 2 * np.sqrt(1.5) * x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="zero gradient"):
            LeastSquares(enable_stability=True).least_squares(
                model, [1.0, 0.0], xdata=x, ydata=y
            )


def test_c03_growth_p0_converges_with_baseline_on_shifted_x():
    x = np.linspace(50, 60, 100)
    y = 2e-10 * np.exp(0.3 * x) + 5
    p0 = exponential_growth.estimate_p0(x, y)
    # model at the start of the data is within 1% of the data (a offset-corrected)
    model0 = p0[0] * np.exp(p0[1] * x[0]) + p0[2]
    assert model0 == pytest.approx(y[0], rel=1e-9)
    res = curve_fit(
        exponential_growth, x, y, p0="auto", bounds=exponential_growth.bounds()
    )
    assert res[0] == pytest.approx([2e-10, 0.3, 5.0], rel=1e-5)


def test_p02_decay_p0_unsigned_input_matches_float():
    x = np.array([0.0, 1.0, 2.0])
    p_u8 = exponential_decay.estimate_p0(x, np.array([200, 210, 220], dtype=np.uint8))
    p_f = exponential_decay.estimate_p0(x, np.array([200.0, 210.0, 220.0]))
    assert p_u8 == p_f


def test_p02_uncached_wrapper_sees_changed_global_names(monkeypatch):
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    g = {"coef": 2.0, "coef2": 3.0}
    exec("def f1(x, a):\n    return coef * a * x", g)
    exec("def f2(x, a):\n    return coef2 * a * x", g)
    ls = LeastSquares()
    x = np.array([1.0, 2.0, 3.0])
    y = 6 * x
    a1 = ls.least_squares(g["f1"], [1.0], xdata=x, ydata=y).x[0]
    a2 = ls.least_squares(g["f2"], [1.0], xdata=x, ydata=y).x[0]
    assert a1 == pytest.approx(3.0)
    assert a2 == pytest.approx(2.0)


def test_p02_reverse_mode_ignores_zero_sigma_on_masked_row():
    res = curve_fit(
        lambda x, a: a * x,
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        p0=[1.0],
        sigma=np.array([1.0, 0.0, 1.0]),
        data_mask=np.array([True, False, True]),
        jacobian_mode="rev",
    )
    assert res[0][0] == pytest.approx(2.0)


def test_p02_wrong_length_data_mask_rejected():
    with pytest.raises(ValueError, match="data_mask"):
        LeastSquares().least_squares(
            lambda x, a: x * a,
            [1.0],
            xdata=jnp.array([1.0, 2.0]),
            ydata=jnp.array([2.0, 4.0]),
            data_mask=[False],
        )


def test_p02_nonfinite_x0_rejected():
    with pytest.raises(ValueError, match="finite"):
        LeastSquares().least_squares(lambda p: jnp.ones(1), [np.inf])


def test_p02_nonfinite_custom_loss_cost_rejected():
    def loss(z):
        return jnp.stack(
            [jnp.full_like(z, jnp.nan), jnp.ones_like(z), jnp.zeros_like(z)]
        )

    with pytest.raises(ValueError, match="initial cost"):
        LeastSquares().least_squares(lambda p: p - 2, [2.0], loss=loss)


def test_p02_direct_route_rejects_invalid_jacobian_mode():
    with pytest.raises(ValueError, match="jacobian_mode"):
        LeastSquares().least_squares(lambda p: p - 2, [1.0], jacobian_mode="bogus")


def test_agy_one_parameter_1d_analytic_jacobian():
    x = np.linspace(1, 10, 10)
    popt, _ = curve_fit(lambda x, a: a * x, x, 2.0 * x, p0=[1.0], jac=lambda x, a: x)
    assert popt[0] == pytest.approx(2.0)


def test_agy_reused_optimizer_honours_new_jacobian_mode(monkeypatch):
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")

    def model(x, a, b):
        return a * x + b

    x = np.linspace(0, 1, 10)
    y = 2 * x + 1
    ls = LeastSquares()
    ls.least_squares(model, [1.0, 1.0], xdata=x, ydata=y, jacobian_mode="fwd")
    old = ls.jac_none
    ls.least_squares(model, [1.0, 1.0], xdata=x, ydata=y, jacobian_mode="rev")
    assert ls.jac_none is not old


def test_p02_nested_fit_in_callback_keeps_outer_history(monkeypatch, tmp_path):
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", "1")
    monkeypatch.setenv("NLSQ_LOG_DIR", str(tmp_path))
    seen = {}

    def outer_cb(*args, **kwargs):
        if not seen:
            seen["x"] = 1
            LeastSquares().least_squares(lambda p: p - 40, [1.0])

    LeastSquares().least_squares(lambda p: p - 10, [1.0], callback=outer_cb)
    files = sorted(tmp_path.glob("*.npz"))
    assert files, "iteration history was not saved"
    firsts = [np.load(f)["cost"][0] for f in files if "cost" in np.load(f).files]
    # outer fit starts at cost 0.5 * 9**2 = 40.5, inner at 0.5 * 39**2 = 760.5
    assert any(c == pytest.approx(40.5) for c in firsts)


def test_p02_huber_grad_at_zero_is_finite():
    loss = LossFunctionsJIT().get_loss_function("huber")
    g = jax.grad(lambda f: loss(f, 1.0, cost_only=True))(0.0)
    assert float(g) == 0.0


def test_p02_arctan_rho2_does_not_overflow():
    rho = LossFunctionsJIT().get_loss_function("arctan")(jnp.array([1e40]), 1.0)
    assert float(rho[2, 0]) == pytest.approx(-2e-240, rel=1e-12)


def test_p02_finite_difference_jacobian_with_large_ydata_offset():
    J = SparseJacobianComputer().compute_sparse_jacobian(
        None,
        np.array([1.0]),
        np.array([1.0, 2.0]),
        np.array([1e16, 1e16]),
        func=lambda x, a: a * x,
    )
    np.testing.assert_allclose(J.toarray().ravel(), [1.0, 2.0], rtol=1e-6)


def test_p02_short_data_mask_rejected_in_sparse_jacobian():
    with pytest.raises(ValueError, match="data_mask"):
        SparseJacobianComputer().compute_sparse_jacobian(
            lambda x, xd, yd, m, t: np.ones((len(m), 1)),
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            data_mask=np.array([True]),
        )


def test_p02_square_coordinate_array_is_read_points_last():
    def func(x, a):
        return a * (x[0] + x[1] + x[2])

    sparsity, _ = detect_jacobian_sparsity(
        func, np.array([1.0]), np.ones((3, 3)), n_samples=2
    )
    assert sparsity == 0.0


def test_p02_scalar_output_single_point_and_empty_dims():
    sjc = SparseJacobianComputer()
    _, sparsity = sjc.detect_sparsity_pattern(lambda x, a: a * x[0], [1.0], [5.0], 1)
    assert sparsity == 0.0
    assert sjc.estimate_memory_usage(0, 5)["savings_percent"] == 0.0
    _, info = detect_jacobian_sparsity(lambda x: x, np.array([]), np.array([1.0, 2.0]))
    assert info["max_nnz_per_col"] == 0


@pytest.mark.serial
def test_p02_svd_gpu_fallback_keeps_factors_on_input_device():
    script = textwrap.dedent(
        """
        import jax, jax.numpy as jnp, numpy as np
        jax.config.update("jax_enable_x64", True)
        import nlsq.core.trf_jit as T
        d1 = jax.devices()[1]
        rng = np.random.default_rng(0)
        J = jax.device_put(jnp.asarray(rng.normal(size=(6, 2))), d1)
        f = jax.device_put(jnp.asarray(rng.normal(size=6)), d1)
        d = jax.device_put(jnp.ones(2), d1)
        def boom(*a, **k):
            raise RuntimeError("cuSolver internal error")
        real = T.compute_svd_with_fallback
        # the production fallback commits its factors to cpu:0
        T.compute_svd_with_fallback = lambda A, **k: jax.device_put(
            real(A, **k), jax.devices()[0]
        )
        T._svd_no_bounds_jit = boom
        T.device_policy.use_host_svd_for = lambda J: False
        T._svd_no_bounds(J, d, f)
        J_diag = jax.device_put(jnp.ones((2, 2)), d1)
        f_zeros = jax.device_put(jnp.zeros(2), d1)
        T._svd_bounds_jit = boom
        T._svd_bounds(f, J, d, J_diag, f_zeros)
        print("OK")
        """
    )
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-3000:]


def test_wave3_sampled_sparsity_keeps_points_last_layout():
    from nlsq.core.sparse_jacobian import detect_sparsity_at_p0

    ratio, _ = detect_sparsity_at_p0(
        lambda x, a: a * x[3],
        np.array([1.0]),
        np.ones((4, 6)),
        n_residuals=6,
        sample_size=2,
    )
    assert ratio == 0.0


def test_wave3b_default_loss_accepts_overflowing_initial_cost():
    x = np.linspace(0, 1, 20)
    popt, _ = curve_fit(lambda x, a, b: a * x + b, x, 2 * x + 1, p0=[1e160, 1.0])
    assert popt == pytest.approx([2.0, 1.0], rel=1e-6)

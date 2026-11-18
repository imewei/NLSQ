"""Regression tests for core batch A review (2026-11-15, round 1)."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import OptimizeWarning, curve_fit
from nlsq.core import device_policy
from nlsq.core._option_checks import validate_tolerance
from nlsq.core.covariance import pcov_from_jacobian, pcov_from_normal
from nlsq.core.functions import estimate_p0_power_law
from nlsq.core.loss_functions import LossFunctionsJIT
from nlsq.core.problem import PrepareContext, pad_fit_data, prepare_problem

jax.config.update("jax_enable_x64", True)


def _linear(x, a, b):
    return a * x + b


# --- covariance ---------------------------------------------------------------


def _rank2_of_3():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(30, 2))
    return np.column_stack([J, J[:, 0]])  # third column duplicates the first


@pytest.mark.parametrize("cond_warn", [True, False])
def test_pcov_from_normal_warns_on_rank_deficiency(cond_warn):
    J = _rank2_of_3()
    with pytest.warns(OptimizeWarning, match=r"rank-deficient \(rank 2 < 3"):
        pcov, fired = pcov_from_normal(J.T @ J, 1.0, 30, 3, False, cond_warn=cond_warn)
    assert not fired
    assert np.isfinite(pcov).all()


def test_pcov_from_normal_full_rank_has_no_rank_warning():
    J = np.random.default_rng(1).normal(size=(30, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pcov_from_normal(J.T @ J, 1.0, 30, 3, False)


def test_pcov_from_jacobian_negative_cost_is_inf_not_negative_variance():
    J = jnp.asarray(np.random.default_rng(2).normal(size=(30, 2)))
    svd = lambda A: tuple(np.linalg.svd(np.asarray(A), full_matrices=False)[1:])
    with pytest.warns(OptimizeWarning):
        pcov, warn_cov = pcov_from_jacobian(J, -1.0, 30, 2, False, svd=svd)
    assert warn_cov
    assert np.isposinf(pcov).all()


def test_pcov_from_normal_negative_cost_is_inf():
    H = np.diag([2.0, 3.0])
    with pytest.warns(OptimizeWarning):
        pcov, fired = pcov_from_normal(H, -1.0, 10, 2, False)
    assert fired
    assert np.isposinf(pcov).all()


# --- problem.py ---------------------------------------------------------------


@pytest.mark.parametrize("p0", [np.ones((2, 2)), [[1.0, 2.0]]])
def test_multidimensional_p0_raises_clear_valueerror(p0):
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="at most 1 dimension"):
        curve_fit(_linear, x, 2 * x + 1, p0=p0)


def test_pad_fit_data_points_first_pads_axis0():
    x = np.ones((10, 2))
    y = np.ones(10)
    xp, yp = pad_fit_data(x, y, 2, 5, points_axis_last=False)
    assert xp.shape == (15, 2)
    assert yp.shape == (15,)
    xp, yp = pad_fit_data(x.T, y, 2, 5)  # default: points on the last axis
    assert xp.shape == (2, 15)


def test_prepare_problem_points_first_x_is_padded_to_flength():
    def f(x, a, b):
        return a * x[:, 0] + b * x[:, 1]

    x = np.random.default_rng(0).normal(size=(10, 2))
    prep = prepare_problem(
        f,
        x,
        x[:, 0] + 2 * x[:, 1],
        p0=np.ones(2),
        ctx=PrepareContext(flength=15, x_points_axis_last=False),
    )
    assert prep.x.shape == (15, 2)
    assert prep.y.shape == (15,)


def test_oversized_data_logs_fixed_length_debug_message():
    messages = []

    class _Log:
        def debug(self, msg, **kw):
            messages.append(msg)

        def __getattr__(self, name):
            return lambda *a, **k: None

    x = np.linspace(0, 1, 20)
    prepare_problem(
        _linear,
        x,
        2 * x + 1,
        p0=np.ones(2),
        ctx=PrepareContext(flength=10, logger=_Log()),
    )
    assert any("exceeds fixed length" in m for m in messages)


def test_prepare_problem_docstring_lists_only_injected_defaults():
    doc = prepare_problem.__doc__
    assert "x_scale" in doc
    assert "``ftol``/``xtol``/``gtol``/``max_nfev``/" not in doc


# --- loss functions -----------------------------------------------------------


@pytest.mark.parametrize("name", ["huber", "soft_l1", "cauchy", "arctan"])
def test_loss_cost_only_without_data_mask_matches_all_true_mask(name):
    fn = LossFunctionsJIT().get_loss_function(name)
    f = jnp.asarray([0.1, 2.0, -3.0])
    mask = jnp.ones(3, dtype=bool)
    assert fn(f, 1.5, cost_only=True) == pytest.approx(
        float(fn(f, 1.5, mask, cost_only=True))
    )


@pytest.mark.parametrize("name", ["huber", "soft_l1", "cauchy", "arctan"])
def test_loss_cost_only_scalar_input(name):
    fn = LossFunctionsJIT().get_loss_function(name)
    vec = fn(jnp.asarray([2.0]), 1.0, jnp.ones(1, dtype=bool), cost_only=True)
    assert float(fn(jnp.asarray(2.0), 1.0, cost_only=True)) == pytest.approx(float(vec))


# --- device policy ------------------------------------------------------------


def test_use_host_svd_for_inspects_array_platform_not_default_backend(monkeypatch):
    J = jnp.ones((10, 3))  # lives on CPU in this test run
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert not device_policy.use_host_svd_for(J)


def test_use_host_svd_for_uses_array_platform(monkeypatch):
    J = jnp.ones((10, 3))

    class _Dev:
        platform = "gpu"

    monkeypatch.setattr(type(J), "devices", lambda self: {_Dev()}, raising=False)
    assert device_policy.use_host_svd_for(J)


@pytest.mark.parametrize("shape", [(), (5,), (4, 3, 2)])
def test_use_host_for_small_svd_tolerates_non_2d_shapes(shape):
    assert isinstance(device_policy.use_host_for_small_svd(shape, "gpu"), bool)


# --- functions.py -------------------------------------------------------------


def test_power_law_p0_overflow_falls_back():
    x = np.array([10.0, 20.0, 50.0, 100.0])
    y = 10.0 ** (400.0 - 200.0 * np.log10(x))  # a = 1e400 overflows exp
    p0 = estimate_p0_power_law(x, y)
    assert np.isfinite(p0).all()
    assert p0 == [1.0, 1.0]


def test_functions_docstring_see_also_paths_exist():
    import importlib

    from nlsq.core import functions

    assert "nlsq.parameter_estimation :" not in functions.__doc__
    importlib.import_module("nlsq.precision.parameter_estimation")
    assert "L + b" in functions.sigmoid.__doc__


# --- _option_checks -----------------------------------------------------------


@pytest.mark.parametrize("bad", [-1.0, -1e-12])
def test_validate_tolerance_negative_warns_like_scipy(bad):
    from scipy.optimize import least_squares

    with warnings.catch_warnings(record=True) as sp:
        warnings.simplefilter("always")
        least_squares(lambda x: x - 1, [0.0], ftol=bad)
    assert any("ftol" in str(w.message) for w in sp)  # SciPy warns, does not raise
    with pytest.warns(OptimizeWarning, match="ftol"):
        assert validate_tolerance("ftol", bad) == bad


# --- round 2 (codex) ----------------------------------------------------------


def _np_svd(A):
    _, s, VT = np.linalg.svd(np.asarray(A), full_matrices=False)
    return s, VT


@pytest.mark.parametrize("absolute_sigma", [True, False])
def test_pcov_from_jacobian_rank_deficient_is_moore_penrose_in_original_coords(
    absolute_sigma,
):
    J = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    cost = 0.7
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        pcov, warn_cov = pcov_from_jacobian(
            jnp.asarray(J), cost, 3, 2, absolute_sigma, svd=_np_svd
        )
    assert not warn_cov
    expect = np.linalg.pinv(J.T @ J)
    if not absolute_sigma:
        expect = expect * (2 * cost / (3 - 2))
    np.testing.assert_allclose(pcov, expect, rtol=1e-10)
    np.testing.assert_allclose(pcov @ np.array([-2.0, 1.0]), 0.0, atol=1e-12)


def test_pcov_from_jacobian_full_rank_equilibrated_path_unchanged():
    # full rank: svd is called exactly once, on the equilibrated J
    J = np.random.default_rng(3).normal(size=(20, 3)) * [1.0, 1e4, 1e-3]
    calls = []

    def svd(A):
        calls.append(np.asarray(A))
        return _np_svd(A)

    pcov, _ = pcov_from_jacobian(jnp.asarray(J), 1.0, 20, 3, True, svd=svd)
    assert len(calls) == 1
    np.testing.assert_allclose(pcov, np.linalg.inv(J.T @ J), rtol=1e-8)


@pytest.mark.parametrize("points_last", [True, False])
@pytest.mark.parametrize("trailing", [(), (2,), (2, 3)])
def test_pad_fit_data_any_number_of_feature_axes(points_last, trailing):
    m, extra = 5, 2
    if not trailing:
        x = np.arange(m, dtype=float)
    elif points_last:
        x = np.ones((*trailing, m))
    else:
        x = np.ones((m, *trailing))
    y = np.arange(m, dtype=float)
    xp, yp = pad_fit_data(x, y, x.ndim, extra, points_axis_last=points_last)
    axis = -1 if (x.ndim > 1 and points_last) else 0
    expect = list(x.shape)
    expect[axis] += extra
    assert xp.shape == tuple(expect)
    assert yp.shape == (m + extra,)
    # original data preserved, pad values are the module's EPS
    np.testing.assert_array_equal(np.take(xp, range(m), axis=axis), x)
    pad = np.take(xp, range(m, m + extra), axis=axis)
    assert np.all(pad == pad.flat[0]) and pad.flat[0] > 0


def test_prepare_problem_pads_3d_points_first_x():
    prep = prepare_problem(
        lambda x, a: a * x[:, 0, 0],
        np.ones((5, 2, 3)),
        np.arange(5.0),
        p0=[1.0],
        ctx=PrepareContext(flength=7, x_points_axis_last=False),
    )
    assert prep.x.shape == (7, 2, 3)


def test_prepare_problem_pads_3d_points_last_x():
    prep = prepare_problem(
        lambda x, a: a * x[0, 0, :],
        np.ones((2, 3, 5)),
        np.arange(5.0),
        p0=[1.0],
        ctx=PrepareContext(flength=7),
    )
    assert prep.x.shape == (2, 3, 7)


def _omit_inputs(sigma):
    x = np.array([0.0, np.nan, 2.0])
    y = np.array([1.0, 9.0, 5.0])
    return x, y, np.asarray(sigma, dtype=float)


def test_nan_omit_ignores_zero_sigma_on_omitted_row():
    x, y, sigma = _omit_inputs([1.0, 0.0, 1.0])
    prep = prepare_problem(_linear, x, y, p0=[2.0, 1.0], sigma=sigma, nan_policy="omit")
    assert prep.m_real == 2
    np.testing.assert_array_equal(np.asarray(prep.sigma)[:2], [1.0, 1.0])


def test_nan_omit_still_rejects_zero_sigma_on_retained_row():
    x, y, sigma = _omit_inputs([1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="sigma"):
        prepare_problem(_linear, x, y, p0=[2.0, 1.0], sigma=sigma, nan_policy="omit")


def test_nan_omit_with_data_mask_validates_only_rows_used_by_both():
    x, y, sigma = _omit_inputs([1.0, 0.0, 0.0])
    # row 1 is omitted (NaN x), row 2 is masked out: only row 0 is validated
    prep = prepare_problem(
        _linear,
        x,
        y,
        p0=[2.0, 1.0],
        sigma=sigma,
        nan_policy="omit",
        data_mask=np.array([True, True, False]),
    )
    assert prep.m_real == 2


@pytest.mark.parametrize("sigma_kind", ["1d", "cov"])
def test_nan_omit_stability_ignores_bad_sigma_on_omitted_row(sigma_kind):
    x, y, sigma = _omit_inputs([1.0, 0.0, 1.0])
    if sigma_kind == "cov":
        sigma = np.diag([1.0, 0.0, 1.0])
    ctx = PrepareContext(enable_stability=True)
    prep = prepare_problem(
        _linear, x, y, p0=[2.0, 1.0], sigma=sigma, nan_policy="omit", ctx=ctx
    )
    assert prep.m_real == 2


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nan_omit_stability_still_rejects_bad_sigma_on_retained_row(bad):
    x, y, sigma = _omit_inputs([1.0, 1.0, bad])
    with pytest.raises(ValueError, match="sigma"):
        prepare_problem(
            _linear,
            x,
            y,
            p0=[2.0, 1.0],
            sigma=sigma,
            nan_policy="omit",
            ctx=PrepareContext(enable_stability=True),
        )


# --- round 3 (codex) ----------------------------------------------------------


def _rank_def_pcov(J, n_eff):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pcov, warn_cov = pcov_from_jacobian(
            jnp.asarray(J), 1.0, n_eff, J.shape[1], True, svd=_np_svd
        )
    assert any(issubclass(w.category, OptimizeWarning) for w in caught)
    return pcov, warn_cov, [str(w.message) for w in caught]


def test_pcov_from_jacobian_rank_deficient_tiny_orthogonal_column_is_warned_failure():
    # cols 0 and 2 identical (rank 2 of 3), col 1 orthogonal and 1e20 smaller: the exact
    # variance of col 1 is 2.5e19, but roundoff (~1e-17) in the equilibrated singular
    # vectors times the 1e10 column scale (1e-7) swamps the 2e-10 column, so the
    # original-coordinate pseudo-inverse cannot be resolved: all-inf + warning, never
    # a finite wrong matrix (a raw SVD of J returned pcov[1,1] ~ 3).
    u = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([1.0, -1.0, -1.0, 1.0])
    J = np.column_stack([1e10 * u, 1e-10 * v, 1e10 * u])
    pcov, warn_cov, msgs = _rank_def_pcov(J, 4)
    assert warn_cov
    assert np.isposinf(pcov).all()
    assert any("original parameter coordinates" in m for m in msgs)


@pytest.mark.parametrize("q", [1e-12, 1e-16])
def test_pcov_from_jacobian_rank_deficient_small_correlation_never_finite_wrong(q):
    a, b = 1e-5, 1e-20
    J = np.array(
        [[a, q * b, a], [0.0, b, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )  # rank 2, null vector [1, 0, -1]
    pcov, warn_cov, _ = _rank_def_pcov(J, 4)
    if warn_cov:
        assert np.isposinf(pcov).all()
        return
    np.testing.assert_allclose(pcov[0, 0], (1 + q * q) / (4 * a * a), rtol=1e-6)
    np.testing.assert_allclose(pcov[0, 1], -q / (2 * a * b), rtol=1e-6)
    np.testing.assert_allclose(
        pcov @ np.array([1.0, 0.0, -1.0]), 0.0, atol=1e-6 * pcov[1, 1]
    )


@pytest.mark.parametrize("scales", [[1, 1, 1], [1e3, 1, 1e-3], [1e6, 1e-2, 1]])
def test_pcov_from_jacobian_rank_deficient_moderate_scales_match_pinv(scales):
    rng = np.random.default_rng(0)
    B = rng.normal(size=(50, 3)) * scales
    J = np.column_stack([B, 2.0 * B[:, 0] + 0.5 * B[:, 1]])
    pcov, warn_cov, _ = _rank_def_pcov(J, 50)
    assert not warn_cov
    Pj = np.linalg.pinv(J, rcond=1e-10)
    ref = Pj @ Pj.T
    d = np.sqrt(np.outer(np.diag(ref), np.diag(ref)))
    assert np.max(np.abs(pcov - ref) / d) < 1e-6


def test_pcov_from_jacobian_rank_deficient_calls_svd_hook_only_on_equilibrated_j():
    J = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    calls = []

    def svd(A):
        calls.append(np.asarray(A))
        return _np_svd(A)

    with pytest.warns(OptimizeWarning):
        pcov_from_jacobian(jnp.asarray(J), 1.0, 3, 2, True, svd=svd)
    assert len(calls) == 1
    np.testing.assert_allclose(
        np.linalg.norm(calls[0], axis=0), 1.0
    )  # equilibrated, never the raw J


def _mask_inputs(sigma_kind, sigma_bad):
    x = np.arange(4.0)
    y = 2 * x + 1
    sigma = np.ones(4)
    sigma[1] = sigma_bad
    if sigma_kind == "cov":
        sigma = np.diag(sigma)
    return x, y, sigma


@pytest.mark.parametrize("sigma_kind", ["1d", "cov"])
def test_data_mask_stability_ignores_bad_sigma_on_masked_row(sigma_kind):
    x, y, sigma = _mask_inputs(sigma_kind, 0.0)
    prep = prepare_problem(
        _linear,
        x,
        y,
        p0=[2.0, 1.0],
        sigma=sigma,
        data_mask=np.array([True, False, True, True]),
        ctx=PrepareContext(enable_stability=True),
    )
    assert prep.m_real == 4


@pytest.mark.parametrize("sigma_kind", ["1d", "cov"])
def test_data_mask_plus_omit_stability_neutralizes_union(sigma_kind):
    x = np.array([0.0, np.nan, 2.0, 3.0])
    y = np.array([1.0, 9.0, 5.0, 7.0])
    sigma = np.ones(4)
    sigma[1] = 0.0  # omitted
    sigma[3] = 0.0  # masked out
    if sigma_kind == "cov":
        sigma = np.diag(sigma)
    prep = prepare_problem(
        _linear,
        x,
        y,
        p0=[2.0, 1.0],
        sigma=sigma,
        nan_policy="omit",
        data_mask=np.array([True, True, True, False]),
        ctx=PrepareContext(enable_stability=True),
    )
    assert prep.m_real == 3


def test_data_mask_stability_still_rejects_zero_sigma_on_retained_row():
    x, y, sigma = _mask_inputs("1d", 1.0)
    sigma[2] = 0.0
    with pytest.raises(ValueError, match="sigma"):
        prepare_problem(
            _linear,
            x,
            y,
            p0=[2.0, 1.0],
            sigma=sigma,
            data_mask=np.array([True, False, True, True]),
            ctx=PrepareContext(enable_stability=True),
        )


# W2 round 4: non-finite sigma entries on unused rows are not an error


def _nan_sigma(kind, bad_rows, value=np.nan):
    sigma = np.ones(4) if kind == "1d" else np.eye(4)
    for r in bad_rows:
        if kind == "1d":
            sigma[r] = value
        else:
            sigma[r, :] = value
            sigma[:, r] = value
    return sigma


_MASK = np.array([True, False, True, True])


def _unused_case(mode, kind, bad_row):
    """(x, y, kwargs) with the bad sigma on ``bad_row``."""
    x = np.arange(4.0)
    y = 2 * x + 1
    kw = {"p0": [2.0, 1.0], "sigma": _nan_sigma(kind, [bad_row])}
    if mode in ("mask", "both"):
        kw["data_mask"] = _MASK
    if mode in ("omit", "both"):
        x = x.copy()
        x[2] = np.nan  # row 2 is omitted
        kw["nan_policy"] = "omit"
    return x, y, kw


@pytest.mark.parametrize("stability", [False, True])
@pytest.mark.parametrize("kind", ["1d", "cov"])
@pytest.mark.parametrize(
    ("mode", "bad_row"),
    [("mask", 1), ("omit", 2), ("both", 1), ("both", 2)],
)
def test_nonfinite_sigma_on_unused_row_is_accepted(mode, kind, bad_row, stability):
    x, y, kw = _unused_case(mode, kind, bad_row)
    prep = prepare_problem(
        _linear, x, y, ctx=PrepareContext(enable_stability=stability), **kw
    )
    omitted = set()
    if mode in ("omit", "both"):
        omitted.add(2)  # NaN x
        if kind == "1d":
            omitted.add(bad_row)  # omit also drops a non-finite 1-D sigma row
    assert prep.m_real == 4 - len(omitted)


@pytest.mark.parametrize("stability", [False, True])
@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize(
    ("mode", "kind"),
    # a non-finite 1-D sigma row is omitted by nan_policy='omit', so only the
    # covariance-matrix form is retained (and rejected) in the omit modes
    [
        ("mask", "1d"),
        ("mask", "cov"),
        ("omit", "cov"),
        ("both", "cov"),
    ],
)
def test_nonfinite_sigma_on_retained_row_still_raises(mode, kind, value, stability):
    x, y, kw = _unused_case(mode, kind, 0)  # row 0 is used in every mode
    kw["sigma"] = _nan_sigma(kind, [0], value)
    with pytest.raises(ValueError, match=r"sigma|Sigma"):
        prepare_problem(
            _linear, x, y, ctx=PrepareContext(enable_stability=stability), **kw
        )


# --- round 5 (codex) ----------------------------------------------------------


def _warned_inf_or_close(pcov, warn_cov, ref, rtol):
    if warn_cov:
        assert np.isposinf(pcov).all()
        return "warned"
    d = np.sqrt(np.outer(np.abs(np.diag(ref)), np.abs(np.diag(ref))))
    d = np.where(d > 0, d, 1.0)
    np.testing.assert_array_less(np.abs(pcov - ref) / d, rtol)
    return "finite"


def test_pcov_from_jacobian_rank_deficient_hidden_cross_term_is_rejected():
    # columns 0 and 2 identical: exact null vector [1, 0, -1]; the exact
    # Moore-Penrose cross terms pcov[0,1] = pcov[2,1] = -5e-5 are not resolvable from
    # the rounded singular vectors (a finite -1.06e-4 / 5.5e-6 pair used to be returned).
    J = np.array(
        [[1.0, 1e-12, 1.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    pcov, warn_cov, msgs = _rank_def_pcov(J, 4)
    assert warn_cov
    assert np.isposinf(pcov).all()
    assert any("original parameter coordinates" in m for m in msgs)


@pytest.mark.parametrize("unit", [1e-15, 1.0, 1e15])
def test_pcov_from_jacobian_rank_deficient_zero_column_is_unit_free(unit):
    J = unit * np.array(
        [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0], [0.0, 0.0, 0.0]]
    )
    pcov, warn_cov, _ = _rank_def_pcov(J, 4)
    assert not warn_cov
    # J^T J = 14 u^2 [[1,1],[1,1]] (u = unit): pinv = [[1,1],[1,1]] / (4 * 14 u^2)
    expect = np.zeros((3, 3))
    expect[:2, :2] = 1.0 / (56.0 * unit**2)
    np.testing.assert_allclose(pcov, expect, rtol=1e-10)
    assert (pcov[2] == 0).all() and (pcov[:, 2] == 0).all()


def _exact_reference(Jr, T):
    """pinv(T^T Jr^T Jr T) = T^+ (Jr^T Jr)^-1 T^+T for full-row-rank T, in mpmath."""
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 80
    G = mpmath.matrix(Jr.T.tolist()) * mpmath.matrix(Jr.tolist())
    Ginv = G**-1
    Tm = mpmath.matrix(T.tolist())
    Tp = Tm.T * (Tm * Tm.T) ** -1
    X = Tp * Ginv * Tp.T
    return np.array([[float(X[i, j]) for j in range(X.cols)] for i in range(X.rows)])


def _random_rank_deficient(rng, log_lo, log_hi):
    """Float J whose rank deficiency is exact: duplicated/power-of-two copies of columns."""
    r = int(rng.integers(1, 4))
    p = r + int(rng.integers(1, 3))
    n = int(rng.integers(max(p, 4), 12))
    Jr = rng.normal(size=(n, r)) * 10.0 ** rng.uniform(log_lo, log_hi, size=r)
    T = np.zeros((r, p))
    T[:, :r] = np.eye(r)
    for k in range(r, p):
        T[int(rng.integers(0, r)), k] = 2.0 ** int(rng.integers(-3, 4))
    perm = rng.permutation(p)
    T = T[:, perm]
    return Jr @ T, Jr, T  # Jr @ T is exact (one nonzero power-of-two per added column)


@pytest.mark.parametrize(
    ("log_lo", "log_hi", "max_warned"),
    [(-1.5, 1.5, 0), (-3.0, 3.0, 10), (-8.0, 8.0, 200)],
)
def test_rank_deficient_pcov_is_warned_or_accurate_property(log_lo, log_hi, max_warned):
    # 200 seeded rank-deficient J (exact duplicate / power-of-two copies of columns,
    # log-uniform independent-column scales, random cost and n_eff): all-inf with a
    # warning, or within 1e-4 (scaled by sqrt(X_ii X_jj)) of an mpmath reference.
    # Measured over three seeds (600 problems each): 0 finite-wrong in every range;
    # warned 0 / ~1% / ~28% for +-1.5 / +-3 / +-8 decades (column-norm ratios
    # beyond ~1e6 cannot be resolved to 1e-4 in original coordinates).
    rng = np.random.default_rng(12345)
    counts = {"finite": 0, "warned": 0}
    for _ in range(200):
        J, Jr, T = _random_rank_deficient(rng, log_lo, log_hi)
        n_eff = J.shape[0]
        cost = float(rng.uniform(0.1, 5.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcov, warn_cov = pcov_from_jacobian(
                jnp.asarray(J), cost, n_eff + 3, J.shape[1], True, svd=_np_svd
            )
        ref = _exact_reference(Jr, T)
        counts[_warned_inf_or_close(pcov, warn_cov, ref, 1e-4)] += 1
    assert counts["warned"] <= max_warned, counts
    assert counts["finite"] > 0


# --- round 6 (codex) ----------------------------------------------------------


def test_pcov_from_jacobian_rank_deficient_nearly_dependent_retained_columns():
    # columns 0 and 2 identical; column 1 differs from them at ~1e-13: its retained
    # singular value is ~1e-13 of the leading one, so eps * s_1 / s_r ~ 1e-3 and the
    # pseudo-inverse is only good to ~1% (exact pcov[1,1] = 3.5377e25; 3.499e25 was
    # returned as if it were accurate).
    u = np.array(
        [1.3858854430909429, -1.685784678368757, -0.37775860666514965,
         -2.7284856868728458, -0.6463967359069595]
    )  # fmt: skip
    w = np.array(
        [1.3858854430910543, -1.6857846783688413, -0.3777586066652133,
         -2.7284856868728133, -0.6463967359068807]
    )  # fmt: skip
    J = np.column_stack((u, w, u))
    pcov, warn_cov, msgs = _rank_def_pcov(J, 5)
    assert warn_cov
    assert np.isposinf(pcov).all()
    assert any("original parameter coordinates" in m for m in msgs)


def test_pcov_from_jacobian_rank_deficient_huge_column_ratio_warns_not_overflows():
    J = np.array(
        [[1e100, 0.0, 1e100], [0.0, 1e-100, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    pcov, warn_cov, msgs = _rank_def_pcov(J, 4)
    assert warn_cov
    assert np.isposinf(pcov).all()
    assert any("original parameter coordinates" in m for m in msgs)


# --- round 7 (codex): float32 Jacobians ----------------------------------------


@pytest.mark.parametrize("absolute_sigma", [True, False])
def test_pcov_from_jacobian_float32_rank_one_is_exact_pinv(absolute_sigma):
    from nlsq.core.minpack import _covariance_svd

    J = np.array([[1, 2, 1], [2, 4, 2], [3, 6, 3], [4, 8, 4]], dtype=np.float32)
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        pcov, warn_cov = pcov_from_jacobian(
            jnp.asarray(J), 1.0, 4, 3, absolute_sigma, svd=_covariance_svd
        )
    assert not warn_cov
    J64 = J.astype(np.float64)  # float32 entries taken as exact
    ref = np.linalg.pinv(J64.T @ J64)
    if not absolute_sigma:
        ref = ref * (2.0 / (4 - 3))
    np.testing.assert_allclose(pcov, ref, rtol=1e-10)


def test_pcov_from_jacobian_float32_nearly_dependent_is_warned_or_accurate():
    from nlsq.core.minpack import _covariance_svd

    u = np.array([-0.7612727284431458, -1.4279347658157349], dtype=np.float32)
    w = np.array([-0.7612850666046143, -1.4279320240020752], dtype=np.float32)
    J = np.column_stack((u, w, u))
    pcov, warn_cov = pcov_from_jacobian(
        jnp.asarray(J), 1.0, 2, 3, True, svd=_covariance_svd
    )
    if warn_cov:
        assert np.isposinf(pcov).all()
        return
    J64 = J.astype(np.float64)
    ref = np.linalg.pinv(J64.T @ J64, rcond=1e-12)
    d = np.sqrt(np.outer(np.diag(ref), np.diag(ref)))
    assert np.max(np.abs(pcov - ref) / d) < 1e-4


def test_pcov_from_jacobian_float32_full_rank_matches_float64():
    from nlsq.core.minpack import _covariance_svd

    J = np.random.default_rng(5).normal(size=(30, 3)).astype(np.float32)
    got, warn_got = pcov_from_jacobian(
        jnp.asarray(J), 2.0, 30, 3, False, svd=_covariance_svd
    )
    J64 = J.astype(np.float64)
    ref = np.linalg.inv(J64.T @ J64) * (2 * 2.0 / 27)
    assert not warn_got
    np.testing.assert_allclose(got, ref, rtol=1e-9)


@pytest.fixture
def x64_off():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.serial
def test_pcov_from_jacobian_float64_not_narrowed_when_x64_off(x64_off):
    from nlsq.core.minpack import _covariance_svd

    J = np.array([[1, 1], [1, 1 + 1e-8], [1, 1 - 1e-8]], dtype=np.float64)
    pcov, warn_cov = pcov_from_jacobian(J, 1.0, 3, 2, True, svd=_covariance_svd)
    assert not warn_cov
    _, sv, vt = np.linalg.svd(J, full_matrices=False)
    ref = (vt.T / sv**2) @ vt
    np.testing.assert_allclose(np.diag(pcov), np.diag(ref), rtol=1e-6)
    assert pcov.dtype == np.float64


@pytest.mark.serial
def test_pcov_from_jacobian_huge_float64_not_overflowed_when_x64_off(x64_off):
    from nlsq.core.minpack import _covariance_svd

    J = np.array([[1e40, 0.0], [0.0, 2e40], [1e40, 1e40]])
    pcov, warn_cov = pcov_from_jacobian(J, 1.0, 3, 2, True, svd=_covariance_svd)
    assert not warn_cov
    np.testing.assert_allclose(pcov, np.linalg.inv(J.T @ J), rtol=1e-9)


@pytest.mark.serial
@pytest.mark.parametrize("absolute_sigma", [True, False])
def test_pcov_from_jacobian_x64_off_tiny_column_not_flushed(x64_off, absolute_sigma):
    from nlsq.core.minpack import _covariance_svd

    J = np.array([[1.0, 0.0], [0.0, 1e-50], [0.0, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pcov, warn_cov = pcov_from_jacobian(
            J, 1.0, 3, 2, absolute_sigma, svd=_covariance_svd
        )
    assert not warn_cov
    scale = 1.0 if absolute_sigma else 2.0 / (3 - 2)  # s_sq = 2 * cost / dof
    np.testing.assert_allclose(np.diag(pcov), [scale, scale * 1e100], rtol=1e-9)


@pytest.mark.serial
def test_pcov_from_jacobian_x64_off_huge_float64_has_no_overflow_warning(x64_off):
    from nlsq.core.minpack import _covariance_svd

    J = np.array([[1e40, 0.0], [0.0, 2e40], [1e40, 1e40]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pcov_from_jacobian(J, 1.0, 3, 2, True, svd=_covariance_svd)


@pytest.mark.serial
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_pcov_from_jacobian_x64_off_nonfinite_is_warned_all_inf(x64_off, bad):
    from nlsq.core.minpack import _covariance_svd

    J = np.array([[bad, 0.0], [0.0, 1.0], [0.0, 0.0]])
    pcov, warn_cov = pcov_from_jacobian(J, 1.0, 3, 2, True, svd=_covariance_svd)
    assert warn_cov
    assert np.isposinf(pcov).all()


@pytest.mark.parametrize("as_jax", [True, False])
def test_pcov_from_jacobian_float32_subnormal_survives_widening(as_jax):
    from nlsq.core.minpack import _covariance_svd

    t = np.nextafter(np.float32(0), np.float32(1))
    J = np.array([[1, 0], [0, t], [0, 0]], dtype=np.float32)
    pcov, warn_cov = pcov_from_jacobian(
        jnp.asarray(J) if as_jax else J, 1.0, 3, 2, True, svd=_covariance_svd
    )
    assert not warn_cov
    np.testing.assert_allclose(np.diag(pcov), [1.0, 1.0 / float(t) ** 2], rtol=1e-9)

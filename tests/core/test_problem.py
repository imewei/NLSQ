"""Unit tests for ``nlsq.core.problem.prepare_problem``."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import CurveFit, curve_fit
from nlsq.core.problem import PrepareContext, PreparedProblem, prepare_problem
from nlsq.utils.validators import InputValidator

N = 60
P0 = np.array([1.0, 1.0])
BOUNDS = (-np.inf, np.inf)


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n=N, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4.0, n)
    y = 2.0 * np.exp(-1.3 * x) + 0.01 * rng.standard_normal(n)
    return x, y


def _prepare(x, y, *, data_mask=None, nan_policy="raise", ctx=None, **kw):
    kwargs = {} if ctx is None else {"ctx": ctx}
    return prepare_problem(
        _model,
        x,
        y,
        P0,
        BOUNDS,
        None,
        data_mask,
        True,
        nan_policy,
        kwargs={},
        **kw,
        **kwargs,
    )


class TestPreparedProblemFields:
    def test_n_effective_counts_unmasked_rows(self):
        x, y = _data()
        mask = np.ones(N, dtype=bool)
        mask[::6] = False  # 10 real rows excluded
        prep = _prepare(x, y, data_mask=mask, ctx=PrepareContext(flength=100))
        assert prep.m_real == N
        assert len(prep.mask) == 100  # padded with False
        assert prep.n_effective == 50 == int(mask.sum())

    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _prepare(np.array([]), np.array([]))

    def test_omit_realigns_mask(self):
        x, y = _data()
        y_nan = y.copy()
        y_nan[[3, 10]] = np.nan
        keep = np.isfinite(y_nan)
        mask = np.ones(N, dtype=bool)
        mask[0] = False
        prep = _prepare(x, y_nan, data_mask=mask, nan_policy="omit")
        assert prep.m_real == N - 2
        np.testing.assert_array_equal(prep.mask, mask[keep])
        assert prep.n_effective == int(mask[keep].sum())
        assert not prep.none_mask

    def test_len_diff_is_padding_length(self):
        x, y = _data()
        prep = _prepare(x, y, ctx=PrepareContext(flength=100))
        assert prep.should_pad
        assert prep.len_diff == 100 - N
        assert prep.x.shape == prep.y.shape == (100,)
        assert prep.none_mask
        assert prep.n_effective == N  # padding rows are masked out


# case: (pad to flength=100, caller mask excluding 10 rows, omit NaN rows,
#        expected n_effective, expected data/mask length)
_NEFF_CASES = {
    "unpadded": (False, False, False, N, N),
    "padded": (True, False, False, N, 100),
    "user_mask": (False, True, False, 50, N),
    "omitted": (False, False, True, N - 2, N - 2),
    "padded_user_mask": (True, True, False, 50, 100),
}


class TestNEffectiveEquivalence:
    """``n_effective`` replaces ``ysize = int(np.sum(data_mask))`` downstream."""

    @pytest.mark.parametrize("case", list(_NEFF_CASES))
    def test_n_effective_equals_old_ysize(self, case):
        pad, user_mask, omit, expected_n, expected_len = _NEFF_CASES[case]
        x, y = _data()
        ctx = PrepareContext(flength=100) if pad else PrepareContext()
        mask = None
        if user_mask:
            mask = np.ones(N, dtype=bool)
            mask[::6] = False
        if omit:
            y = y.copy()
            y[[3, 10]] = np.nan
        prep = _prepare(
            x, y, data_mask=mask, nan_policy="omit" if omit else "raise", ctx=ctx
        )
        assert len(prep.x) == len(prep.y) == len(prep.mask) == expected_len
        assert prep.should_pad is pad
        assert prep.n_effective == int(np.sum(prep.mask))  # pre-refactor expression
        assert prep.n_effective == expected_n


class TestListInputs:
    """Plain list/tuple inputs that ``curve_fit`` accepted before the extraction."""

    @pytest.mark.parametrize("mask_type", [list, tuple])
    @pytest.mark.parametrize("flength", [None, 60])
    def test_list_mask_n_effective(self, mask_type, flength):
        x, y = _data(50)
        mask = mask_type([True] * 45 + [False] * 5)
        prep = _prepare(x, y, data_mask=mask, ctx=PrepareContext(flength=flength))
        assert prep.n_effective == 45
        assert not prep.none_mask

    def test_list_xy_and_sigma(self):
        x, y = _data(50)
        prep = prepare_problem(
            _model,
            list(x),
            list(y),
            P0,
            BOUNDS,
            [0.1] * 50,
            None,
            True,
            "raise",
            kwargs={},
        )
        assert prep.m_real == 50
        assert prep.n_effective == 50
        np.testing.assert_array_equal(prep.x, x)

    @pytest.mark.parametrize("mask_type", [list, tuple])
    @pytest.mark.parametrize("flength", [None, 60])
    def test_curve_fit_list_mask_matches_array_mask(self, mask_type, flength):
        x, y = _data(50)
        arr = np.array([True] * 45 + [False] * 5)
        kw = {"p0": [1.0, 1.0], "data_mask": arr}
        ref_popt, ref_pcov = CurveFit(flength=flength).curve_fit(_model, x, y, **kw)
        kw["data_mask"] = mask_type(arr.tolist())
        popt, pcov = CurveFit(flength=flength).curve_fit(_model, x, y, **kw)
        np.testing.assert_allclose(popt, ref_popt, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(pcov, ref_pcov, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("mask_type", [list, tuple])
    def test_public_curve_fit_list_mask_uses_sum_mask_dof(self, mask_type):
        x, y = _data(50)
        arr = np.array([True] * 45 + [False] * 5)
        p_ref, c_ref = curve_fit(_model, x, y, p0=[1.0, 1.0], data_mask=arr)
        popt, pcov = curve_fit(
            _model, x, y, p0=[1.0, 1.0], data_mask=mask_type(arr.tolist())
        )
        np.testing.assert_allclose(popt, p_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(pcov, c_ref, rtol=1e-8, atol=1e-10)
        # dof = sum(mask) - n_params: pcov = (J^T J)^-1 * sum(r^2) / (45 - 2)
        xm, ym = x[arr], y[arr]
        a, b = popt
        e = np.exp(-b * xm)
        jac = np.stack([e, -a * xm * e], axis=1)
        r = ym - a * e
        manual = np.linalg.inv(jac.T @ jac) * (r @ r) / (45 - 2)
        np.testing.assert_allclose(pcov, manual, rtol=1e-6)


class TestTypes:
    def test_prepared_problem_is_frozen(self):
        x, y = _data()
        prep = _prepare(x, y)
        assert isinstance(prep, PreparedProblem)
        with pytest.raises(dataclasses.FrozenInstanceError):
            prep.n = 5  # type: ignore[misc]

    def test_prepare_context_defaults(self):
        ctx = PrepareContext()
        assert ctx.flength is None
        assert ctx.use_dynamic_sizing is False
        assert ctx.enable_stability is False
        assert isinstance(ctx.validator, InputValidator)
        assert ctx.validator.fast_mode is True
        assert ctx.logger is not None  # the shared "curve_fit" logger
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.flength = 3  # type: ignore[misc]


class TestCurveFitValidatorWiring:
    def test_falsy_custom_validator_is_used(self):
        """An explicitly set validator must be used even if it is falsy."""
        calls = []

        class FalsyValidator(InputValidator):
            def __bool__(self):
                return False

            def validate_security_constraints(self, *args, **kwargs):
                calls.append(1)
                return super().validate_security_constraints(*args, **kwargs)

        x, y = _data()
        cf = CurveFit()
        cf.validator = FalsyValidator(fast_mode=True)
        cf.curve_fit(_model, x, y, p0=P0)
        assert calls, "CurveFit ignored its falsy custom validator"


class TestMaskNormalization:
    def test_list_mask_becomes_ndarray(self):
        x, y = _data()
        prep = _prepare(x, y, data_mask=[True] * (N - 1) + [False])
        assert isinstance(prep.mask, np.ndarray)
        assert prep.n_effective == N - 1


def _exp_decay(x, a, b):
    return a * jnp.exp(-b * x)


def _decay_data(n=50):
    x = np.linspace(0, 4, n)
    y = 2.5 * np.exp(-1.3 * x) + 0.05 * np.random.default_rng(0).normal(size=n)
    return x, y


class TestCurveFitApiRegressions:
    """Regression tests for the curve_fit wrapper/prepare_problem review findings."""

    def test_auto_p0_with_nan_omit(self):
        # API2: p0='auto' was estimated on NaN data before NaN removal.
        from nlsq.core.functions import gaussian

        x = np.linspace(0, 10, 200)
        y = 10 * np.exp(-((x - 5) ** 2) / 2)
        y[7] = np.nan
        r = curve_fit(gaussian, x, y, p0="auto", nan_policy="omit")
        np.testing.assert_allclose(r.popt, [10.0, 5.0, 1.0], rtol=1e-3)

    @pytest.mark.parametrize(
        "kw",
        [
            {"method": "cmaes", "bounds": (0, 10)},
            {"multistart": True, "n_starts": 3, "bounds": ([0, 0], [10, 10])},
            {"stability": "check"},
        ],
    )
    def test_auto_p0_wrapper_routes(self, kw):
        # API3: the raw 'auto' sentinel crashed cmaes / multistart / stability.
        x, y = _decay_data()
        r = curve_fit(_exp_decay, x, y, p0="auto", **kw)
        np.testing.assert_allclose(r.popt, [2.5, 1.3], atol=0.15)

    def test_return_eval_strips_padding_with_data_mask(self):
        # API5: padded rows leaked when a user data_mask was given.
        x, y = _decay_data()
        mask = np.ones(50, bool)
        mask[::5] = False
        out = CurveFit(flength=64).curve_fit(
            _exp_decay, x, y, p0=[1, 1], data_mask=mask, return_eval=True
        )
        assert len(out[2]) == 50

    def test_invalid_nan_policy_raises(self):
        # API6
        x, y = _decay_data()
        with pytest.raises(ValueError, match="nan_policy"):
            curve_fit(_exp_decay, x, y, p0=[1, 1], nan_policy="bogus")

    def test_nan_policy_none_is_scipy_default_raise(self):
        # scipy's default is nan_policy=None (== 'raise'); must stay accepted.
        x, y = _decay_data()
        popt, _ = curve_fit(_exp_decay, x, y, p0=[1, 1], nan_policy=None)
        ref, _ = curve_fit(_exp_decay, x, y, p0=[1, 1], nan_policy="raise")
        np.testing.assert_allclose(popt, ref)
        y_nan = y.copy()
        y_nan[3] = np.nan
        with pytest.raises(ValueError):
            curve_fit(_exp_decay, x, y_nan, p0=[1, 1], nan_policy=None)

    def test_cmaes_requires_bounds(self):
        # API7
        x, y = _decay_data()
        with pytest.raises(ValueError, match="requires bounds"):
            curve_fit(_exp_decay, x, y, p0=[1, 1], method="cmaes")

    def test_zero_sigma_rejected(self):
        # UCL4: zero sigma failed with a misleading "Residuals are not finite".
        x, y = _decay_data()
        sigma = np.full(50, 0.1)
        sigma[3] = 0.0
        with pytest.raises(ValueError, match="sigma values must be non-zero"):
            curve_fit(_exp_decay, x, y, p0=[1, 1], sigma=sigma)

    @pytest.mark.parametrize("absolute_sigma", [False, True])
    def test_negative_sigma_matches_scipy(self, absolute_sigma):
        # Only the sign of a residual weight flips, so SciPy (and HEAD) accept it.
        import scipy.optimize as so

        x, y = _decay_data()
        sigma = 0.05 + 0.1 * np.random.default_rng(1).random(50)
        sigma[3] = -sigma[3]
        ref = so.curve_fit(
            lambda x, a, b: a * np.exp(-b * x),
            x,
            y,
            p0=[1, 1],
            sigma=sigma,
            absolute_sigma=absolute_sigma,
        )
        r = curve_fit(
            _exp_decay, x, y, p0=[1, 1], sigma=sigma, absolute_sigma=absolute_sigma
        )
        # NLSQ vs SciPy differ by ~4e-6 in pcov even for all-positive sigma
        # (termination tolerance), so compare to SciPy at 1e-4 ...
        np.testing.assert_allclose(r.popt, ref[0], rtol=1e-6)
        np.testing.assert_allclose(r.pcov, ref[1], rtol=1e-4)
        # ... and pin the sign-invariance tightly against NLSQ's own positive run.
        r_pos = curve_fit(
            _exp_decay,
            x,
            y,
            p0=[1, 1],
            sigma=np.abs(sigma),
            absolute_sigma=absolute_sigma,
        )
        np.testing.assert_allclose(r.popt, r_pos.popt, rtol=1e-9)
        np.testing.assert_allclose(r.pcov, r_pos.pcov, rtol=1e-9)

    def test_negative_sigma_rejected_in_strict_mode(self):
        sigma = np.array([0.1, -0.1, 0.1])
        errors, _ = InputValidator()._validate_sigma(sigma, np.ones(3))
        assert any("must be positive" in e for e in errors)
        errors, _ = InputValidator()._validate_sigma(sigma, np.ones(3), strict=False)
        assert errors == []

    def test_scalar_and_covariance_sigma_still_valid(self):
        x, y = _decay_data()
        curve_fit(_exp_decay, x, y, p0=[1, 1], sigma=0.1)
        cov = np.diag(np.full(50, 0.01))
        cov[0, 1] = cov[1, 0] = -0.001  # negative off-diagonal is legitimate
        curve_fit(_exp_decay, x, y, p0=[1, 1], sigma=cov)


class TestExplicitTrSolverAndKwOnlyParams:
    def test_explicit_tr_solver_is_respected(self, monkeypatch):
        # m*n >= 10000 makes solver='auto' pick lsmr; an explicit tr_solver must win.
        from nlsq.core.least_squares import LeastSquares

        seen = []
        orig = LeastSquares.least_squares

        def spy(self, *a, **kw):
            seen.append(kw.get("tr_solver"))
            return orig(self, *a, **kw)

        monkeypatch.setattr(LeastSquares, "least_squares", spy)
        x = np.linspace(0, 4, 5001)
        y = 2.5 * np.exp(-1.3 * x) + 0.5
        f = lambda x, a, b, c: a * jnp.exp(-b * x) + c
        curve_fit(f, x, y, p0=[1, 1, 0], tr_solver="exact")
        assert seen and set(seen) == {"exact"}
        seen.clear()
        curve_fit(f, x, y, p0=[1, 1, 0])
        assert set(seen) == {"lsmr"}  # 'auto' selection unchanged

    @pytest.mark.parametrize("p0", [None, "auto"])
    def test_keyword_only_params_not_counted(self, p0):
        def f(x, a, b, *, c=0.0):
            return a * x + b + c

        x = np.linspace(0, 1, 50)
        r = curve_fit(f, x, 2 * x + 1, p0=p0)
        np.testing.assert_allclose(r.popt, [2.0, 1.0], atol=1e-6)

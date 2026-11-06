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


class TestEstimateP0OrderIndependent:
    """FWHM-based p0 estimates must not depend on the ordering of x."""

    @pytest.mark.parametrize("name", ["gaussian", "lorentzian"])
    @pytest.mark.parametrize("order", ["reversed", "shuffled"])
    def test_peak_p0_same_for_any_x_order(self, name, order):
        from nlsq.core import functions

        x = np.linspace(0, 100, 2000)
        rng = np.random.default_rng(0)
        y = getattr(functions, name)(x, 10.0, 50.0, 8.0)
        y = np.asarray(y) + 0.01 * rng.normal(size=x.size)
        idx = (
            np.arange(x.size)[::-1] if order == "reversed" else rng.permutation(x.size)
        )
        est = getattr(functions, f"estimate_p0_{name}")
        np.testing.assert_allclose(est(x[idx], y[idx]), est(x, y))
        assert est(x, y)[2] > 1.0  # sanity: not the 0.01 clamp


class TestFallbackSpecialOutputModes:
    """fallback=True must return what the non-fallback call returns (R6M-1)."""

    @pytest.mark.parametrize("mode", ["full_output", "return_eval", "timeit"])
    def test_tuple_modes_match_non_fallback(self, mode):
        x, y = _decay_data()
        plain = curve_fit(_exp_decay, x, y, p0=[1, 1], **{mode: True})
        fb = curve_fit(_exp_decay, x, y, p0=[1, 1], fallback=True, **{mode: True})
        assert type(fb) is type(plain) is tuple
        assert len(fb) == len(plain)
        np.testing.assert_allclose(fb[0], plain[0], rtol=1e-6)

    def test_default_returns_annotated_fallback_result(self):
        x, y = _decay_data()
        fb = curve_fit(_exp_decay, x, y, p0=[1, 1], fallback=True)
        popt, _pcov = fb
        np.testing.assert_allclose(popt, [2.5, 1.3], atol=0.15)
        assert fb["multistart_diagnostics"]["bypassed"] is True

    def test_fallback_with_multistart(self):
        x, y = _decay_data()
        fb = curve_fit(
            _exp_decay,
            x,
            y,
            p0=[1, 1],
            bounds=([0, 0], [10, 10]),
            fallback=True,
            multistart=True,
            n_starts=3,
        )
        np.testing.assert_allclose(fb.popt, [2.5, 1.3], atol=0.15)


def _lin(x, a, b):
    return a * x + b


class TestAutoBoundsAndMisc:
    """Round-7 findings F3, F4, F8, F9."""

    def test_auto_bounds_with_nan_omit_and_p0_none(self):
        # F3: auto_bounds estimated p0 on the raw NaN y.
        from nlsq.core.functions import gaussian

        x = np.linspace(0, 10, 50)
        y = 10 * np.exp(-((x - 5) ** 2) / 2)
        y[3] = np.nan
        r = curve_fit(gaussian, x, y, auto_bounds=True, nan_policy="omit")
        np.testing.assert_allclose(r.popt, [10.0, 5.0, 1.0], rtol=1e-2)

    @pytest.mark.parametrize(
        ("a", "b", "p0"),
        [(1.37, 1.32, None), (-2.0, 1.0, None), (1.0, 15.0, [1.0, 1.0])],
    )
    def test_auto_bounds_never_silently_excludes_the_solution(self, a, b, p0):
        # F4: inferred bounds clipped the solution and success=True was reported.
        from nlsq.result import OptimizeWarning

        x = np.linspace(0, 10, 100)
        with pytest.warns(OptimizeWarning, match="inferred bound"):
            r = curve_fit(_lin, x, a * x + b, p0=p0, auto_bounds=True)
        np.testing.assert_allclose(r.popt, [a, b], atol=1e-3)

    def test_auto_bounds_no_warning_when_bound_not_active(self):
        import warnings

        x = np.linspace(0, 4, 50)
        y = 2.5 * np.exp(-1.3 * x) + 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            curve_fit(
                lambda x, a, b, c: a * jnp.exp(-b * x) + c,
                x,
                y,
                p0=[2, 1, 0.4],
                auto_bounds=True,
            )

    def test_lm_with_auto_bounds_names_auto_bounds(self):
        # F8
        x = np.linspace(0, 10, 50)
        with pytest.raises(ValueError, match="auto_bounds"):
            curve_fit(_lin, x, 2 * x + 1, p0=[1, 1], method="lm", auto_bounds=True)

    def test_unknown_workflow_lists_only_accepted_names(self):
        # F9
        from nlsq import fit

        x = np.linspace(0, 10, 50)
        with pytest.raises(ValueError, match="Unknown workflow") as exc:
            fit(_lin, x, 2 * x + 1, workflow="bogus")
        msg = str(exc.value)
        for ok in ("auto", "auto_global", "hpc"):
            assert f"'{ok}'" in msg
        for removed in ("standard", "quality", "large_robust"):
            assert removed not in msg

    def test_numpy_jac_gives_clear_typeerror(self):
        # F7
        x = np.linspace(0, 10, 50)

        def np_jac(p, x, *a):
            return np.stack([np.asarray(x), np.ones(len(x))])

        with pytest.raises(TypeError, match="traceable"):
            curve_fit(_lin, x, 2 * x + 1, p0=[1, 1], jac=np_jac)


class TestRound8Routes:
    """R8-1 (auto_bounds relax on every route), R8-2 (cmaes output modes), R8-6."""

    @staticmethod
    def _clipped_problem():
        x = np.linspace(0, 1, 50)
        y = 500 * x - 3 + 0.01 * np.random.default_rng(3).standard_normal(50)
        kw = {"p0": [0.5, 0.3], "auto_bounds": True, "bounds_safety_factor": 1.0}
        return x, y, kw

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "route",
        [
            {},
            {"full_output": True},
            {"return_eval": True},
            {"timeit": True},
            {"multistart": True, "n_starts": 3},
            {"fallback": True},
        ],
    )
    def test_auto_bounds_relax_on_every_route(self, route):
        x, y, kw = self._clipped_problem()
        popt = np.asarray(curve_fit(_lin, x, y, **kw, **route)[0])
        np.testing.assert_allclose(popt, [500.0, -3.0], atol=0.05)

    def test_auto_bounds_tuple_mode_keeps_tuple_shape(self):
        x, y, kw = self._clipped_problem()
        out = curve_fit(_lin, x, y, full_output=True, **kw)
        assert type(out) is tuple and len(out) == 5

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        ("mode", "length"),
        [("full_output", 5), ("return_eval", 3), ("timeit", 5)],
    )
    def test_cmaes_output_modes(self, mode, length):
        from nlsq.global_optimization import CMAESConfig

        x = np.linspace(0, 4, 200)
        y = 2.5 * np.exp(-1.3 * x) + 0.5
        f = lambda x, a, b, c: a * jnp.exp(-b * x) + c
        kw = {"p0": [1, 1, 0], "bounds": ([0, 0, -2], [10, 5, 3])}
        trf = curve_fit(f, x, y, **kw, **{mode: True})
        cm = curve_fit(
            f,
            x,
            y,
            **kw,
            method="cmaes",
            cmaes_config=CMAESConfig(seed=0),
            **{mode: True},
        )
        assert type(cm) is type(trf) is tuple
        assert len(cm) == len(trf) == length
        np.testing.assert_allclose(cm[0], [2.5, 1.3, 0.5], atol=1e-3)
        if mode == "return_eval":
            assert len(cm[2]) == len(x)

    @pytest.mark.filterwarnings("ignore")
    def test_auto_bounds_with_cmaes(self):
        from nlsq.global_optimization import CMAESConfig

        x = np.linspace(0, 4, 200)
        y = 2.5 * np.exp(-1.3 * x) + 0.5
        f = lambda x, a, b, c: a * jnp.exp(-b * x) + c
        r = curve_fit(
            f,
            x,
            y,
            p0=[2, 1, 0.4],
            auto_bounds=True,
            method="cmaes",
            cmaes_config=CMAESConfig(seed=0),
        )
        np.testing.assert_allclose(r.popt, [2.5, 1.3, 0.5], atol=1e-2)


class TestUnknownKwargs:
    """R8-4: unknown keywords must not be swallowed silently."""

    def test_unknown_kwargs_warn(self):
        from nlsq.result import OptimizeWarning

        x = np.linspace(0, 10, 50)
        with pytest.warns(OptimizeWarning, match=r"foo_bar.*seed|seed.*foo_bar") as w:
            curve_fit(_lin, x, 2 * x + 1, p0=[1, 0], foo_bar=3, seed=1)
        assert "CMAESConfig" in str(w[0].message)

    def test_known_keywords_do_not_warn(self):
        import warnings

        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        y[3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            curve_fit(
                _lin, x, y, p0=[1, 0], nan_policy="omit", ftol=1e-10, max_nfev=200
            )

"""Unit tests for ``nlsq.core.problem.prepare_problem`` (Part B, Task B2)."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import CurveFit, curve_fit
from nlsq.core.problem import PrepareContext, PreparedProblem, prepare_problem

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
        assert ctx.validator is None
        assert ctx.logger is None
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.flength = 3  # type: ignore[misc]

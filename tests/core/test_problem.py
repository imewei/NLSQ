"""Unit tests for ``nlsq.core.problem.prepare_problem`` (Part B, Task B2)."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

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


class TestNEffectiveEquivalence:
    """``n_effective`` replaces ``ysize = int(np.sum(data_mask))`` downstream."""

    @pytest.mark.parametrize(
        "case",
        ["unpadded", "padded", "user_mask", "omitted", "padded_user_mask"],
    )
    def test_n_effective_equals_old_ysize(self, case):
        x, y = _data()
        ctx = PrepareContext(flength=100) if "padded" in case else PrepareContext()
        mask = None
        nan_policy = "raise"
        if "user_mask" in case:
            mask = np.ones(N, dtype=bool)
            mask[::6] = False
        if case == "omitted":
            y = y.copy()
            y[[3, 10]] = np.nan
            nan_policy = "omit"
        prep = _prepare(x, y, data_mask=mask, nan_policy=nan_policy, ctx=ctx)
        old_ysize = int(np.sum(prep.mask))  # the pre-refactor expression
        assert prep.n_effective == old_ysize
        expected = {
            "unpadded": N,
            "padded": N,
            "user_mask": 50,
            "omitted": N - 2,
            "padded_user_mask": 50,
        }[case]
        assert prep.n_effective == expected


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

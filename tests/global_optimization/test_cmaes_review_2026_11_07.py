"""Regression tests for the 2026-11-07 review of CMA-ES (GO-1..GO-3)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.global_optimization import CMAESConfig, CMAESOptimizer
from nlsq.global_optimization.cmaes_optimizer import _create_fitness_function

LB = jnp.array([0.0, 0.0])
UB = jnp.array([5.0, 5.0])
THETA = jnp.array([0.3, -0.4])  # unbounded-space candidates


def _lin(x, a, b):
    return a * x + b


def _fitness(model, x, y, sigma=None, chunk=None):
    fn = _create_fitness_function(model, x, y, LB, UB, sigma, data_chunk_size=chunk)
    return float(fn(THETA[None, :])[0])


class TestTwoDimSigma:
    """GO-1: 2-D covariance sigma must be whitened, not broadcast to inf."""

    def _data(self, n=50):
        t = jnp.linspace(0, 1, n)
        rng = np.random.default_rng(0)
        y = 1.0 * t + 2.0 + jnp.asarray(rng.normal(0, 0.1, n))
        return t, y

    @pytest.mark.parametrize("chunk", [None, 16, 25])
    def test_diag_covariance_matches_1d_sigma(self, chunk):
        t, y = self._data()
        s = jnp.full(t.shape, 0.1) * (1 + t)
        ref = _fitness(_lin, t, y, s)
        got = _fitness(_lin, t, y, jnp.diag(s**2), chunk)
        assert np.isfinite(got)
        assert got == pytest.approx(ref, rel=1e-9)

    @pytest.mark.parametrize("chunk", [None, 16])
    def test_full_covariance_matches_numpy_reference(self, chunk):
        t, y = self._data()
        n = t.shape[0]
        idx = np.arange(n)
        cov = 0.01 * 0.5 ** np.abs(idx[:, None] - idx[None, :])
        from nlsq.global_optimization.bounds_transform import transform_to_bounds

        p = np.asarray(transform_to_bounds(THETA, LB, UB))
        r = np.asarray(y) - (p[0] * np.asarray(t) + p[1])
        ell = np.linalg.cholesky(cov)
        expected = float(np.sum(np.linalg.solve(ell, r) ** 2))
        assert _fitness(_lin, t, y, jnp.asarray(cov), chunk) == pytest.approx(
            expected, rel=1e-8
        )

    def test_bad_shape_raises(self):
        t, y = self._data()
        with pytest.raises(ValueError, match="sigma"):
            _fitness(_lin, t, y, jnp.eye(t.shape[0] + 1))


class TestMultiDimXdataStreaming:
    """GO-2: chunk along the sample axis for (N,k) and (k,N) xdata."""

    N = 100
    CHUNK = 32  # remainder 4 -> exercises padding

    def _xy(self):
        t = np.linspace(0, 1, self.N)
        X = np.stack([t, 1 + t])  # (2, N)
        y = 1.0 * X[0] + 2.0 * X[1] + 0.05 * np.sin(40 * t)
        return jnp.asarray(X), jnp.asarray(y)

    def test_k_by_n_is_chunked_and_matches(self):
        X, y = self._xy()
        shapes = []

        def model(x, a, b):
            shapes.append(x.shape)
            return a * x[0] + b * x[1]

        ref = _fitness(model, X, y)
        shapes.clear()
        got = _fitness(model, X, y, chunk=self.CHUNK)
        assert (2, self.CHUNK) in shapes  # chunking actually happened
        assert got == pytest.approx(ref, rel=1e-10)

    def test_n_by_k_matches(self):
        X, y = self._xy()
        Xn = X.T  # (N, 2)

        def model(x, a, b):
            return a * x[:, 0] + b * x[:, 1]

        ref = _fitness(model, Xn, y)
        got = _fitness(model, Xn, y, chunk=self.CHUNK)
        assert np.isfinite(got)
        assert got == pytest.approx(ref, rel=1e-10)

    @pytest.mark.parametrize("sample_axis", [0, 1])
    def test_square_xdata_matches_full_evaluation(self, sample_axis, caplog):
        x = jnp.arange(16.0).reshape(4, 4)
        y = jnp.arange(4.0)

        def model(x, a, b):
            return a * jnp.sum(x, axis=1 - sample_axis) + b

        expected = _fitness(model, x, y)
        assert _fitness(model, x, y, chunk=2) == pytest.approx(expected, rel=1e-10)
        assert "sample axis" in caplog.text and "ambiguous" in caplog.text

    def test_padding_masked_for_singular_model(self):
        # padded x=0 would give inf for 1/x if not masked out of the SSR
        X, y = self._xy()
        Xn = X.T + 0.5

        def model(x, a, b):
            return a * x[:, 0] + b / x[:, 1]

        ref = _fitness(model, Xn, y)
        assert _fitness(model, Xn, y, chunk=self.CHUNK) == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("omit", [False, True])
def test_cmaes_excludes_masked_data_with_correlated_sigma(refine, omit):
    x = np.arange(8.0)
    y = 2 * x
    y[-1] = 1000
    mask = np.ones(len(x), dtype=bool)
    mask[-1] = False
    if omit:
        y[1] = np.nan
    covariance = 0.5 ** np.abs(np.subtract.outer(np.arange(8), np.arange(8)))
    config = CMAESConfig(
        max_generations=50, restart_strategy="none", seed=42, refine_with_nlsq=refine
    )
    result = curve_fit(
        lambda x, a: a * x,
        x,
        y,
        p0=[1.0],
        bounds=([0.0], [10.0]),
        sigma=covariance,
        absolute_sigma=True,
        data_mask=mask,
        nan_policy="omit" if omit else "raise",
        method="cmaes",
        cmaes_config=config,
    )
    assert result.success
    assert result.x[0] == pytest.approx(2, abs=0.05)
    assert len(result.ydata) == np.count_nonzero(mask & np.isfinite(y))
    if refine:
        valid = mask & np.isfinite(y)
        expected = 1 / (
            x[valid] @ np.linalg.solve(covariance[np.ix_(valid, valid)], x[valid])
        )
        assert result.pcov[0, 0] == pytest.approx(expected, rel=1e-5)


def test_cmaes_optimizer_directly_excludes_masked_data():
    x = np.arange(8.0)
    y = 2 * x
    y[-1] = 1000
    mask = np.arange(len(x)) != len(x) - 1
    config = CMAESConfig(
        max_generations=50, restart_strategy="none", seed=42, refine_with_nlsq=False
    )
    result = CMAESOptimizer(config).fit(
        lambda x, a: a * x, x, y, p0=[1.0], bounds=([0.0], [10.0]), data_mask=mask
    )
    assert result["popt"][0] == pytest.approx(2, abs=0.05)


def test_cmaes_mask_rejects_wrong_covariance_shape():
    config = CMAESConfig(max_generations=1, restart_strategy="none")
    with pytest.raises(ValueError, match="sigma covariance shape"):
        CMAESOptimizer(config).fit(
            lambda x, a: a * x,
            np.arange(4.0),
            np.arange(4.0),
            p0=[1.0],
            bounds=([0.0], [2.0]),
            sigma=np.eye(5),
            data_mask=[True, True, True, False],
        )


@pytest.mark.slow
class TestBipopScaleAware:
    """GO-3: stagnation must not kill every run for small-magnitude data."""

    def test_small_scale_runs_not_killed_at_gen_5(self):
        s = 1e-7
        x = np.linspace(0, 10, 300)

        def m(x, a, b, c):
            return a * jnp.sin(b * x) + c

        y = np.asarray(m(x, 2 * s, 3.0, 0.5 * s))
        cfg = CMAESConfig(
            max_generations=40,
            restart_strategy="bipop",
            max_restarts=1,
            seed=0,
            refine_with_nlsq=False,
        )
        res = CMAESOptimizer(cfg).fit(
            m, x, y, bounds=([0, 0, -5 * s], [5 * s, 10, 5 * s])
        )
        gens = [h["generations"] for h in res["cmaes_diagnostics"]["restart_history"]]
        assert min(gens) > 5, gens

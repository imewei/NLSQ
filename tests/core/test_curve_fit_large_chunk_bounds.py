"""curve_fit_large must not build an invalid chunk range for small inputs or a
small explicit chunk_size (min_chunk_size used to be floored at 1000 regardless)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit_large


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n):
    x = np.linspace(0, 5, n)
    y = 2.0 * np.exp(-0.7 * x) + 0.01 * np.random.default_rng(0).normal(size=n)
    return x, y


@pytest.mark.parametrize(
    ("n", "chunk_size"),
    [(2000, None), (20_000, 500)],
    ids=["small-dataset", "small-explicit-chunk"],
)
def test_forced_large_path_accepts_small_chunk_bounds(n, chunk_size):
    x, y = _data(n)
    popt, _ = curve_fit_large(
        _model,
        x,
        y,
        p0=[1.0, 1.0],
        auto_size_detection=False,
        chunk_size=chunk_size,
    )
    np.testing.assert_allclose(popt, [2.0, 0.7], atol=0.1)


class _Stop(Exception):
    """Raised by the spy so the test sees the config without running a fit."""


def _bounds_seen_by_fitter(monkeypatch, n, chunk_size):
    from nlsq.streaming.large_dataset import LargeDatasetFitter

    seen = []

    def spy(self, *args, **kwargs):
        seen.append((self.config.min_chunk_size, self.config.max_chunk_size))
        raise _Stop

    monkeypatch.setattr(LargeDatasetFitter, "fit", spy)
    x, y = _data(n)
    with pytest.raises(_Stop):
        curve_fit_large(
            _model,
            x,
            y,
            p0=[1.0, 1.0],
            auto_size_detection=False,
            chunk_size=chunk_size,
        )
    return seen[0]


@pytest.mark.parametrize("n", [2, 5, 9, 1000])
def test_tiny_inputs_run_as_a_single_chunk(monkeypatch, n):
    assert _bounds_seen_by_fitter(monkeypatch, n, None) == (n, n)


def test_default_ceiling_is_a_tenth_of_the_data_once_that_exceeds_the_floor(
    monkeypatch,
):
    assert _bounds_seen_by_fitter(monkeypatch, 20_000, None) == (1000, 2000)


def test_forced_large_path_on_a_tiny_input_matches_a_full_fit():
    # One-point chunks used to be fit separately and silently gave [2.08, 1.5].
    x = np.linspace(0, 1, 5)
    y = 2.0 * np.exp(-3.0 * x)
    popt, _ = curve_fit_large(_model, x, y, p0=[1.0, 1.0], auto_size_detection=False)
    np.testing.assert_allclose(popt, [2.0, 3.0], rtol=1e-6)


def test_explicit_chunk_size_is_the_ceiling_and_caps_the_floor(monkeypatch):
    assert _bounds_seen_by_fitter(monkeypatch, 20_000, 500) == (500, 500)


def test_chunk_size_above_the_default_ceiling_is_kept(monkeypatch):
    assert _bounds_seen_by_fitter(monkeypatch, 20_000, 1_500_000) == (
        1000,
        1_500_000,
    )


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_chunk_size_is_rejected_by_name(bad):
    x, y = _data(20_000)
    with pytest.raises(ValueError, match=r"^chunk_size must be positive"):
        curve_fit_large(
            _model,
            x,
            y,
            p0=[1.0, 1.0],
            auto_size_detection=False,
            chunk_size=bad,
        )


@pytest.mark.parametrize(
    ("n", "chunk_size", "sizes"),
    [
        (1001, 1000, [501, 500]),
        (10, 4, [4, 3, 3]),
        (2500, 1000, [834, 833, 833]),
        (5000, 1000, [1000] * 5),
    ],
)
def test_chunks_are_balanced_so_no_tail_is_tiny(n, chunk_size, sizes):
    from nlsq.streaming.large_dataset import DataChunker

    x = np.arange(n, dtype=float)
    chunks = list(DataChunker.create_chunks(x, x, chunk_size))
    assert [valid for *_, valid in chunks] == sizes
    assert np.concatenate([c[0][: c[3]] for c in chunks]).tolist() == x.tolist()


def test_balanced_chunks_with_sigma_stay_aligned():
    x = np.linspace(0, 5, 1001)
    y = 2 * x + 3 + 0.01 * np.random.default_rng(0).normal(size=1001)
    sigma = 0.01 + 0.5 * x  # varies along x, so a misaligned slice would be caught
    popt, _ = curve_fit_large(
        lambda x, a, b: a * x + b,
        x,
        y,
        p0=[1.0, 1.0],
        sigma=sigma,
        auto_size_detection=False,
    )
    np.testing.assert_allclose(popt, [2.0, 3.0], atol=0.05)


def test_ragged_tail_with_more_parameters_than_tail_points_matches_a_full_fit():
    from scipy.optimize import curve_fit

    def quad(x, a, b, c):
        return a * x * x + b * x + c

    x = np.linspace(0, 5, 10)
    y = 2 * x * x + 3 * x + 4
    y[1] += 100  # 4 + 4 + 2 chunks left a two-point tail for three parameters
    popt, _ = curve_fit_large(
        quad, x, y, p0=[1.0, 1.0, 1.0], chunk_size=4, auto_size_detection=False
    )
    expected, _ = curve_fit(quad, x, y, p0=[1.0, 1.0, 1.0])
    np.testing.assert_allclose(popt, expected, rtol=0.05)


def test_one_point_tail_does_not_corrupt_the_combined_fit():
    # 1001 points used to split 1000 + 1; the rank-deficient tail skewed the result.
    from scipy.optimize import curve_fit

    def line(x, a, b):
        return a * x + b

    x = np.linspace(0, 5, 1001)
    y = 2 * x + 3
    y[607] += 1000
    popt, _ = curve_fit_large(line, x, y, p0=[1.0, 1.0], auto_size_detection=False)
    expected, _ = curve_fit(line, x, y, p0=[1.0, 1.0])
    np.testing.assert_allclose(popt, expected, rtol=0.05)


def test_chunks_smaller_than_the_parameter_count_are_rejected():
    def quad(x, a, b, c):
        return a * x * x + b * x + c

    x = np.linspace(0, 5, 10)
    y = 2 * x * x + 3 * x + 4
    with pytest.raises(ValueError, match=r"chunk_size.*parameters"):
        curve_fit_large(
            quad, x, y, p0=[1.0, 1.0, 1.0], chunk_size=2, auto_size_detection=False
        )


def test_empty_input_yields_no_chunks():
    from nlsq.streaming.large_dataset import DataChunker

    assert DataChunker.chunk_bounds(0, 1000) == []
    assert list(DataChunker.create_chunks(np.array([]), np.array([]), 1000)) == []


def test_chunk_bounds_are_computed_once_per_fit(monkeypatch):
    from nlsq.streaming.large_dataset import DataChunker

    calls = []
    original = DataChunker.chunk_bounds

    def spy(n_points, chunk_size):
        calls.append((n_points, chunk_size))
        return original(n_points, chunk_size)

    monkeypatch.setattr(DataChunker, "chunk_bounds", staticmethod(spy))
    x = np.linspace(0, 5, 3000)
    y = 2 * x + 3
    sigma = np.full_like(x, 0.1)
    curve_fit_large(
        lambda x, a, b: a * x + b,
        x,
        y,
        p0=[1.0, 1.0],
        sigma=sigma,
        auto_size_detection=False,
    )
    assert len(calls) <= 2  # one for the loop's sigma slices, one inside create_chunks

"""Tests for the comparison harness core (types, problems, timing, accuracy gate)."""

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from benchmarks.compare.problems import PROBLEMS
from benchmarks.compare.timing import (
    all_converged,
    case_cost,
    reaches_optimum,
    residual_cost,
    time_fit,
)
from benchmarks.compare.types import FitOutcome


def test_data_is_deterministic_for_seed():
    p = PROBLEMS["exp_decay"]
    x0, y0 = p.data(1000, seed=0)
    x1, y1 = p.data(1000, seed=0)
    _, y2 = p.data(1000, seed=1)
    np.testing.assert_array_equal(x0, x1)
    np.testing.assert_array_equal(y0, y1)
    assert not np.array_equal(y0, y2)


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_model_numpy_and_jax_agree(name):
    p = PROBLEMS[name]
    x = np.linspace(*p.x_range, 500)
    diff = np.abs(p.model(np, x, *p.p_true) - np.asarray(p.model(jnp, x, *p.p_true)))
    assert diff.max() < 1e-10


def test_bounded_problem_truth_violates_a_bound():
    p = PROBLEMS["bounded_exp"]
    assert p.bounds is not None
    assert p.p_true[0] > p.bounds[1][0]


def test_batch_data_shapes_and_independent_noise():
    x, Y = PROBLEMS["gaussian"].batch_data(200, 4)
    assert x.shape == (200,)
    assert Y.shape == (4, 200)
    assert not np.array_equal(Y[0], Y[1])


def test_time_fit_calls_once_cold_then_repeats_warm():
    calls = []

    def run():
        calls.append(1)
        return FitOutcome(np.zeros(3), True, 1)

    timing, out = time_fit(run, repeats=5)
    assert len(calls) == 1 + 5
    assert isinstance(out, FitOutcome)
    assert timing.cold_s >= 0 and timing.warm_median_s >= 0 and timing.warm_iqr_s >= 0


def test_reaches_optimum_gate():
    assert reaches_optimum(1.0 + 1e-9, 1.0)
    assert not reaches_optimum(1.0 + 1e-3, 1.0)
    assert reaches_optimum(0.0, 0.0)
    assert not reaches_optimum(1e-3, 0.0)


def test_case_cost_averages_a_batch_and_all_converged_requires_every_fit():
    p = PROBLEMS["exp_decay"]
    x, y = p.data(100)
    truth = np.asarray(p.p_true)
    single = residual_cost(p, x, y, truth)
    batch = [FitOutcome(truth, True, 1), FitOutcome(truth, False, 1)]
    Y = np.stack([y, y])
    assert case_cost(p, x, Y, batch) == pytest.approx(single)
    assert case_cost(p, x, y, FitOutcome(truth, True, 1)) == pytest.approx(single)
    assert not all_converged(batch)
    assert all_converged(batch[:1])


def test_residual_cost_is_zero_at_noiseless_truth():
    p = PROBLEMS["two_exp"]
    x, y = p.data(200, noise=0.0)
    assert residual_cost(p, x, y, np.asarray(p.p_true)) < 1e-20

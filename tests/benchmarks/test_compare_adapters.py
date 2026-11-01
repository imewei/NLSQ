"""Tests for the comparison-harness library adapters."""

import numpy as np
import pytest

from benchmarks.compare.adapters import ADAPTERS
from benchmarks.compare.problems import PROBLEMS
from benchmarks.compare.timing import residual_cost

NAMES = list(ADAPTERS)
EXP = PROBLEMS["exp_decay"]


def _cases():
    return [(n, m) for n, a in ADAPTERS.items() for m in a.modes]


def test_registry_has_the_seven_adapters():
    assert set(ADAPTERS) == {
        "nlsq",
        "nlsq_large",
        "nlsq_exact",
        "scipy",
        "lmfit",
        "jaxopt",
        "optimistix",
    }


@pytest.mark.parametrize("name", NAMES)
def test_every_adapter_reports_availability_with_reason(name):
    av = ADAPTERS[name].availability()
    if av.ok:
        assert av.version
    else:
        assert av.reason


@pytest.mark.parametrize(("name", "mode"), _cases())
def test_available_adapters_recover_truth_on_exp_decay(name, mode):
    adapter = ADAPTERS[name]
    av = adapter.availability()
    # every adapter's library is a declared dependency: fail rather than pass vacuously
    assert av.ok, f"{name} unavailable: {av.reason}"
    x, y = EXP.data(1000)
    out = adapter.prepare(EXP, x, y, mode)()
    truth = np.asarray(EXP.p_true)
    assert out.converged
    assert np.max(np.abs(np.asarray(out.params) - truth) / np.abs(truth)) < 0.05


@pytest.mark.parametrize("name", ["nlsq", "scipy"])
def test_core_dependencies_are_available(name):
    assert ADAPTERS[name].availability().ok, ADAPTERS[name].availability().reason


@pytest.mark.parametrize("name", NAMES)
def test_unbounded_libraries_decline_the_bounded_problem(name):
    ok, reason = ADAPTERS[name].supports(PROBLEMS["bounded_exp"])
    # nlsq_exact declines bounds until Task 9 (Stage 2) adds them
    if name in ("jaxopt", "optimistix", "nlsq_exact"):
        assert not ok
        assert "bounds" in reason
    else:
        assert (ok, reason) == (True, "")


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_bounded_fit_respects_bounds(name):
    adapter = ADAPTERS[name]
    p = PROBLEMS["bounded_exp"]
    av = adapter.availability()
    assert av.ok, f"{name} unavailable: {av.reason}"
    if not adapter.supports(p)[0]:  # real capability gap, asserted in the test above
        return
    x, y = p.data(1000)
    out = adapter.prepare(p, x, y, "end_to_end")()
    lo, hi = np.asarray(p.bounds)
    params = np.asarray(out.params)
    assert np.all(params >= lo - 1e-9)
    assert np.all(params <= hi + 1e-9)


@pytest.mark.parametrize(("name", "mode"), _cases())
def test_batch_cost_matches_single_fit_cost(name, mode):
    adapter = ADAPTERS[name]
    av = adapter.availability()
    assert av.ok, f"{name} unavailable: {av.reason}"
    x, Y = EXP.batch_data(200, 4)
    batch = adapter.prepare_batch(EXP, x, Y, mode)()
    assert len(batch) == 4
    single = adapter.prepare(EXP, x, Y[0], mode)()
    cb = residual_cost(EXP, x, Y[0], batch[0].params)
    cs = residual_cost(EXP, x, Y[0], single.params)
    assert cb == pytest.approx(cs, rel=1e-6)


def test_nlsq_exact_reports_solver_failure_as_unconverged_outcome(monkeypatch):
    """ExactSolverError (stalled / max_nfev) becomes converged=False, not a crash."""
    from nlsq.result.optimize_result import OptimizeResult
    from nlsq.streaming.exact_lm import ExactSolverError

    res = OptimizeResult(
        x=np.array([1.0, 2.0, 3.0]), nfev=7, success=False, termination="max_nfev"
    )

    def boom(*a, **k):
        raise ExactSolverError("exact_streaming fit failed (max_nfev)", res)

    monkeypatch.setattr("benchmarks.compare.adapters.nlsq.curve_fit_large", boom)
    x, y = EXP.data(100)
    out = ADAPTERS["nlsq_exact"].prepare(EXP, x, y, "end_to_end")()
    assert out.converged is False
    assert out.nfev == 7
    np.testing.assert_array_equal(out.params, [1.0, 2.0, 3.0])


def test_jaxopt_capability_records_unmaintained():
    assert ADAPTERS["jaxopt"].capabilities.maintained is False


def test_jaxopt_stops_at_the_optimum_on_large_n():
    # JAXopt's default grad-l2-norm criterion ignores xtol/gtol and runs to MAX_ITER
    av = ADAPTERS["jaxopt"].availability()
    # jaxopt is a declared `bench` dependency: fail rather than pass vacuously
    assert av.ok, f"JAXopt unavailable: {av.reason}"
    x, y = EXP.data(100_000)
    out = ADAPTERS["jaxopt"].prepare(EXP, x, y, "end_to_end")()
    assert out.converged

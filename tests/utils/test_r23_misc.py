"""Round-23 regressions."""

import numpy as np
import pytest

from nlsq.utils.diagnostics import ConvergenceMonitor
from nlsq.utils.safe_serialize import SafeSerializationError, safe_loads


def test_plateau_then_drop_is_not_oscillation():
    mon = ConvergenceMonitor(window_size=10)
    for c in (5, 5, 2.5, 2.5, 1.25, 1.25, 0.6, 0.6, 0.3, 0.3, 0.15, 0.15):
        mon.cost_history.append(c)
    assert not mon.detect_oscillation()[0]


def test_safe_loads_overflow_is_safe_serialization_error():
    payload = (
        b'{"__nlsq_type__": "ndarray", "dtype": "int8", "shape": [1], "data": [1000]}'
    )
    with pytest.raises(SafeSerializationError):
        safe_loads(payload)


def test_min_chunk_size_must_not_exceed_max():
    from nlsq.streaming.large_dataset import LDMemoryConfig

    with pytest.raises(ValueError, match="min_chunk_size"):
        LDMemoryConfig(min_chunk_size=2000, max_chunk_size=500)


def test_csv_stray_quote_in_preamble_does_not_swallow_data(tmp_path):
    from nlsq.cli.data_loaders import DataLoader

    p = tmp_path / "d.csv"
    p.write_text('"Run 3 notes\nx,y\n1,2\n3,4\n5,6\n')
    x, _y, _ = DataLoader().load(
        p, {"format": "csv", "csv": {"header": True, "skip_header": 1}}
    )
    np.testing.assert_allclose(x, [1, 3, 5])


def test_nlsq_debug_zero_is_off():
    import subprocess
    import sys

    out = (
        subprocess.run(
            [sys.executable, "-c", "import nlsq; print(nlsq._NLSQ_DEBUG)"],
            capture_output=True,
            text=True,
            check=True,
            env={**__import__("os").environ, "NLSQ_DEBUG": "0"},
        )
        .stdout.strip()
        .splitlines()[-1]
    )
    assert out == "False"


def test_rejected_steps_with_monotone_progress_are_not_oscillation():
    mon = ConvergenceMonitor()
    p = np.array([1.0, 2.0])
    cost = 100.0
    for i in range(20):
        if i % 2 == 0:
            cost *= 0.7
            p = p + np.array([0.1, -0.05])
        mon.update(cost, p.copy())
    assert not mon.detect_oscillation()[0]


def test_true_oscillation_still_detected():
    mon = ConvergenceMonitor()
    for i in range(20):
        mon.update(10 + (-1) ** i, np.array([1.0 + 0.1 * (-1) ** i]))
    assert mon.detect_oscillation()[0]


def test_npz_complex_with_zero_imag_loads(tmp_path):
    from nlsq.cli.data_loaders import DataLoader

    p = tmp_path / "c.npz"
    np.savez(p, x=np.arange(5.0) + 0j, y=2 * np.arange(5.0) + 0j)
    x, y, _ = DataLoader().load(p, {"format": "npz"})
    np.testing.assert_allclose(y, 2 * x)


def test_exponential_decay_p0_with_shifted_x():
    from nlsq import curve_fit
    from nlsq.core.functions import exponential_decay

    t = np.linspace(0, 10, 50)
    x = t + 20.0
    y = 3 * np.exp(-0.5 * t) + 1  # == 3*exp(10)*exp(-0.5 x)+1
    popt, _ = curve_fit(
        exponential_decay,
        x,
        y,
        p0="auto",
        bounds=exponential_decay.bounds(),
        max_nfev=2000,
    )
    assert np.sqrt(np.mean((np.asarray(exponential_decay(x, *popt)) - y) ** 2)) < 1e-3


def test_multistart_does_not_hide_unknown_kwarg_warning():
    import warnings

    import jax.numpy as jnp

    from nlsq import curve_fit

    x = np.linspace(0, 4, 100)
    y = 2 * np.exp(-0.7 * x)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        curve_fit(
            lambda x, a, b: a * jnp.exp(-b * x),
            x,
            y,
            p0=[1, 1],
            bounds=([0, 0], [5, 5]),
            multistart=True,
            n_starts=2,
            maxfevv=3,
        )
    assert any("maxfevv" in str(i.message) for i in w)


def test_scipy_bounds_object_accepted():
    import jax.numpy as jnp
    from scipy.optimize import Bounds

    from nlsq import curve_fit

    x = np.linspace(0, 4, 100)
    y = 2 * np.exp(-0.7 * x)
    popt, _ = curve_fit(
        lambda x, a, b: a * jnp.exp(-b * x),
        x,
        y,
        p0=[1, 1],
        bounds=Bounds([0, 0], [5, 5]),
    )
    np.testing.assert_allclose(popt, [2, 0.7], rtol=1e-4)


def test_cmaes_result_reports_refinement_counts():
    import jax.numpy as jnp

    from nlsq import curve_fit
    from nlsq.global_optimization import CMAESConfig

    x = np.linspace(0, 4, 60)
    y = 2 * np.exp(-0.7 * x)
    r = curve_fit(
        lambda x, a, b: a * jnp.exp(-b * x),
        x,
        y,
        p0=[1, 1],
        bounds=([0, 0], [5, 5]),
        method="cmaes",
        cmaes_config=CMAESConfig(max_generations=20, seed=0),
    )
    assert r.nfev > 0 and r.cost is not None


def test_npz_wrong_container_raises_dataloaderror(tmp_path):
    from nlsq.cli.data_loaders import DataLoader, DataLoadError

    p = tmp_path / "a.npz"
    np.save(tmp_path / "a.npy", np.arange(3.0))
    (tmp_path / "a.npy").rename(p)
    with pytest.raises(DataLoadError):
        DataLoader().load(p, {"format": "npz"})

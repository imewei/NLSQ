"""Tests for progress callbacks functionality."""

import io
import os
import tempfile
import warnings

import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.callbacks import (
    CallbackBase,
    CallbackChain,
    EarlyStopping,
    IterationLogger,
    ProgressBar,
    StopOptimization,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_data():
    """Generate simple exponential decay data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 100 * np.exp(-0.5 * x) + 10
    y = y_true + np.random.normal(0, 3, size=len(x))
    return x, y


@pytest.fixture
def simple_model():
    """Simple exponential decay model."""
    import jax.numpy as jnp

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    return model


# ============================================================================
# Unit Tests - CallbackBase
# ============================================================================


def test_callback_base():
    """Test CallbackBase can be subclassed."""

    class TestCallback(CallbackBase):
        def __init__(self):
            self.calls = []

        def __call__(self, iteration, cost, params, info):
            self.calls.append((iteration, cost, params.copy(), info.copy()))

    callback = TestCallback()
    callback(1, 10.5, np.array([1, 2, 3]), {"nfev": 10})

    assert len(callback.calls) == 1
    assert callback.calls[0][0] == 1
    assert callback.calls[0][1] == 10.5
    assert np.array_equal(callback.calls[0][2], [1, 2, 3])
    assert callback.calls[0][3]["nfev"] == 10


def test_callback_base_close():
    """Test CallbackBase close method."""
    callback = CallbackBase()
    callback.close()  # Should not raise


# ============================================================================
# Unit Tests - ProgressBar
# ============================================================================


def test_progressbar_creation():
    """Test ProgressBar can be created."""
    try:
        callback = ProgressBar(max_nfev=100)
        callback.close()
    except ImportError:
        pytest.skip("tqdm not installed")


def test_progressbar_without_tqdm():
    """Test ProgressBar gracefully handles missing tqdm.

    Note: This test is challenging to implement due to Python's import caching.
    The warning is properly issued when tqdm is actually missing, but testing
    this requires preventing the import at module load time.
    """
    # If tqdm is already imported (which it usually is), the test behavior changes
    # The ProgressBar will work but won't issue a warning on subsequent instantiations
    try:
        pass

        # tqdm is available, so we can't test the warning behavior reliably
        # But we can verify ProgressBar still works
        callback = ProgressBar(max_nfev=100)
        callback(1, 10.5, np.array([1, 2, 3]), {"nfev": 10})
        callback.close()
    except ImportError:
        # tqdm not available - this is the scenario we want to test
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            callback = ProgressBar(max_nfev=100)

            # Should warn about missing tqdm
            assert len(w) >= 1
            assert "tqdm not installed" in str(w[0].message).lower()

            # Should still be callable without error
            callback(1, 10.5, np.array([1, 2, 3]), {"nfev": 10})
            callback.close()


def test_progressbar_updates():
    """Test ProgressBar updates with iteration data."""
    try:
        pass

        callback = ProgressBar(max_nfev=10)
        callback(1, 100.0, np.array([1, 2, 3]), {"nfev": 3, "gradient_norm": 10.0})
        callback(2, 50.0, np.array([1.1, 2.1, 3.1]), {"nfev": 6, "gradient_norm": 5.0})
        callback.close()
    except ImportError:
        pytest.skip("tqdm not installed")


# ============================================================================
# Unit Tests - IterationLogger
# ============================================================================


def test_iteration_logger_stdout():
    """Test IterationLogger logs to stdout."""
    buffer = io.StringIO()
    callback = IterationLogger(filename=None, file=buffer, log_params=False)

    callback(1, 100.0, np.array([1, 2, 3]), {"nfev": 3, "gradient_norm": 10.0})
    callback(2, 50.0, np.array([1.1, 2.1, 3.1]), {"nfev": 6, "gradient_norm": 5.0})
    callback.close()

    output = buffer.getvalue()
    assert "Iter" in output and "1" in output
    assert "Iter" in output and "2" in output
    # Values may be in scientific notation (1.000000e+02) or decimal (100.0)
    assert "100" in output or "1.000000e+02" in output
    assert "50" in output or "5.000000e+01" in output


def test_iteration_logger_file():
    """Test IterationLogger logs to file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        filename = f.name

    try:
        callback = IterationLogger(filename=filename, mode="w", log_params=True)
        callback(1, 100.0, np.array([1.0, 2.0, 3.0]), {"nfev": 3})
        callback(2, 50.0, np.array([1.1, 2.1, 3.1]), {"nfev": 6})
        callback.close()

        with open(filename) as f:
            content = f.read()

        assert "Iter" in content and "1" in content
        assert "Iter" in content and "2" in content
        assert "1.0" in content  # Parameter value
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_iteration_logger_no_params():
    """Test IterationLogger without parameter logging."""
    buffer = io.StringIO()
    callback = IterationLogger(filename=None, file=buffer, log_params=False)

    callback(1, 100.0, np.array([1, 2, 3]), {"nfev": 3})
    callback.close()

    output = buffer.getvalue()
    assert "Iter" in output and "1" in output
    # Parameters should not be logged
    assert "params=" not in output.lower()


# ============================================================================
# Unit Tests - EarlyStopping
# ============================================================================


def test_early_stopping_patience():
    """Test EarlyStopping triggers after patience iterations."""
    callback = EarlyStopping(patience=3, min_delta=1e-6, verbose=False)

    # Improving iterations
    callback(1, 100.0, np.array([1, 2, 3]), {"nfev": 3})
    callback(2, 50.0, np.array([1, 2, 3]), {"nfev": 6})
    callback(3, 25.0, np.array([1, 2, 3]), {"nfev": 9})

    # Stalled iterations
    callback(4, 25.0, np.array([1, 2, 3]), {"nfev": 12})
    callback(5, 25.0, np.array([1, 2, 3]), {"nfev": 15})

    # Should raise StopOptimization on 3rd stalled iteration
    with pytest.raises(StopOptimization):
        callback(6, 25.0, np.array([1, 2, 3]), {"nfev": 18})


def test_early_stopping_min_delta():
    """Test EarlyStopping respects min_delta threshold."""
    callback = EarlyStopping(patience=2, min_delta=1.0, verbose=False)

    callback(1, 100.0, np.array([1, 2, 3]), {"nfev": 3})

    # Small improvement (< min_delta) should count as no improvement
    callback(2, 99.5, np.array([1, 2, 3]), {"nfev": 6})

    # Should raise after 2 iterations without significant improvement
    with pytest.raises(StopOptimization):
        callback(3, 99.3, np.array([1, 2, 3]), {"nfev": 9})


def test_early_stopping_reset_on_improvement():
    """Test EarlyStopping resets counter on improvement."""
    callback = EarlyStopping(patience=2, min_delta=1e-6, verbose=False)

    callback(1, 100.0, np.array([1, 2, 3]), {"nfev": 3})
    callback(2, 100.0, np.array([1, 2, 3]), {"nfev": 6})  # Stall
    callback(3, 50.0, np.array([1, 2, 3]), {"nfev": 9})  # Improvement - reset
    callback(4, 50.0, np.array([1, 2, 3]), {"nfev": 12})  # Stall again

    # Should raise after 2 stalled iterations (counter was reset at iter 3)
    with pytest.raises(StopOptimization):
        callback(5, 50.0, np.array([1, 2, 3]), {"nfev": 15})


# ============================================================================
# Unit Tests - CallbackChain
# ============================================================================


def test_callback_chain():
    """Test CallbackChain combines multiple callbacks."""
    calls1 = []
    calls2 = []

    class Callback1(CallbackBase):
        def __call__(self, iteration, cost, params, info):
            calls1.append(iteration)

    class Callback2(CallbackBase):
        def __call__(self, iteration, cost, params, info):
            calls2.append(iteration)

    chain = CallbackChain(Callback1(), Callback2())
    chain(1, 100.0, np.array([1, 2, 3]), {"nfev": 3})
    chain(2, 50.0, np.array([1, 2, 3]), {"nfev": 6})

    assert calls1 == [1, 2]
    assert calls2 == [1, 2]


def test_callback_chain_stops_on_exception():
    """Test CallbackChain stops when callback raises StopOptimization."""

    class Callback1(CallbackBase):
        def __call__(self, iteration, cost, params, info):
            if iteration >= 2:
                raise StopOptimization("Stop!")

    class Callback2(CallbackBase):
        def __call__(self, iteration, cost, params, info):
            pass  # Should not be called after Callback1 raises

    chain = CallbackChain(Callback1(), Callback2())
    chain(1, 100.0, np.array([1, 2, 3]), {"nfev": 3})

    with pytest.raises(StopOptimization):
        chain(2, 50.0, np.array([1, 2, 3]), {"nfev": 6})


def test_callback_chain_close():
    """Test CallbackChain closes all callbacks."""
    closed = []

    class TestCallback(CallbackBase):
        def __init__(self, idx):
            self.idx = idx

        def close(self):
            closed.append(self.idx)

    chain = CallbackChain(TestCallback(1), TestCallback(2), TestCallback(3))
    chain.close()

    assert closed == [1, 2, 3]


# ============================================================================
# Integration Tests - curve_fit with callbacks
# ============================================================================


def test_curve_fit_with_progress_callback(simple_data, simple_model):
    """Test curve_fit with ProgressBar callback."""
    try:
        pass

        x, y = simple_data
        callback = ProgressBar(max_nfev=50)

        popt, _pcov = curve_fit(
            simple_model, x, y, p0=[80, 0.4, 5], callback=callback, max_nfev=50
        )

        callback.close()

        # Should converge to reasonable values
        assert 90 < popt[0] < 110  # amplitude
        assert 0.3 < popt[1] < 0.7  # rate
        assert 5 < popt[2] < 15  # offset
    except ImportError:
        pytest.skip("tqdm not installed")


def test_curve_fit_with_logger_callback(simple_data, simple_model):
    """Test curve_fit with IterationLogger callback."""
    x, y = simple_data
    buffer = io.StringIO()
    callback = IterationLogger(filename=None, file=buffer, log_params=False)

    _popt, _pcov = curve_fit(
        simple_model, x, y, p0=[80, 0.4, 5], callback=callback, max_nfev=50
    )

    callback.close()

    output = buffer.getvalue()
    # Should have logged some iterations
    assert "Iter" in output
    assert "Cost:" in output or "cost=" in output


def test_curve_fit_with_early_stopping(simple_data, simple_model):
    """Test curve_fit with EarlyStopping callback."""
    x, y = simple_data
    callback = EarlyStopping(patience=10, min_delta=1e-6, verbose=False)

    popt, _pcov = curve_fit(
        simple_model,
        x,
        y,
        p0=[80, 0.4, 5],
        callback=callback,
        max_nfev=1000,  # Set high, but early stopping should trigger
    )

    # Should still converge
    assert 90 < popt[0] < 110


def test_curve_fit_with_callback_chain(simple_data, simple_model):
    """Test curve_fit with CallbackChain."""
    x, y = simple_data
    buffer = io.StringIO()

    chain = CallbackChain(
        IterationLogger(filename=None, file=buffer, log_params=False),
        EarlyStopping(patience=10, verbose=False),
    )

    popt, _pcov = curve_fit(
        simple_model, x, y, p0=[80, 0.4, 5], callback=chain, max_nfev=1000
    )

    chain.close()

    # Should converge
    assert 90 < popt[0] < 110

    # Should have logged
    output = buffer.getvalue()
    assert "Iter" in output


def test_curve_fit_callback_receives_correct_data(simple_data, simple_model):
    """Test callback receives correct iteration data."""
    received_data = []

    class DataCapture(CallbackBase):
        def __call__(self, iteration, cost, params, info):
            received_data.append(
                {
                    "iteration": iteration,
                    "cost": cost,
                    "params": params.copy(),
                    "info": info.copy(),
                }
            )

    x, y = simple_data
    callback = DataCapture()

    _popt, _pcov = curve_fit(
        simple_model, x, y, p0=[80, 0.4, 5], callback=callback, max_nfev=50
    )

    # Should have received data for multiple iterations
    assert len(received_data) > 0

    # Check data structure
    for data in received_data:
        assert "iteration" in data
        assert "cost" in data
        assert "params" in data
        assert "info" in data
        assert isinstance(data["iteration"], int)
        assert isinstance(data["cost"], (int, float))
        assert isinstance(data["params"], np.ndarray)
        assert len(data["params"]) == 3  # 3 parameters


def test_curve_fit_callback_error_handling(simple_data, simple_model):
    """Test curve_fit handles callback errors gracefully."""
    call_count = [0]

    class FailingCallback(CallbackBase):
        def __call__(self, iteration, cost, params, info):
            call_count[0] += 1
            if iteration == 2:
                raise ValueError("Test error")

    x, y = simple_data
    callback = FailingCallback()

    # Should continue optimization despite callback error
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        popt, _pcov = curve_fit(
            simple_model, x, y, p0=[80, 0.4, 5], callback=callback, max_nfev=50
        )

        # Should have warned about callback failure
        assert any("Callback raised exception" in str(warning.message) for warning in w)

    # Optimization should have completed
    assert 90 < popt[0] < 110

    # Callback should have been called at least once
    assert call_count[0] > 0


def test_curve_fit_callback_none(simple_data, simple_model):
    """Test curve_fit works with callback=None."""
    x, y = simple_data

    popt, _pcov = curve_fit(
        simple_model,
        x,
        y,
        p0=[80, 0.4, 5],
        callback=None,  # Explicit None
        max_nfev=50,
    )

    # Should converge normally
    assert 90 < popt[0] < 110


def test_curve_fit_without_callback_parameter(simple_data, simple_model):
    """Test curve_fit works without callback parameter (backward compatibility)."""
    x, y = simple_data

    popt, _pcov = curve_fit(
        simple_model,
        x,
        y,
        p0=[80, 0.4, 5],
        max_nfev=50,
        # No callback parameter
    )

    # Should converge normally
    assert 90 < popt[0] < 110

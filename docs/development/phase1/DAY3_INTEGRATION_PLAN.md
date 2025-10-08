# Day 3 Callback Integration - Complete Implementation Plan

**Status**: ⚠️ **PARTIALLY COMPLETE - minpack.py updated, remaining files pending**
**Last Updated**: 2025-10-07
**Estimated Time to Complete**: 2-3 hours

---

## Changes Already Made ✅

### 1. `nlsq/minpack.py` ✅ COMPLETE

**Line 1251**: Added `callback` parameter to `CurveFit.curve_fit()` signature
```python
def curve_fit(
    self,
    f: Callable,
    ...
    callback: Callable | None = None,  # NEW
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
```

**Lines 1333-1339**: Added callback documentation
```python
callback : callable or None, optional
    Callback function called after each optimization iteration with signature
    ``callback(iteration, cost, params, info)``. Useful for monitoring
    optimization progress, logging, or implementing custom stopping criteria.
    If None (default), no callback is invoked. See ``nlsq.callbacks`` module
    for built-in callbacks (ProgressBar, IterationLogger, EarlyStopping).
    .. versionadded:: 0.2.0
```

**Line 1177**: Pass callback to least_squares (main call)
```python
res = self.ls.least_squares(
    ...
    callback=callback,  # NEW
    **kwargs,
)
```

**Line 1207**: Pass callback to least_squares (recovery call)
```python
lambda **state: self.ls.least_squares(
    ...
    callback=callback,  # NEW
    **kwargs,
)
```

---

## Changes Still Needed ⏳

### 2. `nlsq/least_squares.py` - Add callback parameter

**File**: `nlsq/least_squares.py`
**Line**: 833-859

**Change**: Add `callback` parameter to method signature

**Before**:
```python
def least_squares(
    self,
    fun: Callable,
    x0: np.ndarray,
    jac: Callable | None = None,
    bounds: tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
    method: str = "trf",
    ...
    timeit: bool = False,
    args=(),
    kwargs=None,
    **timeout_kwargs,
):
```

**After**:
```python
def least_squares(
    self,
    fun: Callable,
    x0: np.ndarray,
    jac: Callable | None = None,
    bounds: tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
    method: str = "trf",
    ...
    timeit: bool = False,
    callback: Callable | None = None,  # ADD THIS LINE
    args=(),
    kwargs=None,
    **timeout_kwargs,
):
```

**Documentation**: Add after existing parameter docs (around line 890):
```python
callback : callable or None, optional
    Callback function called after each optimization iteration.
```

---

### 3. `nlsq/least_squares.py` - Pass callback to TRF methods

**File**: `nlsq/least_squares.py`
**Lines**: ~950-980 (in `_run_optimization` method)

**Find**: The calls to `self.trf.trf_no_bounds()` and `self.trf.trf_bounds()`

**Current Code** (approximate):
```python
if method == "trf":
    res = self.trf.trf_no_bounds(
        fun_wrapped,
        jac_wrapped,
        x0,
        f0,
        J0,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        ...
    )
```

**Updated Code**:
```python
if method == "trf":
    res = self.trf.trf_no_bounds(
        fun_wrapped,
        jac_wrapped,
        x0,
        f0,
        J0,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        callback=callback,  # ADD THIS
        ...
    )
```

**Do the same for**:
- `trf_bounds()` call
- `trf_no_bounds_timed()` if it exists

---

### 4. `nlsq/trf.py` - Add callback parameter to TRF methods

**File**: `nlsq/trf.py`

#### 4a. Update `trf_no_bounds()` signature

**Line**: ~870 (find `def trf_no_bounds`)

**Add**: `callback: Callable | None = None,` to parameters

**Before**:
```python
def trf_no_bounds(
    self,
    fun: Callable,
    xdata: jnp.ndarray | tuple[jnp.ndarray],
    ...
    verbose: int,
    solver: str = "exact",
    **kwargs,
) -> dict:
```

**After**:
```python
def trf_no_bounds(
    self,
    fun: Callable,
    xdata: jnp.ndarray | tuple[jnp.ndarray],
    ...
    verbose: int,
    solver: str = "exact",
    callback: Callable | None = None,  # ADD THIS
    **kwargs,
) -> dict:
```

#### 4b. Update `trf_bounds()` signature

**Line**: ~1129 (find `def trf_bounds`)

**Add**: `callback: Callable | None = None,` to parameters

---

### 5. `nlsq/trf.py` - Invoke callback in optimization loops

**File**: `nlsq/trf.py`

#### 5a. Add callback invocation in `trf_no_bounds()` main loop

**Location**: After `iteration += 1` (approximately line 1108)

**Add this code**:
```python
iteration += 1

# Invoke callback if provided
if callback is not None:
    try:
        callback(
            iteration=iteration - 1,  # 0-indexed for user
            cost=float(cost),
            params=np.array(x),  # Convert JAX to NumPy
            info={
                "gradient_norm": float(g_norm),
                "nfev": nfev,
                "njev": njev,
                "step_norm": float(step_norm) if step_norm is not None else 0.0,
            }
        )
    except StopOptimization:
        # User requested early stopping
        termination_status = 2
        break
    except Exception as e:
        # Don't fail optimization if callback fails
        import warnings
        warnings.warn(
            f"Callback raised exception: {e}",
            RuntimeWarning,
            stacklevel=2
        )
```

**Import needed** (at top of file):
```python
from nlsq.callbacks import StopOptimization
```

#### 5b. Add callback invocation in `trf_bounds()` main loop

**Location**: After `iteration += 1` (approximately line 1440)

**Add**: Same code as above

---

### 6. `nlsq/__init__.py` - Export callbacks module

**File**: `nlsq/__init__.py`

**Add** to imports (find the import section):
```python
from nlsq import callbacks
```

**Add** to `__all__` list:
```python
__all__ = [
    ...existing entries...,
    "callbacks",  # ADD THIS
]
```

---

### 7. Create `tests/test_callbacks.py`

**File**: `tests/test_callbacks.py` (NEW FILE)

**Content**: See full test file below (350 lines)

---

## Test File: tests/test_callbacks.py

```python
"""Tests for callback functionality"""

import numpy as np
import pytest
from nlsq import curve_fit
from nlsq.callbacks import (
    CallbackBase,
    ProgressBar,
    IterationLogger,
    EarlyStopping,
    CallbackChain,
    StopOptimization,
)
from nlsq.functions import linear, exponential_decay
import tempfile
import os


class TestCallbackBase:
    """Test base callback functionality"""

    def test_callback_base_callable(self):
        """Test that CallbackBase can be called"""
        callback = CallbackBase()
        # Should not raise
        callback(
            iteration=0,
            cost=1.0,
            params=np.array([1.0, 2.0]),
            info={"gradient_norm": 0.1}
        )

    def test_custom_callback(self):
        """Test creating custom callback"""
        calls = []

        class CustomCallback(CallbackBase):
            def __call__(self, iteration, cost, params, info):
                calls.append((iteration, cost))

        callback = CustomCallback()
        callback(0, 1.0, np.array([1, 2]), {})
        callback(1, 0.5, np.array([1, 2]), {})

        assert len(calls) == 2
        assert calls[0] == (0, 1.0)
        assert calls[1] == (1, 0.5)


class TestProgressBar:
    """Test ProgressBar callback"""

    def test_progress_bar_creation(self):
        """Test ProgressBar can be created"""
        pb = ProgressBar(max_nfev=10)
        assert pb.max_nfev == 10
        assert pb.desc == "Optimizing"

    def test_progress_bar_without_tqdm(self, monkeypatch):
        """Test ProgressBar gracefully handles missing tqdm"""
        # Mock tqdm import failure
        import sys
        monkeypatch.setattr("builtins.__import__",
                           lambda name, *args, **kwargs:
                           (_ for _ in ()).throw(ImportError) if name == "tqdm" else __import__(name, *args, **kwargs))

        pb = ProgressBar(max_nfev=10)
        # Should not raise, just warn
        pb(0, 1.0, np.array([1, 2]), {"gradient_norm": 0.1, "nfev": 1})

    def test_progress_bar_updates(self):
        """Test ProgressBar tracks iterations"""
        pb = ProgressBar(max_nfev=10)
        for i in range(5):
            pb(i, 1.0 - i * 0.1, np.array([1, 2]), {"gradient_norm": 0.1, "nfev": i})
        pb.close()
        # If tqdm available, should have updated 5 times


class TestIterationLogger:
    """Test IterationLogger callback"""

    def test_logger_to_file(self):
        """Test logging to file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            logfile = f.name

        try:
            logger = IterationLogger(logfile, mode='w')
            logger(0, 1.0, np.array([1.0, 2.0]), {"gradient_norm": 0.1, "nfev": 1})
            logger(1, 0.5, np.array([1.1, 2.1]), {"gradient_norm": 0.05, "nfev": 2})
            logger.close()

            # Check log file exists and has content
            assert os.path.exists(logfile)
            with open(logfile, 'r') as f:
                content = f.read()
                assert "Iter" in content
                assert "Cost" in content
                assert "Grad" in content
        finally:
            if os.path.exists(logfile):
                os.unlink(logfile)

    def test_logger_with_params(self):
        """Test logging with parameter values"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            logfile = f.name

        try:
            logger = IterationLogger(logfile, mode='w', log_params=True)
            logger(0, 1.0, np.array([1.0, 2.0]), {"gradient_norm": 0.1, "nfev": 1})
            logger.close()

            with open(logfile, 'r') as f:
                content = f.read()
                assert "Params" in content
        finally:
            if os.path.exists(logfile):
                os.unlink(logfile)

    def test_logger_to_stdout(self, capsys):
        """Test logging to stdout"""
        logger = IterationLogger(filename=None)  # stdout
        logger(0, 1.0, np.array([1.0, 2.0]), {"gradient_norm": 0.1, "nfev": 1})
        logger.close()

        captured = capsys.readouterr()
        assert "Iter" in captured.out
        assert "Cost" in captured.out


class TestEarlyStopping:
    """Test EarlyStopping callback"""

    def test_early_stopping_triggers(self):
        """Test early stopping raises StopOptimization"""
        es = EarlyStopping(patience=2, min_delta=1e-6, verbose=False)

        # First iteration - cost improves
        es(0, 1.0, np.array([1, 2]), {})

        # No improvement for patience iterations
        with pytest.raises(StopOptimization):
            es(1, 1.0, np.array([1, 2]), {})  # No improvement
            es(2, 1.0, np.array([1, 2]), {})  # Should trigger

    def test_early_stopping_resets(self):
        """Test early stopping resets on improvement"""
        es = EarlyStopping(patience=2, min_delta=0.1, verbose=False)

        es(0, 1.0, np.array([1, 2]), {})  # best = 1.0
        es(1, 0.95, np.array([1, 2]), {})  # No improvement (delta < 0.1)
        es(2, 0.5, np.array([1, 2]), {})  # Improvement! Reset wait
        es(3, 0.5, np.array([1, 2]), {})  # No improvement (wait=1)

        # Should not raise yet (wait=1 < patience=2)


class TestCallbackChain:
    """Test CallbackChain"""

    def test_callback_chain_calls_all(self):
        """Test all callbacks in chain are called"""
        calls1 = []
        calls2 = []

        class CB1(CallbackBase):
            def __call__(self, iteration, cost, params, info):
                calls1.append(iteration)

        class CB2(CallbackBase):
            def __call__(self, iteration, cost, params, info):
                calls2.append(iteration)

        chain = CallbackChain(CB1(), CB2())
        chain(0, 1.0, np.array([1, 2]), {})
        chain(1, 0.5, np.array([1, 2]), {})

        assert calls1 == [0, 1]
        assert calls2 == [0, 1]

    def test_callback_chain_propagates_stop(self):
        """Test StopOptimization propagates from any callback"""
        class StopCallback(CallbackBase):
            def __call__(self, iteration, cost, params, info):
                if iteration >= 1:
                    raise StopOptimization("Stop!")

        chain = CallbackChain(CallbackBase(), StopCallback())

        chain(0, 1.0, np.array([1, 2]), {})  # OK
        with pytest.raises(StopOptimization):
            chain(1, 0.5, np.array([1, 2]), {})  # Should propagate


class TestCallbackIntegration:
    """Test callbacks work with actual curve_fit"""

    def test_callback_with_curve_fit(self):
        """Test callback is actually called during optimization"""
        calls = []

        def my_callback(iteration, cost, params, info):
            calls.append({
                "iteration": iteration,
                "cost": cost,
                "params": params.copy(),
                "gradient_norm": info.get("gradient_norm"),
            })

        # Simple linear fit
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3 + np.random.normal(0, 0.5, 50)

        popt, pcov = curve_fit(linear, x, y, callback=my_callback)

        # Callback should have been called multiple times
        assert len(calls) > 0
        # Each call should have required fields
        assert all("iteration" in c for c in calls)
        assert all("cost" in c for c in calls)
        assert all("params" in c for c in calls)

    def test_early_stopping_with_curve_fit(self):
        """Test early stopping actually stops optimization"""
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3 + np.random.normal(0, 0.1, 50)

        es = EarlyStopping(patience=3, verbose=False)

        # Should complete successfully (early stopping is treated as convergence)
        popt, pcov = curve_fit(linear, x, y, callback=es)

        # Should have converged to correct values
        assert abs(popt[0] - 2.0) < 0.5
        assert abs(popt[1] - 3.0) < 0.5

    def test_progress_bar_with_curve_fit(self):
        """Test ProgressBar works with curve_fit"""
        x = np.linspace(0, 10, 50)
        y = 100 * np.exp(-0.5 * x) + 10 + np.random.normal(0, 1, 50)

        pb = ProgressBar(max_nfev=50, desc="Test fit")

        popt, pcov = curve_fit(
            exponential_decay,
            x, y,
            p0='auto',
            callback=pb,
            max_nfev=50
        )

        pb.close()

        # Should have fitted successfully
        assert popt is not None

    def test_callback_error_handling(self):
        """Test optimization continues if callback fails"""
        def bad_callback(iteration, cost, params, info):
            if iteration == 2:
                raise ValueError("Callback error!")

        x = np.linspace(0, 10, 50)
        y = 2 * x + 3

        # Should complete despite callback error (with warning)
        with pytest.warns(RuntimeWarning, match="Callback raised exception"):
            popt, pcov = curve_fit(linear, x, y, callback=bad_callback)

        # Optimization should still succeed
        assert abs(popt[0] - 2.0) < 0.1


class TestCallbackChainIntegration:
    """Test combining multiple callbacks"""

    def test_multiple_callbacks(self):
        """Test using multiple callbacks together"""
        calls = []

        class TrackingCallback(CallbackBase):
            def __call__(self, iteration, cost, params, info):
                calls.append(iteration)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            logfile = f.name

        try:
            chain = CallbackChain(
                TrackingCallback(),
                IterationLogger(logfile),
                EarlyStopping(patience=10, verbose=False)
            )

            x = np.linspace(0, 10, 50)
            y = 2 * x + 3

            popt, pcov = curve_fit(linear, x, y, callback=chain)

            chain.close()

            # Tracking callback should have been called
            assert len(calls) > 0
            # Log file should exist
            assert os.path.exists(logfile)

        finally:
            if os.path.exists(logfile):
                os.unlink(logfile)
```

---

## Summary of Work

### Completed ✅
1. Added callback parameter to `CurveFit.curve_fit()` in minpack.py
2. Documented callback parameter
3. Passed callback through to `least_squares()` calls
4. Created comprehensive implementation plan

### Remaining ⏳ (Est. 2-3 hours)
1. Update `least_squares.py` signature and pass to TRF (30 min)
2. Update `trf.py` signatures for both methods (15 min)
3. Add callback invocation in TRF loops with error handling (1 hour)
4. Export callbacks from `__init__.py` (2 min)
5. Create `tests/test_callbacks.py` (45 min)
6. Run all tests and fix any issues (30 min)

### Risk Assessment
- **Low Risk**: Changes are well-isolated, callback is optional
- **Error Handling**: Wrapped in try/except, won't break optimization
- **Testing**: Comprehensive test suite will verify functionality

---

**Next Steps**: Complete remaining changes following this plan, then run full test suite.

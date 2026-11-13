"""
Progress callbacks for monitoring optimization iterations.

This module provides built-in callbacks for monitoring curve_fit optimization
progress, including progress bars, logging, early stopping, and live plotting.

Callbacks are called after each optimization iteration with information about
the current state: iteration number, cost, parameters, gradient norm, etc.
"""

import inspect
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "CallbackBase",
    "CallbackChain",
    "EarlyStopping",
    "IterationLogger",
    "ProgressBar",
    "StopOptimization",
]


class StopOptimization(Exception):
    """Exception raised by callbacks to request early termination."""


_CALLBACK_ARGS = ("iteration", "cost", "params", "info")


def call_user_callback(
    callback: Callable[..., Any],
    iteration: int,
    cost: float,
    params: np.ndarray,
    info: dict[str, Any],
) -> Any:
    """Call ``callback(iteration, cost, params, info)`` honouring its signature.

    Keyword dispatch is used when the four keywords bind to the signature
    (``Signature.bind`` never invokes the callable) and either the positional
    call would not bind (required keyword-only parameters, ``**kwargs`` with
    fewer than four positionals) or the signature names any of the four
    arguments (so ``(params, cost, iteration, info)`` receives the right
    values). Otherwise the arguments are passed positionally (arbitrary names,
    ``*args``, builtins, callables whose signature cannot be inspected).
    A pure ``(*args, **kwargs)`` ``functools.wraps`` wrapper is dispatched by the
    wrapped signature but still called positionally unless the wrapped names
    are in a non-canonical order.
    """
    passthrough = False
    try:
        # follow_wrapped=False: a functools.wraps decorator must not hide the real call shape
        sig = inspect.signature(callback, follow_wrapped=False)
        if hasattr(callback, "__wrapped__") and {
            p.kind for p in sig.parameters.values()
        } == {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            # pure (*args, **kwargs) passthrough: dispatch by the wrapped signature
            sig = inspect.signature(callback)
            passthrough = True
    except (TypeError, ValueError):
        return callback(iteration, cost, params, info)
    try:
        sig.bind(iteration=iteration, cost=cost, params=params, info=info)
    except TypeError:
        return callback(iteration, cost, params, info)
    try:
        sig.bind(iteration, cost, params, info)
        positional_ok = True
    except TypeError:
        positional_ok = False
    if passthrough and positional_ok:
        # Wrappers commonly read args[0] or forward only *args, so keep the
        # positional call unless the wrapped names are in a non-canonical order.
        named = [
            _CALLBACK_ARGS.index(n)
            for n, p in sig.parameters.items()
            if n in _CALLBACK_ARGS
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        use_keywords = named != sorted(named)
    else:
        use_keywords = not positional_ok or any(
            n in sig.parameters for n in _CALLBACK_ARGS
        )
    if use_keywords:
        return callback(iteration=iteration, cost=cost, params=params, info=info)
    return callback(iteration, cost, params, info)


class CallbackBase:
    """Base class for optimization callbacks.

    Subclass this to create custom callbacks. Override the `__call__` method
    to define what happens at each iteration.

    Examples
    --------
    >>> class CustomCallback(CallbackBase):
    ...     def __call__(self, iteration, cost, params, info):
    ...         print(f"Iter {iteration}: cost={cost:.6f}")
    """

    def __call__(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        """Called after each optimization iteration.

        Parameters
        ----------
        iteration : int
            Current iteration number (0-indexed)
        cost : float
            Current cost/objective function value
        params : np.ndarray
            Current parameter values
        info : dict
            Additional information (gradient_norm, nfev, etc.)
        """

    def close(self) -> None:
        """Clean up resources.

        Override this method if your callback uses resources that need
        explicit cleanup (files, network connections, etc.).
        """


class ProgressBar(CallbackBase):
    """Progress bar callback using tqdm.

    Displays a progress bar showing optimization progress with current cost,
    gradient norm, and iteration statistics.

    Parameters
    ----------
    max_nfev : int, optional
        Maximum number of function evaluations. If provided, progress bar
        will be based on nfev. Otherwise, shows indefinite progress.
    desc : str, optional
        Description to display in progress bar. Default: "Optimizing"
    **tqdm_kwargs
        Additional keyword arguments passed to tqdm

    Examples
    --------
    >>> from nlsq import curve_fit
    >>> from nlsq.callbacks import ProgressBar
    >>> callback = ProgressBar(max_nfev=100)
    >>> popt, pcov = curve_fit(f, x, y, callback=callback)
    """

    def __init__(
        self,
        max_nfev: int | None = None,
        desc: str = "Optimizing",
        **tqdm_kwargs,
    ):
        self.max_nfev = max_nfev
        self.desc = desc
        self.tqdm_kwargs = tqdm_kwargs
        self._pbar = None
        self._last_nfev = 0

        # Try to import tqdm
        try:
            from tqdm.auto import tqdm  # type: ignore[import-untyped]

            self.tqdm = tqdm
        except ImportError:
            self.tqdm = None
            warnings.warn(
                "tqdm not installed. Install with 'pip install tqdm' "
                "to use ProgressBar callback.",
                UserWarning,
                stacklevel=2,
            )

    def __call__(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        """Update progress bar."""
        if self.tqdm is None:
            return

        # Initialize progress bar on first call
        if self._pbar is None:
            self._pbar = self.tqdm(
                total=self.max_nfev,
                desc=self.desc,
                **self.tqdm_kwargs,
            )

        # Update progress
        nfev = info.get("nfev", iteration + 1)
        delta_nfev = nfev - self._last_nfev
        self._last_nfev = nfev

        # Update postfix with current status
        g_norm = info.get("gradient_norm", np.nan)
        assert self._pbar is not None  # guaranteed by check above
        self._pbar.set_postfix(
            {
                "cost": f"{cost:.6e}",
                "grad": f"{g_norm:.3e}",
                "iter": iteration,
            },
        )

        # Update progress bar
        self._pbar.update(delta_nfev if self.max_nfev else 1)

    def close(self):
        """Close progress bar."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        self._last_nfev = 0

    def __del__(self):
        """Ensure progress bar is closed on deletion."""
        self.close()


class IterationLogger(CallbackBase):
    """Log optimization progress to file or stdout.

    Parameters
    ----------
    filename : str, optional
        File to log to. If None and file is None, logs to stdout.
    mode : str, optional
        File open mode. Default: 'w' (overwrite)
    log_params : bool, optional
        Whether to log parameter values. Default: False
    file : file-like object, optional
        File-like object to write to. If provided, filename is ignored.

    Examples
    --------
    >>> from nlsq.callbacks import IterationLogger
    >>> callback = IterationLogger("optimization.log")
    >>> popt, pcov = curve_fit(f, x, y, callback=callback)
    """

    def __init__(
        self,
        filename: str | None = None,
        mode: str = "w",
        log_params: bool = False,
        file: Any | None = None,
    ):
        self.filename = filename
        self.mode = mode
        self.log_params = log_params
        self._file = file  # Use provided file object if given
        self._file_provided = file is not None  # Track if file was provided externally
        self._start_time = None
        self._lock = threading.RLock()

    def __deepcopy__(self, memo: dict) -> "IterationLogger":
        # Write-only sink: copies would reopen (and truncate) the same file.
        return self

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state["_file"] = None  # handles are not picklable; reopened lazily
        state["_start_time"] = None
        if self.filename is not None and state["mode"].startswith("w"):
            state["mode"] = "a"  # an unpickled copy must not truncate the log
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def __call__(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        """Log iteration information."""
        with self._lock:
            self._log(iteration, cost, params, info)

    def _log(self, iteration, cost, params, info) -> None:
        if self._start_time is None:
            self._start_time = time.time()
            self._open_file()
            self._write_header()

        # Compute elapsed time
        elapsed = time.time() - self._start_time if self._start_time else 0

        # Build log message
        g_norm = info.get("gradient_norm", np.nan)
        nfev = info.get("nfev", iteration + 1)

        msg = (
            f"Iter {iteration:4d} | "
            f"Cost: {cost:.6e} | "
            f"Grad: {g_norm:.3e} | "
            f"NFev: {nfev:4d} | "
            f"Time: {elapsed:.2f}s"
        )

        if self.log_params:
            params_str = np.array2string(
                params,
                precision=6,
                separator=", ",
                suppress_small=True,
            )
            msg += f" | Params: {params_str}"

        self._write(msg)

    def _open_file(self):
        """Open log file."""
        # Only open file if not already provided and filename is given
        if not self._file_provided and self.filename is not None:
            self._file = open(self.filename, self.mode)  # noqa: SIM115

    def _write_header(self):
        """Write log header."""
        header = "=" * 80 + "\n"
        header += "NLSQ Optimization Log\n"
        header += f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "=" * 80 + "\n"
        self._write(header)

    def _write(self, msg: str):
        """Write message to log."""
        if self._file is not None:
            self._file.write(msg + "\n")
            self._file.flush()
        else:
            print(msg)

    def close(self):
        """Close log file."""
        if self._file is not None:
            elapsed = time.time() - self._start_time if self._start_time else 0
            footer = "=" * 80 + "\n"
            footer += f"Optimization completed in {elapsed:.2f}s\n"
            footer += "=" * 80 + "\n"
            self._write(footer)
            # Only close file if we opened it (not if it was provided externally)
            if not self._file_provided:
                self._file.close()
                self._file = None
        self._start_time = None

    def __del__(self):
        """Ensure file is closed on deletion."""
        self.close()


class EarlyStopping(CallbackBase):
    """Stop optimization early if no improvement for patience iterations.

    Parameters
    ----------
    patience : int, optional
        Number of iterations with no improvement to wait before stopping.
        Default: 10
    min_delta : float, optional
        Minimum change in cost to qualify as an improvement. Default: 1e-6
    verbose : bool, optional
        Whether to print messages. Default: True

    Examples
    --------
    >>> from nlsq.callbacks import EarlyStopping
    >>> callback = EarlyStopping(patience=5, min_delta=1e-4)
    >>> popt, pcov = curve_fit(f, x, y, callback=callback)

    Notes
    -----
    Raises StopOptimization exception when patience is exceeded, which
    will be caught by the optimizer, which reports ``status=-2`` (as
    SciPy does) and returns the best-so-far parameters; ``success`` stays True
    so ``curve_fit`` does not raise. State
    is reset automatically when a new fit starts, so one instance can be reused.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_cost = np.inf
        self.wait = 0
        self._last_iter = -1

    def __call__(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        """Check for improvement and stop if patience exceeded."""
        # Iteration numbers restart each fit, so a non-increasing one means a new fit
        if iteration <= self._last_iter:
            self.best_cost = np.inf
            self.wait = 0
        self._last_iter = iteration
        # NaN cost must not count as improvement or patience increment
        if not np.isfinite(cost):
            return
        # Check if we have improvement
        if cost < self.best_cost - self.min_delta:
            self.best_cost = cost
            self.wait = 0
        else:
            self.wait += 1

        # Stop if patience exceeded
        if self.wait >= self.patience:
            if self.verbose:
                print(
                    f"\nEarly stopping triggered at iteration {iteration}. "
                    f"No improvement for {self.patience} iterations.",
                )
            raise StopOptimization(
                f"Early stopping after {self.patience} iterations without improvement",
            )


class CallbackChain(CallbackBase):
    """Chain multiple callbacks together.

    Calls each callback in sequence. If any callback raises StopOptimization,
    propagates it to stop the optimization.

    Parameters
    ----------
    *callbacks : CallbackBase
        Callbacks to chain together

    Examples
    --------
    >>> from nlsq.callbacks import CallbackChain, ProgressBar, EarlyStopping
    >>> callback = CallbackChain(
    ...     ProgressBar(max_nfev=100),
    ...     EarlyStopping(patience=5)
    ... )
    >>> popt, pcov = curve_fit(f, x, y, callback=callback)
    """

    def __init__(self, *callbacks: CallbackBase):
        self.callbacks = list(callbacks)

    def __call__(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        """Call all callbacks in sequence."""
        for callback in self.callbacks:
            call_user_callback(callback, iteration, cost, params, info)

    def close(self):
        """Close all callbacks that have a close method."""
        for callback in self.callbacks:
            if hasattr(callback, "close"):
                callback.close()

    def __del__(self):
        """Ensure all callbacks are closed."""
        self.close()

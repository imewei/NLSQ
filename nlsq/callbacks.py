"""
Progress callbacks for monitoring optimization iterations.

This module provides built-in callbacks for monitoring curve_fit optimization
progress, including progress bars, logging, early stopping, and live plotting.

Callbacks are called after each optimization iteration with information about
the current state: iteration number, cost, parameters, gradient norm, etc.
"""

import time
import warnings
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
        self._pbar.set_postfix(  # type: ignore[attr-defined]
            {
                "cost": f"{cost:.6e}",
                "grad": f"{g_norm:.3e}",
                "iter": iteration,
            }
        )

        # Update progress bar
        self._pbar.update(delta_nfev if self.max_nfev else 1)  # type: ignore[attr-defined]

    def close(self):
        """Close progress bar."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

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

    def __call__(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        """Log iteration information."""
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
                params, precision=6, separator=", ", suppress_small=True
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
    will be caught by the optimizer and treated as successful convergence.
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

    def __call__(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        """Check for improvement and stop if patience exceeded."""
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
                    f"No improvement for {self.patience} iterations."
                )
            raise StopOptimization(
                f"Early stopping after {self.patience} iterations without improvement"
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
            callback(iteration, cost, params, info)

    def close(self):
        """Close all callbacks that have a close method."""
        for callback in self.callbacks:
            if hasattr(callback, "close"):
                callback.close()

    def __del__(self):
        """Ensure all callbacks are closed."""
        self.close()

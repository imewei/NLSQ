"""Comprehensive logging system for NLSQ package.

This module provides structured logging for monitoring package operations,
error tracking, and debugging. Key features:

- Operation tracking with unique IDs for tracing
- Structured logging with consistent formats
- Performance metrics and timing
- Memory usage monitoring
- Error context with actionable suggestions

Environment Variables
---------------------
NLSQ_DEBUG : str
    Set to "1" to enable debug mode with detailed logging
NLSQ_VERBOSE : str
    Set to "1" to enable verbose mode (INFO level)
NLSQ_LOG_DIR : str
    Directory for debug log files (default: current directory)
NLSQ_TRACE_JAX : str
    Set to "1" to trace JAX compilation events
NLSQ_SAVE_ITERATIONS : str
    Directory to save optimization iteration history

Example
-------
>>> from nlsq.utils.logging import get_logger
>>> logger = get_logger("my_module")
>>> with logger.operation("curve_fit", n_points=1000, n_params=3):
...     # Your fitting code here
...     logger.info("Fitting completed successfully")
"""

import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from threading import local
from typing import Any

import numpy as np


class LogLevel(IntEnum):
    """Custom log levels for NLSQ."""

    DEBUG = logging.DEBUG  # 10
    INFO = logging.INFO  # 20
    PERFORMANCE = 25  # Custom level for performance logs
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR  # 40
    CRITICAL = logging.CRITICAL  # 50


# Thread-local storage for operation context
_context = local()


def _get_operation_id() -> str | None:
    """Get current operation ID from thread-local context."""
    return getattr(_context, "operation_id", None)


def _format_kwargs(kwargs: dict[str, Any]) -> str:
    """Format kwargs for log message, handling special types."""
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, float):
            if abs(v) < 1e-3 or abs(v) > 1e4:
                parts.append(f"{k}={v:.4e}")
            else:
                parts.append(f"{k}={v:.4f}")
        elif isinstance(v, (list, tuple)) and len(v) > 5:
            parts.append(f"{k}=[{len(v)} items]")
        elif isinstance(v, np.ndarray):
            parts.append(f"{k}=array{v.shape}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


class NLSQLogger:
    """Comprehensive logger for NLSQ optimization routines.

    Features:
    - Operation tracking with unique IDs
    - Structured logging with consistent formats
    - Performance tracking and timing
    - Memory usage monitoring
    - JAX compilation event logging
    - Debug mode with detailed tracing

    Examples
    --------
    >>> logger = NLSQLogger("curve_fit")
    >>> with logger.operation("fit", dataset_size=10000):
    ...     logger.info("Starting optimization")
    ...     # ... fitting code ...
    ...     logger.fit_complete(iterations=50, final_cost=1.2e-6)
    """

    def __init__(self, name: str, level: int | LogLevel = LogLevel.INFO):
        """Initialize NLSQ logger.

        Parameters
        ----------
        name : str
            Logger name, typically the module name
        level : int | LogLevel
            Initial logging level
        """
        self.name = f"nlsq.{name}" if not name.startswith("nlsq.") else name
        self.logger = logging.getLogger(self.name)

        # Override level for debug mode
        debug_mode = os.getenv("NLSQ_DEBUG", "0") == "1"
        if debug_mode:
            level = LogLevel.DEBUG

        self.logger.setLevel(level)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

        # Performance tracking
        self.timers: dict[str, float] = {}

        # Optimization tracking
        self.optimization_history: list[dict[str, Any]] = []

        # Register custom log level
        if not hasattr(logging, "PERFORMANCE"):
            logging.addLevelName(LogLevel.PERFORMANCE, "PERFORMANCE")

    def _setup_handlers(self):
        """Setup console and optional file handlers."""
        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)

        # Check for debug mode
        debug_mode = os.getenv("NLSQ_DEBUG", "0") == "1"
        verbose_mode = os.getenv("NLSQ_VERBOSE", "0") == "1"

        if debug_mode:
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
                datefmt="%H:%M:%S",
            )
        elif verbose_mode:
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        else:
            console_handler.setLevel(logging.WARNING)
            formatter = logging.Formatter("[%(levelname)s] %(message)s")

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Optional file handler for debug mode
        if debug_mode:
            log_dir = Path(os.getenv("NLSQ_LOG_DIR", "."))
            log_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"nlsq_debug_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"Debug logging enabled: {log_file}")

    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with operation context and kwargs."""
        parts = [message]

        # Add operation ID if present
        op_id = _get_operation_id()
        if op_id:
            parts.insert(0, f"[op:{op_id[:8]}]")

        # Add structured kwargs
        if kwargs:
            parts.append(_format_kwargs(kwargs))

        return " ".join(parts)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self.logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self.logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self.logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message with optional exception info."""
        self.logger.error(self._format_message(message, **kwargs), exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Log critical error with exception info."""
        self.logger.critical(self._format_message(message, **kwargs), exc_info=exc_info)

    def performance(self, message: str, **kwargs):
        """Log performance-related message."""
        self.logger.log(LogLevel.PERFORMANCE, self._format_message(message, **kwargs))

    @contextmanager
    def operation(self, name: str, **context):
        """Context manager for tracking operations with unique IDs.

        Provides operation-level context for all log messages within the block,
        including timing and memory usage tracking.

        Parameters
        ----------
        name : str
            Name of the operation (e.g., "curve_fit", "jacobian")
        **context
            Additional context to log (e.g., n_points, n_params)

        Examples
        --------
        >>> with logger.operation("curve_fit", n_points=10000, n_params=5):
        ...     # All logs within this block include operation context
        ...     logger.info("Starting optimization")
        """
        op_id = uuid.uuid4().hex
        _context.operation_id = op_id
        start_time = time.perf_counter()

        # Log operation start
        context_str = _format_kwargs(context) if context else ""
        self.info(f"START {name} | {context_str}" if context_str else f"START {name}")

        try:
            yield op_id
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self.error(
                f"FAILED {name} after {elapsed:.3f}s: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            self.timers[f"{name}_{op_id[:8]}"] = elapsed
            self.info(f"END {name}", elapsed=f"{elapsed:.3f}s")
            _context.operation_id = None

    def fit_start(
        self,
        n_points: int,
        n_params: int,
        method: str = "trf",
        bounds: str = "none",
        **kwargs,
    ):
        """Log the start of a fitting operation.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_params : int
            Number of parameters to fit
        method : str
            Optimization method
        bounds : str
            Bounds type ("none", "bounded", "semi-bounded")
        """
        self.info(
            "Fit started",
            n_points=n_points,
            n_params=n_params,
            method=method,
            bounds=bounds,
            **kwargs,
        )

    def fit_complete(
        self,
        success: bool = True,
        iterations: int | None = None,
        final_cost: float | None = None,
        termination: str | None = None,
        **kwargs,
    ):
        """Log completion of a fitting operation.

        Parameters
        ----------
        success : bool
            Whether the fit converged successfully
        iterations : int, optional
            Number of iterations taken
        final_cost : float, optional
            Final cost/residual value
        termination : str, optional
            Termination reason
        """
        status = "SUCCESS" if success else "FAILED"
        metrics: dict[str, str | int | float] = {"status": status}
        if iterations is not None:
            metrics["iterations"] = iterations
        if final_cost is not None:
            metrics["final_cost"] = final_cost
        if termination:
            metrics["termination"] = termination
        metrics.update(kwargs)

        if success:
            self.info("Fit complete", **metrics)
        else:
            self.warning("Fit incomplete", **metrics)

    def optimization_step(
        self,
        iteration: int,
        cost: float,
        gradient_norm: float | None = None,
        step_size: float | None = None,
        nfev: int | None = None,
        **kwargs,
    ):
        """Log optimization iteration details.

        Parameters
        ----------
        iteration : int
            Current iteration number
        cost : float
            Current cost/loss value
        gradient_norm : float, optional
            Norm of the gradient
        step_size : float, optional
            Size of the step taken
        nfev : int, optional
            Number of function evaluations
        **kwargs
            Additional metrics to log
        """
        metrics: dict[str, Any] = {
            "iter": iteration,
            "cost": cost,
        }

        if gradient_norm is not None:
            metrics["grad_norm"] = gradient_norm

        if step_size is not None:
            metrics["step"] = step_size

        if nfev is not None:
            metrics["nfev"] = nfev

        metrics.update(kwargs)

        # Store in history
        self.optimization_history.append({"timestamp": time.time(), **metrics})

        # Format and log
        self.performance("Iteration", **metrics)

    def convergence(
        self,
        reason: str,
        iterations: int,
        final_cost: float,
        time_elapsed: float | None = None,
        **kwargs,
    ):
        """Log convergence information.

        Parameters
        ----------
        reason : str
            Reason for convergence
        iterations : int
            Total iterations
        final_cost : float
            Final cost value
        time_elapsed : float, optional
            Total time taken
        **kwargs
            Additional convergence metrics
        """
        metrics = {
            "reason": reason,
            "iterations": iterations,
            "final_cost": final_cost,
        }

        if time_elapsed is not None:
            metrics["elapsed"] = f"{time_elapsed:.3f}s"

        metrics.update(kwargs)
        self.info("Convergence", **metrics)

    def numerical_issue(
        self,
        issue_type: str,
        details: str,
        suggestion: str | None = None,
        **kwargs,
    ):
        """Log numerical issues with actionable suggestions.

        Parameters
        ----------
        issue_type : str
            Type of issue (e.g., "ill-conditioned", "overflow", "nan")
        details : str
            Description of the issue
        suggestion : str, optional
            Suggested fix or action
        """
        msg = f"Numerical issue ({issue_type}): {details}"
        if suggestion:
            msg += f" | Suggestion: {suggestion}"
        self.warning(msg, **kwargs)

    def memory_usage(self, label: str = "current"):
        """Log current memory usage.

        Parameters
        ----------
        label : str
            Label for this memory checkpoint
        """
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / (1024**3)
            self.debug(f"Memory ({label})", rss_gb=mem_gb)
        except ImportError:
            pass  # psutil not available

    def jax_compilation(
        self,
        function_name: str,
        input_shape: tuple | None = None,
        compilation_time: float | None = None,
        **kwargs,
    ):
        """Log JAX compilation events.

        Parameters
        ----------
        function_name : str
            Name of function being compiled
        input_shape : tuple, optional
            Shape of input data
        compilation_time : float, optional
            Time taken to compile
        **kwargs
            Additional compilation details
        """
        if os.getenv("NLSQ_TRACE_JAX") != "1":
            return

        metrics = {"function": function_name}

        if input_shape is not None:
            metrics["shape"] = str(input_shape)

        if compilation_time is not None:
            metrics["time"] = f"{compilation_time:.3f}s"

        metrics.update(kwargs)
        self.debug("JAX compilation", **metrics)

    @contextmanager
    def timer(self, name: str, log_result: bool = True):
        """Context manager for timing code sections.

        Parameters
        ----------
        name : str
            Name of the timed section
        log_result : bool
            Whether to log the timing result

        Examples
        --------
        >>> with logger.timer('jacobian_computation'):
        ...     J = compute_jacobian(x)
        """
        start_time = time.perf_counter()
        self.timers[name] = start_time

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.timers[name] = elapsed

            if log_result:
                self.performance(f"Timer: {name}", elapsed=f"{elapsed:.6f}s")

    def matrix_info(
        self, name: str, matrix: np.ndarray, compute_condition: bool = False
    ):
        """Log information about a matrix.

        Parameters
        ----------
        name : str
            Name of the matrix
        matrix : np.ndarray
            The matrix to analyze
        compute_condition : bool
            Whether to compute condition number (expensive)
        """
        info: dict[str, Any] = {
            "shape": matrix.shape,
            "dtype": str(matrix.dtype),
            "range": f"[{np.min(matrix):.2e}, {np.max(matrix):.2e}]",
        }

        if compute_condition and matrix.ndim == 2:
            try:
                cond = np.linalg.cond(matrix)
                info["condition"] = cond
                if cond > 1e10:
                    self.warning(
                        f"Matrix {name} is ill-conditioned",
                        condition=cond,
                        suggestion="Consider rescaling parameters or data",
                    )
            except (np.linalg.LinAlgError, ValueError):
                info["condition"] = "failed"

        self.debug(f"Matrix {name}", **info)

    def data_summary(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray | None = None,
    ):
        """Log summary statistics of input data.

        Parameters
        ----------
        x : np.ndarray
            Independent variable data
        y : np.ndarray
            Dependent variable data
        sigma : np.ndarray, optional
            Uncertainty/weights
        """
        info = {
            "n_points": len(x),
            "x_range": f"[{np.min(x):.4g}, {np.max(x):.4g}]",
            "y_range": f"[{np.min(y):.4g}, {np.max(y):.4g}]",
        }

        # Check for potential issues
        if np.any(~np.isfinite(x)):
            self.warning("Input x contains non-finite values")
        if np.any(~np.isfinite(y)):
            self.warning("Input y contains non-finite values")

        if sigma is not None:
            info["sigma_range"] = f"[{np.min(sigma):.4g}, {np.max(sigma):.4g}]"
            if np.any(sigma <= 0):
                self.warning("Sigma contains non-positive values")

        self.debug("Data summary", **info)

    def parameter_update(
        self,
        params: np.ndarray,
        param_names: list[str] | None = None,
    ):
        """Log parameter values during optimization.

        Parameters
        ----------
        params : np.ndarray
            Current parameter values
        param_names : list[str], optional
            Names for each parameter
        """
        if param_names and len(param_names) == len(params):
            param_dict = dict(zip(param_names, params, strict=True))
            self.debug("Parameters", **param_dict)
        else:
            self.debug("Parameters", values=params.tolist())

    def save_iteration_data(self, output_dir: str | None = None):
        """Save optimization history to file.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save data. Uses NLSQ_SAVE_ITERATIONS env var if not provided.
        """
        if not self.optimization_history:
            return

        save_dir = output_dir or os.getenv("NLSQ_SAVE_ITERATIONS")
        if not save_dir:
            return

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_path / f"optimization_history_{self.name}_{timestamp}.npz"

        # Convert history to arrays
        data: dict[str, np.ndarray] = {}
        for key in self.optimization_history[0]:
            values: list[float] = []
            for entry in self.optimization_history:
                val = entry.get(key, np.nan)
                # Handle string values
                if isinstance(val, str):
                    try:
                        val = float(val.replace("e", "E"))
                    except (ValueError, AttributeError):
                        val = np.nan
                values.append(val)
            data[key] = np.array(values)

        np.savez(file=filename, **data)  # type: ignore[arg-type]
        self.info(f"Saved optimization history to {filename}")


# Module-level convenience functions
_loggers: dict[str, NLSQLogger] = {}


def get_logger(name: str, level: int | LogLevel = LogLevel.INFO) -> NLSQLogger:
    """Get or create a logger for the given name.

    Parameters
    ----------
    name : str
        Logger name (will be prefixed with "nlsq." if not already)
    level : int | LogLevel
        Logging level

    Returns
    -------
    NLSQLogger
        Logger instance

    Examples
    --------
    >>> logger = get_logger("my_module")
    >>> logger.info("Processing started", n_items=100)
    """
    if name not in _loggers:
        _loggers[name] = NLSQLogger(name, level)
    return _loggers[name]


def set_global_level(level: int | LogLevel):
    """Set logging level for all NLSQ loggers.

    Parameters
    ----------
    level : int | LogLevel
        New logging level
    """
    for logger in _loggers.values():
        logger.logger.setLevel(level)

    # Also set for root NLSQ logger
    root_logger = logging.getLogger("nlsq")
    root_logger.setLevel(level)


def enable_debug_mode():
    """Enable debug mode with detailed logging.

    Sets NLSQ_DEBUG=1 and configures all loggers for DEBUG level output.
    """
    os.environ["NLSQ_DEBUG"] = "1"
    set_global_level(LogLevel.DEBUG)

    # Recreate handlers for existing loggers
    for logger in _loggers.values():
        logger.logger.handlers.clear()
        logger._setup_handlers()


def enable_verbose_mode():
    """Enable verbose mode with INFO level logging.

    Sets NLSQ_VERBOSE=1 for detailed operational messages.
    """
    os.environ["NLSQ_VERBOSE"] = "1"
    set_global_level(LogLevel.INFO)


def enable_performance_tracking():
    """Enable performance tracking mode.

    Enables JAX tracing and iteration history saving.
    """
    os.environ["NLSQ_TRACE_JAX"] = "1"
    os.environ["NLSQ_SAVE_ITERATIONS"] = "1"
    set_global_level(LogLevel.PERFORMANCE)

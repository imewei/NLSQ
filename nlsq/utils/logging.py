"""Comprehensive logging system for NLSQ package."""

import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from enum import IntEnum
from pathlib import Path

import numpy as np


class LogLevel(IntEnum):
    """Custom log levels for NLSQ."""

    DEBUG = logging.DEBUG  # 10
    INFO = logging.INFO  # 20
    PERFORMANCE = 25  # Custom level for performance logs
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR  # 40
    CRITICAL = logging.CRITICAL  # 50


class NLSQLogger:
    """Comprehensive logger for NLSQ optimization routines.

    Features:
    - Structured logging with different levels
    - Performance tracking
    - Optimization step monitoring
    - JAX compilation event logging
    - Debug mode with detailed tracing
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
        self.name = f"nlsq.{name}"
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
        self.optimization_history: list = []

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
                "%(asctime)s - [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
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
                "%(asctime)s - [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"Debug logging enabled. Log file: {log_file}")

    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.debug(message)

    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.info(message)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message with optional exception info."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Log critical error with exception info."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.critical(message, exc_info=exc_info)

    def performance(self, message: str, **kwargs):
        """Log performance-related message."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.log(LogLevel.PERFORMANCE, message)

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
        metrics = {
            "iter": iteration,
            "cost": f"{cost:.6e}",
        }

        if gradient_norm is not None:
            metrics["‖∇f‖"] = f"{gradient_norm:.6e}"

        if step_size is not None:
            metrics["step"] = f"{step_size:.6e}"

        if nfev is not None:
            metrics["nfev"] = nfev

        metrics.update(kwargs)

        # Store in history
        self.optimization_history.append({"timestamp": time.time(), **metrics})

        # Format message
        message = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self.performance(f"Optimization: {message}")

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
            "final_cost": f"{final_cost:.6e}",
        }

        if time_elapsed is not None:
            metrics["time"] = f"{time_elapsed:.3f}s"

        metrics.update(kwargs)

        message = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self.info(f"Convergence: {message}")

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

        message = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self.debug(f"JAX Compilation: {message}")

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
                self.performance(f"Timer: {name} took {elapsed:.6f}s")

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
        info = {
            "shape": matrix.shape,
            "dtype": str(matrix.dtype),
            "min": f"{np.min(matrix):.6e}",
            "max": f"{np.max(matrix):.6e}",
            "mean": f"{np.mean(matrix):.6e}",
        }

        if compute_condition and matrix.ndim == 2:
            try:
                cond = np.linalg.cond(matrix)
                info["condition"] = f"{cond:.2e}"
            except (np.linalg.LinAlgError, ValueError):
                info["condition"] = "failed"

        message = " | ".join(f"{k}={v}" for k, v in info.items())
        self.debug(f"Matrix {name}: {message}")

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
        data = {}
        for key in self.optimization_history[0]:
            values = []
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

        np.savez(filename, **data)
        self.info(f"Saved optimization history to {filename}")


# Module-level convenience functions
_loggers: dict[str, NLSQLogger] = {}


def get_logger(name: str, level: int | LogLevel = LogLevel.INFO) -> NLSQLogger:
    """Get or create a logger for the given name.

    Parameters
    ----------
    name : str
        Logger name
    level : int | LogLevel
        Logging level

    Returns
    -------
    NLSQLogger
        Logger instance
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
    """Enable debug mode with detailed logging."""
    os.environ["NLSQ_DEBUG"] = "1"
    set_global_level(LogLevel.DEBUG)

    # Recreate handlers for existing loggers
    for logger in _loggers.values():
        logger.logger.handlers.clear()
        logger._setup_handlers()


def enable_performance_tracking():
    """Enable performance tracking mode."""
    os.environ["NLSQ_TRACE_JAX"] = "1"
    os.environ["NLSQ_SAVE_ITERATIONS"] = "1"
    set_global_level(LogLevel.PERFORMANCE)


# Usage examples - see module docstring for details

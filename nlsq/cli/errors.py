"""CLI-specific exceptions and logging infrastructure for NLSQ.

This module provides:
- CLI-specific exception classes with actionable error messages
- Dual logging system (file + console)
- Structured JSON logging support
- Log rotation support
- Verbosity level control

Exception Hierarchy
-------------------

- CLIError (base)

  - ConfigError (YAML configuration issues)
  - DataLoadError (data file loading failures)
  - ModelError (model resolution issues)
  - FitError (curve fitting failures)

Logging System
--------------
The dual logging system provides:
- File logging: Python logging module to configurable log file
- Console logging: Formatted output with colored severity levels
- Structured logging: JSON format for external tool ingestion
- Log rotation: Automatic rotation with configurable size and backup count

Verbosity Levels
----------------
- 0: Silent (errors only)
- 1: Progress (default, warnings and progress messages)
- 2: Detailed (info-level messages)
- 3: Debug (debug-level messages)
"""

import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar

# =============================================================================
# ANSI Color Codes for Console Output
# =============================================================================


class Colors:
    """ANSI color codes for console output."""

    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"

    @classmethod
    def is_terminal(cls) -> bool:
        """Check if stdout is a terminal (supports colors)."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# =============================================================================
# CLI-Specific Exception Classes
# =============================================================================


class CLIError(Exception):
    """Base exception class for NLSQ CLI errors.

    All CLI-specific exceptions inherit from this class, allowing users
    to catch all CLI errors with a single except clause.

    Attributes
    ----------
    message : str
        Human-readable error message.
    context : dict
        Additional context information for debugging.
    suggestion : str, optional
        Actionable suggestion for resolving the error.
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ):
        """Initialize CLIError.

        Parameters
        ----------
        message : str
            Human-readable error message.
        context : dict, optional
            Additional context information for debugging.
        suggestion : str, optional
            Actionable suggestion for resolving the error.
        """
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context and suggestion."""
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "suggestion": self.suggestion,
        }


class ConfigError(CLIError):
    """Exception raised for YAML configuration issues.

    Raised when:
    - YAML file cannot be parsed (syntax errors)
    - Required configuration keys are missing
    - Configuration values are invalid
    - Configuration file does not exist

    Examples
    --------
    >>> raise ConfigError(
    ...     "Missing required key 'data.input_file'",
    ...     context={"config_file": "workflow.yaml"},
    ...     suggestion="Add 'input_file' under the 'data' section"
    ... )
    """

    def __init__(
        self,
        message: str,
        config_file: str | Path | None = None,
        key: str | None = None,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ):
        """Initialize ConfigError.

        Parameters
        ----------
        message : str
            Human-readable error message.
        config_file : str or Path, optional
            Path to the configuration file.
        key : str, optional
            The configuration key that caused the error.
        context : dict, optional
            Additional context information.
        suggestion : str, optional
            Actionable suggestion for resolving the error.
        """
        ctx = context or {}
        if config_file is not None:
            ctx["config_file"] = str(config_file)
        if key is not None:
            ctx["key"] = key
        super().__init__(message, context=ctx, suggestion=suggestion)


class DataLoadError(CLIError):
    """Exception raised for data file loading failures.

    Raised when:
    - Data file does not exist
    - Data file format cannot be detected
    - Data file cannot be parsed
    - Required columns are missing
    - Data contains invalid values (NaN/Inf when not allowed)

    Examples
    --------
    >>> raise DataLoadError(
    ...     "Column 'time' not found in data.csv",
    ...     file_path="data/experiment.csv",
    ...     context={"available_columns": ["x", "y", "sigma"]},
    ...     suggestion="Use one of the available columns: x, y, sigma"
    ... )
    """

    def __init__(
        self,
        message: str,
        file_path: str | Path | None = None,
        file_format: str | None = None,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ):
        """Initialize DataLoadError.

        Parameters
        ----------
        message : str
            Human-readable error message.
        file_path : str or Path, optional
            Path to the data file.
        file_format : str, optional
            Expected or detected file format.
        context : dict, optional
            Additional context information.
        suggestion : str, optional
            Actionable suggestion for resolving the error.
        """
        ctx = context or {}
        if file_path is not None:
            ctx["file_path"] = str(file_path)
        if file_format is not None:
            ctx["file_format"] = file_format
        super().__init__(message, context=ctx, suggestion=suggestion)


class ModelError(CLIError):
    """Exception raised for model resolution issues.

    Raised when:
    - Builtin model name is not recognized
    - Custom model file does not exist
    - Custom model function cannot be found
    - Model function signature is invalid
    - Polynomial degree is invalid

    Examples
    --------
    >>> raise ModelError(
    ...     "Model 'exponential_growth' not found in builtin models",
    ...     model_name="exponential_growth",
    ...     context={"available_models": ["linear", "exponential_decay", "gaussian"]},
    ...     suggestion="Did you mean 'exponential_decay'?"
    ... )
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        model_type: str | None = None,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ):
        """Initialize ModelError.

        Parameters
        ----------
        message : str
            Human-readable error message.
        model_name : str, optional
            Name of the model that caused the error.
        model_type : str, optional
            Type of model (builtin, custom, polynomial).
        context : dict, optional
            Additional context information.
        suggestion : str, optional
            Actionable suggestion for resolving the error.
        """
        ctx = context or {}
        if model_name is not None:
            ctx["model_name"] = model_name
        if model_type is not None:
            ctx["model_type"] = model_type
        super().__init__(message, context=ctx, suggestion=suggestion)


class FitError(CLIError):
    """Exception raised for curve fitting failures.

    Raised when:
    - curve_fit() fails to converge
    - Covariance matrix cannot be estimated
    - Fit produces invalid results (NaN/Inf)
    - Maximum iterations exceeded

    Examples
    --------
    >>> raise FitError(
    ...     "Curve fitting failed to converge",
    ...     context={"iterations": 1000, "final_cost": 1e10},
    ...     suggestion="Try different initial parameters or relax tolerances"
    ... )
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ):
        """Initialize FitError.

        Parameters
        ----------
        message : str
            Human-readable error message.
        model_name : str, optional
            Name of the model being fitted.
        context : dict, optional
            Additional context information (iterations, cost, etc.).
        suggestion : str, optional
            Actionable suggestion for resolving the error.
        """
        ctx = context or {}
        if model_name is not None:
            ctx["model_name"] = model_name
        super().__init__(message, context=ctx, suggestion=suggestion)


# =============================================================================
# Logging Infrastructure
# =============================================================================


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs log records as single-line JSON objects for easy parsing
    by external tools (log aggregators, monitoring systems, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            JSON-formatted log line.
        """
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key in ("context", "file_path", "config_file", "model_name"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        return json.dumps(log_data, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with colored severity levels.

    Applies ANSI color codes based on log level for better
    visibility in terminal output.
    """

    LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }

    def __init__(self, use_colors: bool = True):
        """Initialize ColoredConsoleFormatter.

        Parameters
        ----------
        use_colors : bool
            Whether to use ANSI color codes.
        """
        super().__init__(
            fmt="%(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_colors = use_colors and Colors.is_terminal()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            Formatted log line.
        """
        message = super().format(record)

        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, "")
            if color:
                message = f"{color}{message}{Colors.RESET}"

        return message


class CLILogger:
    """Logger wrapper providing dual logging (file + console).

    This class manages the logging infrastructure for the CLI,
    supporting both file and console output with configurable
    formats and verbosity levels.
    """

    def __init__(self, name: str = "nlsq.cli"):
        """Initialize CLILogger.

        Parameters
        ----------
        name : str
            Logger name.
        """
        self.logger = logging.getLogger(name)
        self._file_handler: logging.Handler | None = None
        self._console_handler: logging.Handler | None = None
        self._verbosity = 1

    @property
    def verbosity(self) -> int:
        """Get current verbosity level."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: int) -> None:
        """Set verbosity level and update logger level.

        Parameters
        ----------
        level : int
            Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug).
        """
        self._verbosity = max(0, min(3, level))  # Clamp to 0-3

        # Map verbosity to logging level
        level_map = {
            0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG,
        }
        self.logger.setLevel(level_map[self._verbosity])

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)


# Global logger instance
_cli_logger: CLILogger | None = None


def setup_logging(
    log_file: str | Path | None = None,
    console: bool = True,
    verbosity: int = 1,
    structured: bool = False,
    rotation_enabled: bool = False,
    max_bytes: int = 10485760,  # 10 MB
    backup_count: int = 5,
    use_colors: bool = True,
) -> CLILogger:
    """Set up dual logging system.

    Configures both file and console logging handlers based on
    the provided configuration options.

    Parameters
    ----------
    log_file : str or Path, optional
        Path to log file. If None, file logging is disabled.
    console : bool
        Whether to enable console logging.
    verbosity : int
        Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug).
    structured : bool
        Whether to use structured JSON format for file logging.
    rotation_enabled : bool
        Whether to enable log rotation.
    max_bytes : int
        Maximum bytes per log file before rotation (default: 10 MB).
    backup_count : int
        Number of backup files to keep (default: 5).
    use_colors : bool
        Whether to use ANSI colors in console output.

    Returns
    -------
    CLILogger
        Configured logger instance.

    Examples
    --------
    >>> logger = setup_logging(
    ...     log_file="logs/workflow.log",
    ...     console=True,
    ...     verbosity=2,
    ...     structured=False,
    ...     rotation_enabled=True,
    ... )
    >>> logger.info("Starting workflow")
    """
    global _cli_logger  # noqa: PLW0603

    logger = CLILogger()
    logger.verbosity = verbosity

    # Clear existing handlers
    logger.logger.handlers.clear()

    # Set up file handler if log_file provided
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if rotation_enabled:
            file_handler: logging.Handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")

        # Set formatter based on structured option
        if structured:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        logger.logger.addHandler(file_handler)
        logger._file_handler = file_handler

    # Set up console handler if enabled
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(ColoredConsoleFormatter(use_colors=use_colors))

        # Console level based on verbosity
        level_map = {
            0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG,
        }
        console_handler.setLevel(level_map[verbosity])
        logger.logger.addHandler(console_handler)
        logger._console_handler = console_handler

    # Prevent propagation to root logger
    logger.logger.propagate = False

    _cli_logger = logger
    return logger


def get_logger() -> CLILogger:
    """Get the global CLI logger instance.

    If logging has not been set up, initializes with default settings.

    Returns
    -------
    CLILogger
        The global CLI logger instance.

    Examples
    --------
    >>> logger = get_logger()
    >>> logger.info("Processing file")
    """
    global _cli_logger  # noqa: PLW0603
    if _cli_logger is None:
        _cli_logger = setup_logging()
    return _cli_logger


def setup_logging_from_config(logging_config: dict[str, Any]) -> CLILogger:
    """Set up logging from a configuration dictionary.

    Parameters
    ----------
    logging_config : dict
        Configuration dictionary with keys:
        - log_file: str - Path to log file
        - console: bool - Enable console logging
        - structured.enabled: bool - Enable JSON format
        - rotation.enabled: bool - Enable log rotation
        - rotation.max_bytes: int - Max bytes per file
        - rotation.backup_count: int - Number of backups

    Returns
    -------
    CLILogger
        Configured logger instance.

    Examples
    --------
    >>> config = {
    ...     "log_file": "workflow.log",
    ...     "console": True,
    ...     "structured": {"enabled": False},
    ...     "rotation": {"enabled": True, "max_bytes": 10485760, "backup_count": 5},
    ... }
    >>> logger = setup_logging_from_config(config)
    """
    structured_config = logging_config.get("structured", {})
    rotation_config = logging_config.get("rotation", {})

    return setup_logging(
        log_file=logging_config.get("log_file"),
        console=logging_config.get("console", True),
        verbosity=logging_config.get("verbosity", 1),
        structured=structured_config.get("enabled", False),
        rotation_enabled=rotation_config.get("enabled", False),
        max_bytes=rotation_config.get("max_bytes", 10485760),
        backup_count=rotation_config.get("backup_count", 5),
    )

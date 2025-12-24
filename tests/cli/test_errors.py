"""Tests for NLSQ CLI error handling and logging infrastructure.

This module tests:
- CLI-specific exception classes (CLIError, ConfigError, DataLoadError, ModelError, FitError)
- Dual logging system (file + console)
- Structured JSON logging format
- Log rotation
- Verbosity levels

Test Categories
---------------
1. Exception class tests (ConfigError, DataLoadError, ModelError, FitError)
2. Exception serialization and message formatting
3. Dual logging initialization (file + console)
4. Structured JSON logging format
5. Log rotation functionality
6. Verbosity level handling
"""

import json
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def temp_log_dir():
    """Context manager for temp directory with proper logger cleanup.

    On Windows, file handles must be closed before the temp directory
    can be deleted. This context manager ensures all logging handlers
    are properly closed before cleanup.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            yield Path(tmpdir)
        finally:
            # Close all handlers on the nlsq.cli logger to release file handles
            # This is critical on Windows where open file handles prevent deletion
            nlsq_logger = logging.getLogger("nlsq.cli")
            handlers = nlsq_logger.handlers[:]
            for handler in handlers:
                handler.close()
                nlsq_logger.removeHandler(handler)


from nlsq.cli.errors import (
    CLIError,
    CLILogger,
    ColoredConsoleFormatter,
    ConfigError,
    DataLoadError,
    FitError,
    JsonFormatter,
    ModelError,
    get_logger,
    setup_logging,
    setup_logging_from_config,
)

# =============================================================================
# Test 1: ConfigError for Invalid YAML Configuration Scenarios
# =============================================================================


class TestConfigError:
    """Tests for ConfigError exception class."""

    def test_config_error_basic_message(self):
        """Test ConfigError with basic message."""
        error = ConfigError("Invalid YAML syntax")
        assert "Invalid YAML syntax" in str(error)
        assert error.message == "Invalid YAML syntax"
        assert isinstance(error, CLIError)

    def test_config_error_with_config_file(self):
        """Test ConfigError with config file context."""
        error = ConfigError(
            "Missing required key 'data.input_file'",
            config_file="workflow.yaml",
        )
        assert "workflow.yaml" in error.context["config_file"]
        assert "Missing required key" in error.message

    def test_config_error_with_key_and_suggestion(self):
        """Test ConfigError with key and actionable suggestion."""
        error = ConfigError(
            "Invalid value for 'fitting.method'",
            config_file="config.yaml",
            key="fitting.method",
            suggestion="Valid methods are: 'trf', 'lm', 'dogbox'",
        )
        assert error.context["key"] == "fitting.method"
        assert "Valid methods" in error.suggestion
        # Check formatted message includes suggestion
        assert "Suggestion:" in str(error)

    def test_config_error_to_dict(self):
        """Test ConfigError serialization to dictionary."""
        error = ConfigError(
            "Missing key",
            config_file="test.yaml",
            key="model.name",
            suggestion="Add model.name to config",
        )
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ConfigError"
        assert error_dict["message"] == "Missing key"
        assert error_dict["context"]["config_file"] == "test.yaml"
        assert error_dict["suggestion"] == "Add model.name to config"

    def test_config_error_can_be_raised_and_caught(self):
        """Test ConfigError can be raised and caught as CLIError."""
        with pytest.raises(CLIError) as exc_info:
            raise ConfigError("Test error", config_file="test.yaml")
        assert isinstance(exc_info.value, ConfigError)


# =============================================================================
# Test 2: DataLoadError for Missing/Malformed Data Files
# =============================================================================


class TestDataLoadError:
    """Tests for DataLoadError exception class."""

    def test_data_load_error_file_not_found(self):
        """Test DataLoadError for missing file."""
        error = DataLoadError(
            "Data file not found",
            file_path="/path/to/missing.csv",
            suggestion="Check that the file path is correct",
        )
        assert "Data file not found" in str(error)
        assert error.context["file_path"] == "/path/to/missing.csv"

    def test_data_load_error_column_not_found(self):
        """Test DataLoadError for missing column with available columns."""
        error = DataLoadError(
            "Column 'time' not found in data.csv",
            file_path="data/experiment.csv",
            file_format="csv",
            context={"available_columns": ["x", "y", "sigma"]},
            suggestion="Use one of the available columns: x, y, sigma",
        )
        assert error.context["file_format"] == "csv"
        assert error.context["available_columns"] == ["x", "y", "sigma"]
        assert "available columns" in error.suggestion

    def test_data_load_error_malformed_data(self):
        """Test DataLoadError for malformed data with NaN/Inf values."""
        error = DataLoadError(
            "Data contains 5 NaN values",
            file_path="data/experiment.txt",
            file_format="ascii",
            context={"nan_count": 5, "inf_count": 0},
            suggestion="Set validation.require_finite: false or clean the data",
        )
        assert error.context["nan_count"] == 5
        assert "require_finite" in error.suggestion

    def test_data_load_error_inherits_from_cli_error(self):
        """Test DataLoadError inheritance."""
        error = DataLoadError("Test")
        assert isinstance(error, CLIError)
        assert isinstance(error, Exception)


# =============================================================================
# Test 3: ModelError for Unresolvable Model Functions
# =============================================================================


class TestModelError:
    """Tests for ModelError exception class."""

    def test_model_error_builtin_not_found(self):
        """Test ModelError for unrecognized builtin model."""
        error = ModelError(
            "Model 'exponential_growth' not found in builtin models",
            model_name="exponential_growth",
            model_type="builtin",
            context={"available_models": ["linear", "exponential_decay", "gaussian"]},
            suggestion="Did you mean 'exponential_decay'?",
        )
        assert error.context["model_name"] == "exponential_growth"
        assert error.context["model_type"] == "builtin"
        assert "exponential_decay" in error.suggestion

    def test_model_error_custom_file_not_found(self):
        """Test ModelError for missing custom model file."""
        error = ModelError(
            "Custom model file not found",
            model_name="my_model",
            model_type="custom",
            context={"file_path": "/path/to/model.py"},
            suggestion="Check that the model file exists and is readable",
        )
        assert error.context["model_type"] == "custom"
        assert "file exists" in error.suggestion

    def test_model_error_invalid_signature(self):
        """Test ModelError for invalid model function signature."""
        error = ModelError(
            "Model function must have signature f(x, *params)",
            model_name="bad_model",
            context={"actual_signature": "f(a, b)"},
        )
        assert "signature" in error.message
        assert error.context["actual_signature"] == "f(a, b)"


# =============================================================================
# Test 4: FitError for curve_fit Failures
# =============================================================================


class TestFitError:
    """Tests for FitError exception class."""

    def test_fit_error_convergence_failure(self):
        """Test FitError for convergence failure."""
        error = FitError(
            "Curve fitting failed to converge",
            model_name="gaussian",
            context={"iterations": 1000, "final_cost": 1e10},
            suggestion="Try different initial parameters or relax tolerances",
        )
        assert "failed to converge" in error.message
        assert error.context["model_name"] == "gaussian"
        assert error.context["iterations"] == 1000

    def test_fit_error_covariance_estimation(self):
        """Test FitError for covariance estimation failure."""
        error = FitError(
            "Could not estimate covariance matrix (singular Hessian)",
            context={"condition_number": float("inf")},
            suggestion="Check for parameter redundancy or add bounds",
        )
        assert "covariance" in error.message
        assert "singular Hessian" in str(error)

    def test_fit_error_nan_result(self):
        """Test FitError for NaN results."""
        error = FitError(
            "Fit produced NaN values in parameters",
            model_name="exponential_decay",
            context={"nan_indices": [0, 2]},
            suggestion="Check data for outliers or use bounds to constrain parameters",
        )
        assert error.context["nan_indices"] == [0, 2]


# =============================================================================
# Test 5: Dual Logging Initialization (File + Console)
# =============================================================================


class TestDualLogging:
    """Tests for dual logging system initialization."""

    def test_setup_logging_console_only(self):
        """Test setting up console-only logging."""
        logger = setup_logging(
            log_file=None,
            console=True,
            verbosity=2,
        )
        assert isinstance(logger, CLILogger)
        assert logger.verbosity == 2
        assert logger._console_handler is not None
        assert logger._file_handler is None

    def test_setup_logging_file_only(self):
        """Test setting up file-only logging."""
        with temp_log_dir() as tmpdir:
            log_file = tmpdir / "test.log"
            logger = setup_logging(
                log_file=log_file,
                console=False,
                verbosity=1,
            )
            assert logger._file_handler is not None
            assert log_file.exists() or log_file.parent.exists()

    def test_setup_logging_dual(self):
        """Test setting up both file and console logging."""
        with temp_log_dir() as tmpdir:
            log_file = tmpdir / "dual.log"
            logger = setup_logging(
                log_file=log_file,
                console=True,
                verbosity=2,
            )
            assert logger._file_handler is not None
            assert logger._console_handler is not None

    def test_logging_messages_written_to_file(self):
        """Test that log messages are written to file."""
        with temp_log_dir() as tmpdir:
            log_file = tmpdir / "messages.log"
            logger = setup_logging(
                log_file=log_file,
                console=False,
                verbosity=3,  # Debug level
            )

            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")

            # Force flush
            for handler in logger.logger.handlers:
                handler.flush()

            content = log_file.read_text()
            assert "Test info message" in content
            assert "Test warning message" in content
            assert "Test error message" in content

    def test_get_logger_returns_singleton(self):
        """Test that get_logger returns consistent instance after setup."""
        with temp_log_dir() as tmpdir:
            log_file = tmpdir / "singleton.log"
            logger1 = setup_logging(log_file=log_file, console=True)
            logger2 = get_logger()
            # After setup, get_logger should return the configured logger
            assert logger2 is logger1


# =============================================================================
# Test 6: Structured JSON Logging Format
# =============================================================================


class TestStructuredLogging:
    """Tests for structured JSON logging format."""

    def test_json_formatter_basic(self):
        """Test JsonFormatter produces valid JSON."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="nlsq.cli",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_structured_logging_file_output(self):
        """Test structured JSON logging to file."""
        with temp_log_dir() as tmpdir:
            log_file = tmpdir / "structured.log"
            logger = setup_logging(
                log_file=log_file,
                console=False,
                verbosity=2,
                structured=True,
            )

            logger.info("Structured test message")

            # Force flush
            for handler in logger.logger.handlers:
                handler.flush()

            content = log_file.read_text().strip()
            data = json.loads(content)
            assert data["level"] == "INFO"
            assert data["message"] == "Structured test message"

    def test_setup_logging_from_config(self):
        """Test setup_logging_from_config with structured logging."""
        with temp_log_dir() as tmpdir:
            log_file = tmpdir / "config.log"
            config = {
                "log_file": str(log_file),
                "console": False,
                "verbosity": 2,
                "structured": {"enabled": True},
                "rotation": {"enabled": False},
            }
            logger = setup_logging_from_config(config)
            logger.info("Config-based log")

            for handler in logger.logger.handlers:
                handler.flush()

            content = log_file.read_text().strip()
            data = json.loads(content)
            assert data["message"] == "Config-based log"


# =============================================================================
# Additional Tests: Log Rotation and Verbosity
# =============================================================================


class TestLogRotation:
    """Tests for log rotation functionality."""

    def test_rotation_enabled(self):
        """Test log rotation is configured when enabled."""
        with temp_log_dir() as tmpdir:
            log_file = tmpdir / "rotating.log"
            logger = setup_logging(
                log_file=log_file,
                console=False,
                rotation_enabled=True,
                max_bytes=1024,
                backup_count=3,
            )
            # Verify handler type
            from logging.handlers import RotatingFileHandler

            assert isinstance(logger._file_handler, RotatingFileHandler)


class TestVerbosityLevels:
    """Tests for verbosity level handling."""

    def test_verbosity_level_0_silent(self):
        """Test verbosity 0 sets ERROR level."""
        logger = setup_logging(console=True, verbosity=0)
        assert logger.verbosity == 0
        assert logger.logger.level == logging.ERROR

    def test_verbosity_level_1_progress(self):
        """Test verbosity 1 sets WARNING level."""
        logger = setup_logging(console=True, verbosity=1)
        assert logger.verbosity == 1
        assert logger.logger.level == logging.WARNING

    def test_verbosity_level_2_detailed(self):
        """Test verbosity 2 sets INFO level."""
        logger = setup_logging(console=True, verbosity=2)
        assert logger.verbosity == 2
        assert logger.logger.level == logging.INFO

    def test_verbosity_level_3_debug(self):
        """Test verbosity 3 sets DEBUG level."""
        logger = setup_logging(console=True, verbosity=3)
        assert logger.verbosity == 3
        assert logger.logger.level == logging.DEBUG

    def test_verbosity_clamped_to_valid_range(self):
        """Test verbosity is clamped to 0-3 range."""
        logger = CLILogger()
        logger.verbosity = 10
        assert logger.verbosity == 3
        logger.verbosity = -5
        assert logger.verbosity == 0


class TestColoredConsoleFormatter:
    """Tests for ColoredConsoleFormatter."""

    def test_colored_formatter_creates_formatted_output(self):
        """Test ColoredConsoleFormatter produces formatted output."""
        formatter = ColoredConsoleFormatter(use_colors=False)
        record = logging.LogRecord(
            name="nlsq.cli",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "WARNING" in output
        assert "Warning message" in output

"""NLSQ CLI module for running curve fitting workflows from YAML configuration files.

This module provides command-line interface commands for NLSQ:
- `nlsq fit workflow.yaml` - Execute single curve fit from YAML configuration
- `nlsq batch w1.yaml w2.yaml ...` - Execute parallel batch fitting
- `nlsq info` - Display system and environment information

Example Usage
-------------
From command line:
    $ nlsq fit workflow.yaml
    $ nlsq batch configs/*.yaml
    $ nlsq info

From Python:
    >>> from nlsq.cli.errors import ConfigError, DataLoadError, FitError
    >>> from nlsq.cli.errors import setup_logging, get_logger
    >>> from nlsq.cli.data_loaders import DataLoader
    >>> from nlsq.cli.model_registry import ModelRegistry
    >>> from nlsq.cli.result_exporter import ResultExporter
    >>> from nlsq.cli.workflow_runner import WorkflowRunner
    >>> from nlsq.cli.visualization import FitVisualizer
    >>> from nlsq.cli.main import main
"""

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import (
    CLIError,
    ConfigError,
    DataLoadError,
    FitError,
    ModelError,
    get_logger,
    setup_logging,
)
from nlsq.cli.main import main
from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.result_exporter import ResultExporter
from nlsq.cli.visualization import FitVisualizer
from nlsq.cli.workflow_runner import WorkflowRunner

__all__ = [
    "CLIError",
    "ConfigError",
    "DataLoadError",
    "DataLoader",
    "FitError",
    "FitVisualizer",
    "ModelError",
    "ModelRegistry",
    "ResultExporter",
    "WorkflowRunner",
    "get_logger",
    "main",
    "setup_logging",
]

"""Diagnostic Plugin System for domain-specific extensions.

This module provides an extensible plugin architecture for domain-specific
diagnostic checks. Users can create custom plugins that integrate with
the standard health report.

The plugin system consists of:
- DiagnosticPlugin: Protocol defining the plugin interface
- PluginRegistry: Thread-safe global registry for plugins
- run_plugins(): Function to execute all registered plugins

Example
-------
>>> from nlsq.diagnostics import (
...     DiagnosticPlugin,
...     PluginRegistry,
...     PluginResult,
...     ModelHealthIssue,
...     IssueCategory,
...     IssueSeverity,
... )
>>> import numpy as np
>>>
>>> class MyDomainPlugin:
...     @property
...     def name(self) -> str:
...         return "my-domain"
...
...     def analyze(
...         self,
...         jacobian: np.ndarray,
...         parameters: np.ndarray,
...         residuals: np.ndarray,
...         **context
...     ) -> PluginResult:
...         issues = []
...         # Domain-specific analysis...
...         return PluginResult(
...             plugin_name=self.name,
...             data={"custom_metric": 1.0},
...             issues=issues,
...         )
>>>
>>> # Register the plugin
>>> PluginRegistry.register(MyDomainPlugin())
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from nlsq.diagnostics.types import PluginResult


@runtime_checkable
class DiagnosticPlugin(Protocol):
    """Protocol for diagnostic plugins.

    Users implement this protocol to create custom diagnostics that
    integrate with NLSQ's health report system.

    Attributes
    ----------
    name : str
        Unique plugin name. Should be a short, descriptive identifier.
        Convention: lowercase with hyphens (e.g., "optical-scattering").

    Protocol
    --------
    analyze(jacobian, parameters, residuals, **context)
        Run the plugin's analysis and return results.

    Example
    -------
    >>> class OpticalScatteringPlugin:
    ...     @property
    ...     def name(self) -> str:
    ...         return "optical-scattering"
    ...
    ...     def analyze(
    ...         self,
    ...         jacobian: np.ndarray,
    ...         parameters: np.ndarray,
    ...         residuals: np.ndarray,
    ...         **context
    ...     ) -> PluginResult:
    ...         issues = []
    ...         # Domain-specific analysis...
    ...         if any(parameters < 0):
    ...             issues.append(ModelHealthIssue(...))
    ...         return PluginResult(
    ...             plugin_name=self.name,
    ...             data={"my_metric": value},
    ...             issues=issues,
    ...         )
    ...
    >>> # Register the plugin
    >>> PluginRegistry.register(OpticalScatteringPlugin())
    """

    @property
    def name(self) -> str:
        """Unique plugin name.

        Should be a short, descriptive identifier.
        Convention: lowercase with hyphens (e.g., "optical-scattering").

        Returns
        -------
        str
            The plugin's unique name.
        """
        ...

    def analyze(
        self,
        jacobian: np.ndarray,
        parameters: np.ndarray,
        residuals: np.ndarray,
        **context: Any,
    ) -> PluginResult:
        """Run plugin analysis.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix at solution (n_residuals x n_params).
        parameters : np.ndarray
            Fitted parameters.
        residuals : np.ndarray
            Residuals at solution.
        **context : Any
            Additional context passed from curve_fit:
            - xdata: Independent variable data
            - ydata: Dependent variable data
            - bounds: Parameter bounds
            - model: Model function
            - config: DiagnosticsConfig

        Returns
        -------
        PluginResult
            Analysis results with any detected issues.
        """
        ...


DiagnosticPlugin.__module__ = "nlsq.diagnostics"
DiagnosticPlugin.analyze.__module__ = "nlsq.diagnostics"


class PluginRegistry:
    """Global registry for diagnostic plugins.

    Thread-safe singleton for managing plugin registration.
    Plugins are stored by name and can be registered, unregistered,
    retrieved, or listed.

    This class uses class-level storage and methods, so it acts as
    a singleton without explicit instantiation.

    Example
    -------
    >>> from nlsq.diagnostics import PluginRegistry
    >>> # Register a plugin
    >>> PluginRegistry.register(my_plugin)
    >>> # Get a plugin by name
    >>> plugin = PluginRegistry.get("my-plugin")
    >>> # List all plugins
    >>> all_plugins = PluginRegistry.all()
    >>> # Clear all plugins (for testing)
    >>> PluginRegistry.clear()
    """

    _plugins: ClassVar[dict[str, DiagnosticPlugin]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(cls, plugin: DiagnosticPlugin) -> None:
        """Register a diagnostic plugin.

        Parameters
        ----------
        plugin : DiagnosticPlugin
            Plugin instance to register.

        Raises
        ------
        ValueError
            If a plugin with the same name is already registered.

        Example
        -------
        >>> PluginRegistry.register(MyPlugin())
        """
        with cls._lock:
            name = plugin.name
            if name in cls._plugins:
                raise ValueError(
                    f"Plugin '{name}' is already registered. "
                    "Use unregister() first to replace it."
                )
            cls._plugins[name] = plugin

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a plugin by name.

        Parameters
        ----------
        name : str
            Name of the plugin to unregister.

        Returns
        -------
        bool
            True if the plugin was found and removed, False otherwise.

        Example
        -------
        >>> PluginRegistry.unregister("my-plugin")
        True
        """
        with cls._lock:
            if name in cls._plugins:
                del cls._plugins[name]
                return True
            return False

    @classmethod
    def get(cls, name: str) -> DiagnosticPlugin | None:
        """Get a plugin by name.

        Parameters
        ----------
        name : str
            Name of the plugin to retrieve.

        Returns
        -------
        DiagnosticPlugin | None
            Plugin instance if found, None otherwise.

        Example
        -------
        >>> plugin = PluginRegistry.get("my-plugin")
        >>> if plugin is not None:
        ...     result = plugin.analyze(...)
        """
        with cls._lock:
            return cls._plugins.get(name)

    @classmethod
    def all(cls) -> list[DiagnosticPlugin]:
        """Get all registered plugins.

        Returns
        -------
        list[DiagnosticPlugin]
            List of all registered plugin instances.
            This is a copy; modifying it does not affect the registry.

        Example
        -------
        >>> for plugin in PluginRegistry.all():
        ...     print(plugin.name)
        """
        with cls._lock:
            return list(cls._plugins.values())

    @classmethod
    def clear(cls) -> None:
        """Unregister all plugins.

        Primarily for testing purposes.

        Example
        -------
        >>> PluginRegistry.clear()
        >>> assert len(PluginRegistry.all()) == 0
        """
        with cls._lock:
            cls._plugins.clear()


def run_plugins(
    jacobian: np.ndarray,
    parameters: np.ndarray,
    residuals: np.ndarray,
    **context: Any,
) -> dict[str, PluginResult]:
    """Execute all registered plugins with exception isolation.

    This function runs each registered plugin's analyze() method and
    collects the results. If a plugin raises an exception, it is caught,
    logged via warnings, and the plugin's result is marked as unavailable.
    Other plugins continue executing normally (FR-014).

    Parameters
    ----------
    jacobian : np.ndarray
        Jacobian matrix at solution (n_residuals x n_params).
    parameters : np.ndarray
        Fitted parameters.
    residuals : np.ndarray
        Residuals at solution.
    **context : Any
        Additional context for plugins:
        - xdata: Independent variable data
        - ydata: Dependent variable data
        - bounds: Parameter bounds
        - model: Model function
        - config: DiagnosticsConfig

    Returns
    -------
    dict[str, PluginResult]
        Mapping of plugin name to result.
        If a plugin fails, its result has available=False with error_message.

    Example
    -------
    >>> from nlsq.diagnostics import run_plugins
    >>> results = run_plugins(jacobian, params, residuals, xdata=x, ydata=y)
    >>> for name, result in results.items():
    ...     if result.available:
    ...         print(f"{name}: {len(result.issues)} issues")
    ...     else:
    ...         print(f"{name}: FAILED - {result.error_message}")
    """
    plugins = PluginRegistry.all()
    results: dict[str, PluginResult] = {}

    for plugin in plugins:
        plugin_name = plugin.name
        start_time = time.perf_counter()

        try:
            result = plugin.analyze(
                jacobian=jacobian,
                parameters=parameters,
                residuals=residuals,
                **context,
            )

            # Handle None return (treat as empty result)
            if result is None:
                result = PluginResult(
                    plugin_name=plugin_name,
                    available=True,
                    data={},
                    issues=[],
                )

            # Handle invalid return type (not PluginResult)
            # Use duck typing to handle module identity issues with pytest-xdist
            if not (
                hasattr(result, "plugin_name")
                and hasattr(result, "data")
                and hasattr(result, "issues")
            ):
                result = PluginResult(
                    plugin_name=plugin_name,
                    available=True,
                    data={},
                    issues=[],
                )

            # Ensure computation time is recorded
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result = PluginResult(
                plugin_name=result.plugin_name,
                available=result.available,
                error_message=result.error_message,
                data=result.data,
                issues=result.issues,
                computation_time_ms=elapsed_ms,
            )

            results[plugin_name] = result

        except Exception as e:
            # Exception isolation per FR-014
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Emit warning about plugin failure
            warnings.warn(
                f"Diagnostic plugin '{plugin_name}' failed: {e}",
                UserWarning,
                stacklevel=2,
            )

            # Create unavailable result with error message
            results[plugin_name] = PluginResult(
                plugin_name=plugin_name,
                available=False,
                error_message=str(e),
                data={},
                issues=[],
                computation_time_ms=elapsed_ms,
            )

    return results

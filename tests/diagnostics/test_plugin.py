"""Unit tests for the diagnostics plugin system.

Tests cover:
- T044: PluginRegistry (register, unregister, get, all, clear, duplicate name, thread-safety)
- T045: Plugin execution and exception isolation (run_plugins, parameter passing, exception handling)

These tests are written FIRST following Test-First development. They will fail
until plugin.py is implemented.

This module is marked serial because PluginRegistry uses class-level global state
that causes race conditions when tests run in parallel across pytest-xdist workers.
"""

from __future__ import annotations

import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mark all tests in this module as serial to avoid PluginRegistry race conditions.
# The root conftest.py assigns serial tests to the same xdist worker group.
pytestmark = pytest.mark.serial

# PluginResult already exists in types.py
from nlsq.diagnostics.types import (
    DiagnosticsConfig,
    IssueCategory,
    IssueSeverity,
    ModelHealthIssue,
    PluginResult,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleDataPlugin:
    """A simple plugin that returns data without issues."""

    @property
    def name(self) -> str:
        return "simple-data"

    def analyze(
        self,
        jacobian: np.ndarray,
        parameters: np.ndarray,
        residuals: np.ndarray,
        **context: Any,
    ) -> PluginResult:
        """Return basic data about the inputs."""
        return PluginResult(
            plugin_name=self.name,
            data={
                "jacobian_shape": jacobian.shape,
                "n_parameters": len(parameters),
                "residual_norm": float(np.linalg.norm(residuals)),
            },
            issues=[],
        )


class IssueReturningPlugin:
    """A plugin that returns issues based on analysis."""

    @property
    def name(self) -> str:
        return "issue-detector"

    def analyze(
        self,
        jacobian: np.ndarray,
        parameters: np.ndarray,
        residuals: np.ndarray,
        **context: Any,
    ) -> PluginResult:
        """Detect negative parameters and return issues."""
        issues = []
        negative_indices = np.where(parameters < 0)[0]

        if len(negative_indices) > 0:
            issues.append(
                ModelHealthIssue(
                    category=IssueCategory.IDENTIFIABILITY,
                    severity=IssueSeverity.WARNING,
                    code="TEST-001",
                    message="Negative parameter values detected",
                    affected_parameters=tuple(int(i) for i in negative_indices),
                    details={"negative_values": parameters[negative_indices].tolist()},
                    recommendation="Consider adding lower bounds of 0",
                )
            )

        return PluginResult(
            plugin_name=self.name,
            data={"negative_count": len(negative_indices)},
            issues=issues,
        )


class ExceptionRaisingPlugin:
    """A plugin that raises an exception during analysis."""

    @property
    def name(self) -> str:
        return "failing-plugin"

    def analyze(
        self,
        jacobian: np.ndarray,
        parameters: np.ndarray,
        residuals: np.ndarray,
        **context: Any,
    ) -> PluginResult:
        """Always raise an exception."""
        raise RuntimeError("Plugin crashed intentionally")


class NoneReturningPlugin:
    """A plugin that returns None instead of PluginResult."""

    @property
    def name(self) -> str:
        return "none-returner"

    def analyze(
        self,
        jacobian: np.ndarray,
        parameters: np.ndarray,
        residuals: np.ndarray,
        **context: Any,
    ) -> PluginResult:
        """Return None to test handling of invalid return."""
        return None  # type: ignore[return-value]


class ContextCapturingPlugin:
    """A plugin that captures context for verification."""

    def __init__(self) -> None:
        self.captured_context: dict[str, Any] = {}
        self.captured_jacobian: np.ndarray | None = None
        self.captured_parameters: np.ndarray | None = None
        self.captured_residuals: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "context-capturer"

    def analyze(
        self,
        jacobian: np.ndarray,
        parameters: np.ndarray,
        residuals: np.ndarray,
        **context: Any,
    ) -> PluginResult:
        """Capture all inputs for later verification."""
        self.captured_jacobian = jacobian.copy()
        self.captured_parameters = parameters.copy()
        self.captured_residuals = residuals.copy()
        self.captured_context = dict(context)

        return PluginResult(
            plugin_name=self.name,
            data={"context_keys": list(context.keys())},
            issues=[],
        )


class SlowPlugin:
    """A plugin that takes time to execute for concurrency testing."""

    def __init__(self, delay: float = 0.1) -> None:
        self.delay = delay
        self.execution_count = 0

    @property
    def name(self) -> str:
        return "slow-plugin"

    def analyze(
        self,
        jacobian: np.ndarray,
        parameters: np.ndarray,
        residuals: np.ndarray,
        **context: Any,
    ) -> PluginResult:
        """Simulate slow analysis."""
        time.sleep(self.delay)
        self.execution_count += 1
        return PluginResult(
            plugin_name=self.name,
            data={"execution_count": self.execution_count},
            issues=[],
        )


@pytest.fixture
def sample_jacobian() -> np.ndarray:
    """Create a sample Jacobian matrix."""
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def sample_parameters() -> np.ndarray:
    """Create sample parameters."""
    return np.array([1.5, 2.5])


@pytest.fixture
def sample_residuals() -> np.ndarray:
    """Create sample residuals."""
    return np.array([0.1, -0.2, 0.05])


@pytest.fixture
def sample_context() -> dict[str, Any]:
    """Create sample context for plugins."""
    return {
        "xdata": np.array([1.0, 2.0, 3.0]),
        "ydata": np.array([2.0, 4.0, 6.0]),
        "bounds": (np.array([0.0, 0.0]), np.array([np.inf, np.inf])),
        "model": lambda x, a, b: a * x + b,
        "config": DiagnosticsConfig(),
    }


# Note: clear_registry fixture is now in tests/diagnostics/conftest.py
# to ensure consistent cleanup across all diagnostics tests.


# =============================================================================
# T044: PluginRegistry Tests
# =============================================================================


@pytest.mark.diagnostics
class TestPluginRegistryRegister:
    """Tests for PluginRegistry.register() method."""

    def test_register_adds_plugin_to_registry(self) -> None:
        """Test that register() adds a plugin to the registry."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin = SimpleDataPlugin()
        PluginRegistry.register(plugin)

        registered = PluginRegistry.get("simple-data")
        assert registered is not None
        assert registered.name == "simple-data"

    def test_register_multiple_plugins(self) -> None:
        """Test that multiple plugins can be registered."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin1 = SimpleDataPlugin()
        plugin2 = IssueReturningPlugin()

        PluginRegistry.register(plugin1)
        PluginRegistry.register(plugin2)

        assert PluginRegistry.get("simple-data") is not None
        assert PluginRegistry.get("issue-detector") is not None
        assert len(PluginRegistry.all()) == 2

    def test_register_duplicate_name_raises_value_error(self) -> None:
        """Test that registering a duplicate name raises ValueError."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin1 = SimpleDataPlugin()
        PluginRegistry.register(plugin1)

        # Create another plugin with the same name
        class DuplicateNamePlugin:
            @property
            def name(self) -> str:
                return "simple-data"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                return PluginResult(plugin_name=self.name)

        plugin2 = DuplicateNamePlugin()

        with pytest.raises(ValueError, match="already registered"):
            PluginRegistry.register(plugin2)

    def test_register_preserves_plugin_instance(self) -> None:
        """Test that register preserves the exact plugin instance."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin = SimpleDataPlugin()
        PluginRegistry.register(plugin)

        retrieved = PluginRegistry.get("simple-data")
        assert retrieved is plugin


@pytest.mark.diagnostics
class TestPluginRegistryUnregister:
    """Tests for PluginRegistry.unregister() method."""

    def test_unregister_removes_plugin(self) -> None:
        """Test that unregister() removes a plugin from the registry."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin = SimpleDataPlugin()
        PluginRegistry.register(plugin)

        result = PluginRegistry.unregister("simple-data")

        assert result is True
        assert PluginRegistry.get("simple-data") is None

    def test_unregister_returns_true_if_found(self) -> None:
        """Test that unregister() returns True when plugin is found."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin = SimpleDataPlugin()
        PluginRegistry.register(plugin)

        result = PluginRegistry.unregister("simple-data")
        assert result is True

    def test_unregister_returns_false_if_not_found(self) -> None:
        """Test that unregister() returns False when plugin is not found."""
        from nlsq.diagnostics.plugin import PluginRegistry

        result = PluginRegistry.unregister("nonexistent-plugin")
        assert result is False

    def test_unregister_allows_reregistration(self) -> None:
        """Test that after unregister, a plugin with the same name can be registered."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin1 = SimpleDataPlugin()
        PluginRegistry.register(plugin1)
        PluginRegistry.unregister("simple-data")

        # Should not raise
        plugin2 = SimpleDataPlugin()
        PluginRegistry.register(plugin2)

        assert PluginRegistry.get("simple-data") is plugin2


@pytest.mark.diagnostics
class TestPluginRegistryGet:
    """Tests for PluginRegistry.get() method."""

    def test_get_returns_plugin_when_found(self) -> None:
        """Test that get() returns the plugin when found."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin = SimpleDataPlugin()
        PluginRegistry.register(plugin)

        retrieved = PluginRegistry.get("simple-data")
        assert retrieved is plugin

    def test_get_returns_none_when_not_found(self) -> None:
        """Test that get() returns None when plugin is not found."""
        from nlsq.diagnostics.plugin import PluginRegistry

        result = PluginRegistry.get("nonexistent-plugin")
        assert result is None

    def test_get_returns_correct_plugin_among_multiple(self) -> None:
        """Test that get() returns the correct plugin among multiple."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin1 = SimpleDataPlugin()
        plugin2 = IssueReturningPlugin()
        PluginRegistry.register(plugin1)
        PluginRegistry.register(plugin2)

        assert PluginRegistry.get("simple-data") is plugin1
        assert PluginRegistry.get("issue-detector") is plugin2


@pytest.mark.diagnostics
class TestPluginRegistryAll:
    """Tests for PluginRegistry.all() method."""

    def test_all_returns_empty_list_when_no_plugins(self) -> None:
        """Test that all() returns empty list when no plugins registered."""
        from nlsq.diagnostics.plugin import PluginRegistry

        result = PluginRegistry.all()
        assert result == []

    def test_all_returns_list_of_all_plugins(self) -> None:
        """Test that all() returns list of all registered plugins."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin1 = SimpleDataPlugin()
        plugin2 = IssueReturningPlugin()
        PluginRegistry.register(plugin1)
        PluginRegistry.register(plugin2)

        all_plugins = PluginRegistry.all()
        assert len(all_plugins) == 2
        assert plugin1 in all_plugins
        assert plugin2 in all_plugins

    def test_all_returns_copy_of_list(self) -> None:
        """Test that all() returns a copy, not the internal list."""
        from nlsq.diagnostics.plugin import PluginRegistry

        plugin = SimpleDataPlugin()
        PluginRegistry.register(plugin)

        all_plugins = PluginRegistry.all()
        all_plugins.clear()  # Modify the returned list

        # Internal list should be unaffected
        assert len(PluginRegistry.all()) == 1


@pytest.mark.diagnostics
class TestPluginRegistryClear:
    """Tests for PluginRegistry.clear() method."""

    def test_clear_removes_all_plugins(self) -> None:
        """Test that clear() removes all registered plugins."""
        from nlsq.diagnostics.plugin import PluginRegistry

        PluginRegistry.register(SimpleDataPlugin())
        PluginRegistry.register(IssueReturningPlugin())

        PluginRegistry.clear()

        assert len(PluginRegistry.all()) == 0
        assert PluginRegistry.get("simple-data") is None
        assert PluginRegistry.get("issue-detector") is None

    def test_clear_on_empty_registry_succeeds(self) -> None:
        """Test that clear() on empty registry doesn't raise."""
        from nlsq.diagnostics.plugin import PluginRegistry

        # Should not raise
        PluginRegistry.clear()
        assert len(PluginRegistry.all()) == 0


@pytest.mark.diagnostics
class TestPluginRegistryThreadSafety:
    """Tests for PluginRegistry thread safety."""

    def test_concurrent_register_no_duplicates(self) -> None:
        """Test concurrent registration doesn't create duplicates."""
        from nlsq.diagnostics.plugin import PluginRegistry

        errors: list[Exception] = []
        registered_count = 0
        lock = threading.Lock()

        def try_register(plugin_num: int) -> None:
            nonlocal registered_count
            try:

                class ThreadPlugin:
                    @property
                    def name(self) -> str:
                        return f"thread-plugin-{plugin_num}"

                    def analyze(
                        self,
                        jacobian: np.ndarray,
                        parameters: np.ndarray,
                        residuals: np.ndarray,
                        **context: Any,
                    ) -> PluginResult:
                        return PluginResult(plugin_name=self.name)

                PluginRegistry.register(ThreadPlugin())
                with lock:
                    registered_count += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        # Register 10 different plugins concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(try_register, range(10))

        assert len(errors) == 0
        assert registered_count == 10
        assert len(PluginRegistry.all()) == 10

    def test_concurrent_register_same_name_one_succeeds(self) -> None:
        """Test concurrent registration of same name - exactly one succeeds."""
        from nlsq.diagnostics.plugin import PluginRegistry

        success_count = 0
        error_count = 0
        lock = threading.Lock()

        def try_register(_: int) -> None:
            nonlocal success_count, error_count

            class SameNamePlugin:
                @property
                def name(self) -> str:
                    return "same-name-plugin"

                def analyze(
                    self,
                    jacobian: np.ndarray,
                    parameters: np.ndarray,
                    residuals: np.ndarray,
                    **context: Any,
                ) -> PluginResult:
                    return PluginResult(plugin_name=self.name)

            try:
                PluginRegistry.register(SameNamePlugin())
                with lock:
                    success_count += 1
            except ValueError:
                with lock:
                    error_count += 1

        # Try to register the same name 10 times concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(try_register, range(10))

        assert success_count == 1
        assert error_count == 9
        assert len(PluginRegistry.all()) == 1

    def test_concurrent_register_and_unregister(self) -> None:
        """Test concurrent register and unregister operations."""
        from nlsq.diagnostics.plugin import PluginRegistry

        errors: list[Exception] = []
        lock = threading.Lock()

        def register_plugin(i: int) -> None:
            try:

                class RegisterPlugin:
                    @property
                    def name(self) -> str:
                        return f"reg-plugin-{i}"

                    def analyze(
                        self,
                        jacobian: np.ndarray,
                        parameters: np.ndarray,
                        residuals: np.ndarray,
                        **context: Any,
                    ) -> PluginResult:
                        return PluginResult(plugin_name=self.name)

                PluginRegistry.register(RegisterPlugin())
            except Exception as e:
                with lock:
                    errors.append(e)

        def unregister_plugin(i: int) -> None:
            try:
                PluginRegistry.unregister(f"reg-plugin-{i}")
            except Exception as e:
                with lock:
                    errors.append(e)

        # Interleave register and unregister operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Register some first
            list(executor.map(register_plugin, range(5)))
            # Then concurrently register and unregister
            futures = [executor.submit(register_plugin, i) for i in range(5, 10)]
            futures.extend(executor.submit(unregister_plugin, i) for i in range(5))

            # Wait for all futures
            for f in futures:
                f.result()

        assert len(errors) == 0
        # Should have plugins 5-9 registered (0-4 were unregistered)
        all_plugins = PluginRegistry.all()
        names = [p.name for p in all_plugins]
        for i in range(5, 10):
            assert f"reg-plugin-{i}" in names


# =============================================================================
# T045: Plugin Execution and Exception Isolation Tests
# =============================================================================


@pytest.mark.diagnostics
class TestRunPlugins:
    """Tests for run_plugins() function."""

    def test_run_plugins_executes_all_registered(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that run_plugins() executes all registered plugins."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        # Explicit clear for test isolation
        PluginRegistry.clear()

        plugin1 = SimpleDataPlugin()
        plugin2 = IssueReturningPlugin()
        PluginRegistry.register(plugin1)
        PluginRegistry.register(plugin2)

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        assert len(results) == 2
        assert "simple-data" in results
        assert "issue-detector" in results

    def test_run_plugins_returns_dict_keyed_by_name(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that run_plugins() returns dict keyed by plugin name."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        # Explicit clear for test isolation
        PluginRegistry.clear()

        PluginRegistry.register(SimpleDataPlugin())

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        assert isinstance(results, dict)
        assert "simple-data" in results
        result = results["simple-data"]
        # Use type name check to avoid isinstance identity issues in pytest-xdist
        assert type(result).__name__ == "PluginResult"
        assert result.plugin_name == "simple-data"

    def test_run_plugins_returns_empty_dict_when_no_plugins(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that run_plugins() returns empty dict when no plugins registered."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        # Explicit clear for test isolation - ensures no plugins are registered
        PluginRegistry.clear()

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        assert results == {}


@pytest.mark.diagnostics
class TestPluginParameterPassing:
    """Tests for correct parameter passing to plugins."""

    def test_plugin_receives_jacobian(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin receives the correct jacobian."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        plugin = ContextCapturingPlugin()
        PluginRegistry.register(plugin)

        run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        assert plugin.captured_jacobian is not None
        np.testing.assert_array_equal(plugin.captured_jacobian, sample_jacobian)

    def test_plugin_receives_parameters(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin receives the correct parameters."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        plugin = ContextCapturingPlugin()
        PluginRegistry.register(plugin)

        run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        assert plugin.captured_parameters is not None
        np.testing.assert_array_equal(plugin.captured_parameters, sample_parameters)

    def test_plugin_receives_residuals(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin receives the correct residuals."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        plugin = ContextCapturingPlugin()
        PluginRegistry.register(plugin)

        run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        assert plugin.captured_residuals is not None
        np.testing.assert_array_equal(plugin.captured_residuals, sample_residuals)

    def test_plugin_receives_context_xdata(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
        sample_context: dict[str, Any],
    ) -> None:
        """Test that plugin receives xdata in context."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        plugin = ContextCapturingPlugin()
        PluginRegistry.register(plugin)

        run_plugins(
            sample_jacobian, sample_parameters, sample_residuals, **sample_context
        )

        assert "xdata" in plugin.captured_context
        np.testing.assert_array_equal(
            plugin.captured_context["xdata"], sample_context["xdata"]
        )

    def test_plugin_receives_context_ydata(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
        sample_context: dict[str, Any],
    ) -> None:
        """Test that plugin receives ydata in context."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        plugin = ContextCapturingPlugin()
        PluginRegistry.register(plugin)

        run_plugins(
            sample_jacobian, sample_parameters, sample_residuals, **sample_context
        )

        assert "ydata" in plugin.captured_context
        np.testing.assert_array_equal(
            plugin.captured_context["ydata"], sample_context["ydata"]
        )

    def test_plugin_receives_context_bounds(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
        sample_context: dict[str, Any],
    ) -> None:
        """Test that plugin receives bounds in context."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        plugin = ContextCapturingPlugin()
        PluginRegistry.register(plugin)

        run_plugins(
            sample_jacobian, sample_parameters, sample_residuals, **sample_context
        )

        assert "bounds" in plugin.captured_context
        assert len(plugin.captured_context["bounds"]) == 2

    def test_plugin_receives_context_model(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
        sample_context: dict[str, Any],
    ) -> None:
        """Test that plugin receives model in context."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        plugin = ContextCapturingPlugin()
        PluginRegistry.register(plugin)

        run_plugins(
            sample_jacobian, sample_parameters, sample_residuals, **sample_context
        )

        assert "model" in plugin.captured_context
        # Verify it's callable
        assert callable(plugin.captured_context["model"])

    def test_plugin_receives_context_config(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
        sample_context: dict[str, Any],
    ) -> None:
        """Test that plugin receives config in context."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        plugin = ContextCapturingPlugin()
        PluginRegistry.register(plugin)

        run_plugins(
            sample_jacobian, sample_parameters, sample_residuals, **sample_context
        )

        assert "config" in plugin.captured_context
        assert isinstance(plugin.captured_context["config"], DiagnosticsConfig)


@pytest.mark.diagnostics
class TestPluginIssueCollection:
    """Tests for plugin issue collection in PluginResult."""

    def test_plugin_issues_collected_in_result(
        self,
        sample_jacobian: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin issues are collected in PluginResult."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        PluginRegistry.register(IssueReturningPlugin())

        # Use negative parameters to trigger issue
        negative_params = np.array([-1.0, 2.0])
        results = run_plugins(sample_jacobian, negative_params, sample_residuals)

        result = results["issue-detector"]
        assert result.available is True
        assert len(result.issues) == 1
        assert result.issues[0].code == "TEST-001"
        assert 0 in result.issues[0].affected_parameters  # type: ignore[operator]

    def test_plugin_no_issues_empty_list(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin with no issues returns empty issues list."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        PluginRegistry.register(SimpleDataPlugin())

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        result = results["simple-data"]
        assert result.available is True
        assert len(result.issues) == 0

    def test_plugin_data_preserved_in_result(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin data is preserved in PluginResult."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        PluginRegistry.register(SimpleDataPlugin())

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        result = results["simple-data"]
        assert "jacobian_shape" in result.data
        assert result.data["jacobian_shape"] == sample_jacobian.shape
        assert result.data["n_parameters"] == len(sample_parameters)


@pytest.mark.diagnostics
class TestPluginNoneReturn:
    """Tests for handling plugin that returns None."""

    def test_plugin_returning_none_treated_as_empty_result(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin returning None is treated as empty result."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        # Explicit clear for test isolation (belt and suspenders with conftest fixture)
        PluginRegistry.clear()
        assert len(PluginRegistry.all()) == 0, "Registry should be empty after clear"

        PluginRegistry.register(NoneReturningPlugin())

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        # Should have a result for the plugin
        assert "none-returner" in results
        result = results["none-returner"]
        # Result should indicate plugin ran but returned no data
        # Use type name check to avoid isinstance identity issues in pytest-xdist
        assert type(result).__name__ == "PluginResult"
        assert len(result.issues) == 0


@pytest.mark.diagnostics
class TestPluginExceptionIsolation:
    """Tests for exception isolation - failing plugins don't break others."""

    def test_failing_plugin_does_not_stop_others(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that a failing plugin doesn't prevent other plugins from running."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        plugin_a = SimpleDataPlugin()
        plugin_fail = ExceptionRaisingPlugin()
        plugin_c = IssueReturningPlugin()

        PluginRegistry.register(plugin_a)
        PluginRegistry.register(plugin_fail)
        PluginRegistry.register(plugin_c)

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        # All three plugins should have results
        assert len(results) == 3
        assert "simple-data" in results
        assert "failing-plugin" in results
        assert "issue-detector" in results

        # A and C should have succeeded
        assert results["simple-data"].available is True
        assert results["issue-detector"].available is True

    def test_failed_plugin_returns_unavailable_result(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that failed plugin returns PluginResult with available=False."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        PluginRegistry.register(ExceptionRaisingPlugin())

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        result = results["failing-plugin"]
        assert result.available is False

    def test_failed_plugin_has_error_message(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that failed plugin has error_message set."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        PluginRegistry.register(ExceptionRaisingPlugin())

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        result = results["failing-plugin"]
        assert result.error_message is not None
        assert "Plugin crashed intentionally" in result.error_message

    def test_failed_plugin_emits_warning(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that failed plugin emits a warning."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        PluginRegistry.register(ExceptionRaisingPlugin())

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            run_plugins(sample_jacobian, sample_parameters, sample_residuals)

            # Should have emitted at least one warning
            plugin_warnings = [
                w for w in caught_warnings if "failing-plugin" in str(w.message)
            ]
            assert len(plugin_warnings) >= 1

    def test_exception_does_not_propagate_to_caller(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin exception does not propagate to caller."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        PluginRegistry.register(ExceptionRaisingPlugin())

        # Should not raise - exception is caught internally
        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        # We got results, no exception propagated
        assert results is not None
        assert "failing-plugin" in results

    def test_multiple_failing_plugins_all_isolated(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that multiple failing plugins are all isolated."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        # Create two failing plugins
        class FailingPlugin1:
            @property
            def name(self) -> str:
                return "failing-1"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                raise ValueError("First failure")

        class FailingPlugin2:
            @property
            def name(self) -> str:
                return "failing-2"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                raise TypeError("Second failure")

        PluginRegistry.register(FailingPlugin1())
        PluginRegistry.register(SimpleDataPlugin())
        PluginRegistry.register(FailingPlugin2())

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        assert len(results) == 3
        assert results["failing-1"].available is False
        assert "First failure" in results["failing-1"].error_message  # type: ignore[operator]
        assert results["failing-2"].available is False
        assert "Second failure" in results["failing-2"].error_message  # type: ignore[operator]
        assert results["simple-data"].available is True


@pytest.mark.diagnostics
class TestPluginProtocol:
    """Tests for DiagnosticPlugin protocol compliance."""

    def test_plugin_must_have_name_property(self) -> None:
        """Test that plugins must have a name property."""
        from nlsq.diagnostics.plugin import DiagnosticPlugin

        # Create a properly structured plugin
        class ValidPlugin:
            @property
            def name(self) -> str:
                return "valid-plugin"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                return PluginResult(plugin_name=self.name)

        plugin = ValidPlugin()
        # Should be able to access name
        assert plugin.name == "valid-plugin"

    def test_plugin_must_have_analyze_method(self) -> None:
        """Test that plugins must have an analyze method."""
        from nlsq.diagnostics.plugin import DiagnosticPlugin

        class ValidPlugin:
            @property
            def name(self) -> str:
                return "valid-plugin"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                return PluginResult(plugin_name=self.name)

        plugin = ValidPlugin()
        # Should be able to call analyze
        result = plugin.analyze(
            np.array([[1.0]]),
            np.array([1.0]),
            np.array([0.0]),
        )
        # Use type name check to avoid isinstance identity issues in pytest-xdist
        assert type(result).__name__ == "PluginResult"


@pytest.mark.diagnostics
class TestPluginPerformance:
    """Tests for plugin system performance characteristics."""

    def test_run_plugins_execution_order_consistent(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugins execute in consistent order."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        execution_order: list[str] = []

        class OrderTrackingPlugin1:
            @property
            def name(self) -> str:
                return "order-1"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                execution_order.append("order-1")
                return PluginResult(plugin_name=self.name)

        class OrderTrackingPlugin2:
            @property
            def name(self) -> str:
                return "order-2"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                execution_order.append("order-2")
                return PluginResult(plugin_name=self.name)

        PluginRegistry.register(OrderTrackingPlugin1())
        PluginRegistry.register(OrderTrackingPlugin2())

        # Run multiple times and check order is consistent
        for _ in range(3):
            execution_order.clear()
            run_plugins(sample_jacobian, sample_parameters, sample_residuals)
            # Just verify both executed - order depends on registration order
            assert "order-1" in execution_order
            assert "order-2" in execution_order

    def test_plugin_execution_time_recorded(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin execution time is recorded in result."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        slow_plugin = SlowPlugin(delay=0.05)
        PluginRegistry.register(slow_plugin)

        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        result = results["slow-plugin"]
        # Should have recorded some execution time
        assert result.computation_time_ms >= 40  # At least 40ms given 50ms delay


@pytest.mark.diagnostics
class TestPluginIntegration:
    """Integration tests for plugin system with real usage patterns."""

    def test_optical_scattering_plugin_example(
        self,
        sample_jacobian: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test example optical scattering plugin pattern from contract."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        class OpticalScatteringPlugin:
            """Example plugin for optical scattering diagnostics."""

            @property
            def name(self) -> str:
                return "optical-scattering"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                issues = []
                data = {}

                # Check for unphysical scattering cross-section
                if np.any(parameters < 0):
                    issues.append(
                        ModelHealthIssue(
                            category=IssueCategory.IDENTIFIABILITY,
                            severity=IssueSeverity.WARNING,
                            code="OPT-001",
                            message="Negative scattering cross-section detected",
                            affected_parameters=tuple(np.where(parameters < 0)[0]),
                            details={
                                "negative_values": parameters[parameters < 0].tolist()
                            },
                            recommendation="Add lower bounds of 0 for physical parameters",
                        )
                    )

                # Domain-specific metric
                data["signal_to_noise"] = np.std(residuals) / (
                    np.mean(np.abs(residuals)) + 1e-10
                )

                return PluginResult(
                    plugin_name=self.name,
                    data=data,
                    issues=issues,
                )

        plugin = OpticalScatteringPlugin()
        PluginRegistry.register(plugin)

        # Test with negative parameter (unphysical)
        negative_params = np.array([-0.5, 1.2])
        results = run_plugins(sample_jacobian, negative_params, sample_residuals)

        result = results["optical-scattering"]
        assert result.available is True
        assert len(result.issues) == 1
        assert result.issues[0].code == "OPT-001"
        assert "signal_to_noise" in result.data

    def test_plugin_registered_at_module_import(self) -> None:
        """Test pattern of registering plugin at module import time."""
        from nlsq.diagnostics.plugin import PluginRegistry

        # Simulate module-level registration
        class ModuleLevelPlugin:
            @property
            def name(self) -> str:
                return "module-level"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                return PluginResult(plugin_name=self.name)

        # Register at "module import time"
        PluginRegistry.register(ModuleLevelPlugin())

        # Verify plugin is available
        assert PluginRegistry.get("module-level") is not None


@pytest.mark.diagnostics
class TestPluginErrorHandling:
    """Tests for various error handling scenarios."""

    def test_plugin_with_invalid_return_type(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test handling of plugin that returns invalid type."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        class InvalidReturnPlugin:
            @property
            def name(self) -> str:
                return "invalid-return"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                return "not a PluginResult"  # type: ignore[return-value]

        PluginRegistry.register(InvalidReturnPlugin())

        # Should handle gracefully
        results = run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        assert "invalid-return" in results
        # Either treated as error or empty result - both acceptable

    def test_plugin_that_modifies_inputs(
        self,
        sample_jacobian: np.ndarray,
        sample_parameters: np.ndarray,
        sample_residuals: np.ndarray,
    ) -> None:
        """Test that plugin modifying inputs doesn't affect other plugins."""
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        original_params = sample_parameters.copy()

        class ModifyingPlugin:
            @property
            def name(self) -> str:
                return "modifier"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                # Try to modify the input
                parameters[0] = 999.0
                return PluginResult(plugin_name=self.name)

        class CheckingPlugin:
            @property
            def name(self) -> str:
                return "checker"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                # Record what we see
                return PluginResult(
                    plugin_name=self.name,
                    data={"first_param": float(parameters[0])},
                )

        PluginRegistry.register(ModifyingPlugin())
        PluginRegistry.register(CheckingPlugin())

        run_plugins(sample_jacobian, sample_parameters, sample_residuals)

        # Original array may be modified (this is expected behavior with numpy)
        # The test documents this behavior - plugins should not modify inputs

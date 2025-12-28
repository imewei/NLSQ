"""Tests for NLSQ GUI session state management.

This module tests the session state management infrastructure for the Streamlit GUI,
including state initialization, persistence, reset, and workflow config loading.
"""

from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import yaml

from nlsq.gui.adapters.config_adapter import (
    export_yaml_config,
    load_yaml_config,
)
from nlsq.gui.presets import (
    PRESETS,
    get_preset,
    get_preset_names,
)
from nlsq.gui.state import (
    SessionState,
    get_current_config,
    initialize_state,
    reset_state,
)


class TestSessionStateInitialization:
    """Tests for session state initialization."""

    def test_initialize_state_returns_session_state(self) -> None:
        """Test that initialize_state returns a SessionState instance."""
        state = initialize_state()
        assert isinstance(state, SessionState)

    def test_initialize_state_has_default_values(self) -> None:
        """Test that initialized state has expected default values."""
        state = initialize_state()

        # Check data-related fields have None defaults
        assert state.xdata is None
        assert state.ydata is None
        assert state.sigma is None

        # Check model-related fields have defaults
        assert state.model_type == "builtin"
        assert state.model_name == "exponential_decay"

        # Check fitting parameters have defaults
        assert state.gtol == 1e-8
        assert state.ftol == 1e-8
        assert state.xtol == 1e-8

        # Check workflow defaults
        assert state.preset == "standard"
        assert state.mode == "guided"

    def test_initialize_state_with_custom_values(self) -> None:
        """Test initialize_state with custom initial values."""
        state = initialize_state(
            model_name="gaussian",
            preset="robust",
            gtol=1e-10,
        )
        assert state.model_name == "gaussian"
        assert state.preset == "robust"
        assert state.gtol == 1e-10


class TestSessionStateReset:
    """Tests for session state reset functionality."""

    def test_reset_state_clears_data(self) -> None:
        """Test that reset_state clears data fields."""
        state = initialize_state()
        state.xdata = [1.0, 2.0, 3.0]
        state.ydata = [1.0, 4.0, 9.0]
        state.model_name = "gaussian"

        reset_state(state)

        assert state.xdata is None
        assert state.ydata is None

    def test_reset_state_preserves_preferences(self) -> None:
        """Test that reset_state preserves user preferences if requested."""
        state = initialize_state()
        state.mode = "advanced"
        state.preset = "quality"

        reset_state(state, preserve_preferences=True)

        assert state.mode == "advanced"
        assert state.preset == "quality"

    def test_reset_state_full_reset(self) -> None:
        """Test that full reset returns state to defaults."""
        state = initialize_state()
        state.mode = "advanced"
        state.preset = "quality"
        state.xdata = [1.0, 2.0, 3.0]

        reset_state(state, preserve_preferences=False)

        assert state.mode == "guided"
        assert state.preset == "standard"
        assert state.xdata is None


class TestGetCurrentConfig:
    """Tests for exporting state as workflow config dict."""

    def test_get_current_config_returns_dict(self) -> None:
        """Test that get_current_config returns a dictionary."""
        state = initialize_state()
        config = get_current_config(state)
        assert isinstance(config, dict)

    def test_get_current_config_contains_fitting_params(self) -> None:
        """Test that config contains fitting parameters."""
        state = initialize_state()
        state.gtol = 1e-10
        state.ftol = 1e-10
        state.xtol = 1e-10
        state.max_iterations = 500

        config = get_current_config(state)

        assert "fitting" in config
        assert config["fitting"]["termination"]["gtol"] == 1e-10
        assert config["fitting"]["termination"]["ftol"] == 1e-10
        assert config["fitting"]["termination"]["xtol"] == 1e-10
        assert config["fitting"]["termination"]["max_iterations"] == 500

    def test_get_current_config_contains_model_config(self) -> None:
        """Test that config contains model configuration."""
        state = initialize_state()
        state.model_type = "builtin"
        state.model_name = "gaussian"
        state.auto_p0 = True

        config = get_current_config(state)

        assert "model" in config
        assert config["model"]["type"] == "builtin"
        assert config["model"]["name"] == "gaussian"
        assert config["model"]["auto_p0"] is True


class TestWorkflowConfigLoading:
    """Tests for loading workflow config from YAML."""

    def test_load_yaml_config_basic(self) -> None:
        """Test loading a basic YAML config."""
        yaml_content = """
        model:
          type: builtin
          name: exponential_decay
          auto_p0: true
        fitting:
          termination:
            gtol: 1.0e-8
            ftol: 1.0e-8
            xtol: 1.0e-8
            max_iterations: 200
        """
        state = load_yaml_config(StringIO(yaml_content))

        assert state.model_type == "builtin"
        assert state.model_name == "exponential_decay"
        assert state.auto_p0 is True
        assert state.gtol == 1e-8
        assert state.max_iterations == 200

    def test_load_yaml_config_with_multistart(self) -> None:
        """Test loading YAML config with multi-start settings."""
        yaml_content = """
        fitting:
          multistart:
            enabled: true
            num_starts: 10
            sampler: lhs
            center_on_p0: true
        """
        state = load_yaml_config(StringIO(yaml_content))

        assert state.enable_multistart is True
        assert state.n_starts == 10
        assert state.sampler == "lhs"
        assert state.center_on_p0 is True

    def test_export_yaml_config_roundtrip(self) -> None:
        """Test that export and load are inverse operations."""
        original_state = initialize_state()
        original_state.model_name = "gaussian"
        original_state.gtol = 1e-10
        original_state.enable_multistart = True
        original_state.n_starts = 15

        # Export to YAML
        yaml_str = export_yaml_config(original_state)

        # Load back
        loaded_state = load_yaml_config(StringIO(yaml_str))

        # Verify key fields match
        assert loaded_state.model_name == original_state.model_name
        assert loaded_state.gtol == original_state.gtol
        assert loaded_state.enable_multistart == original_state.enable_multistart
        assert loaded_state.n_starts == original_state.n_starts


class TestPresets:
    """Tests for preset configurations."""

    def test_get_preset_names_returns_list(self) -> None:
        """Test that get_preset_names returns a list of preset names."""
        names = get_preset_names()
        assert isinstance(names, list)
        assert len(names) >= 3
        assert "fast" in names
        assert "robust" in names
        assert "quality" in names

    def test_get_preset_fast(self) -> None:
        """Test Fast preset configuration."""
        preset = get_preset("fast")
        assert preset["gtol"] == 1e-6
        assert preset["ftol"] == 1e-6
        assert preset["xtol"] == 1e-6
        assert preset["enable_multistart"] is False

    def test_get_preset_robust(self) -> None:
        """Test Robust preset configuration."""
        preset = get_preset("robust")
        assert preset["gtol"] == 1e-8
        assert preset["ftol"] == 1e-8
        assert preset["xtol"] == 1e-8
        assert preset["enable_multistart"] is True
        assert preset["n_starts"] == 10

    def test_get_preset_quality(self) -> None:
        """Test Quality preset configuration."""
        preset = get_preset("quality")
        assert preset["gtol"] == 1e-10
        assert preset["ftol"] == 1e-10
        assert preset["xtol"] == 1e-10
        assert preset["enable_multistart"] is True
        assert preset["n_starts"] == 20

    def test_presets_consistent_with_workflow_presets(self) -> None:
        """Test that GUI presets are consistent with WORKFLOW_PRESETS in minpack.py."""
        from nlsq.core.minpack import WORKFLOW_PRESETS

        # Fast preset should match
        gui_fast = get_preset("fast")
        minpack_fast = WORKFLOW_PRESETS["fast"]
        assert gui_fast["gtol"] == minpack_fast["gtol"]
        assert gui_fast["ftol"] == minpack_fast["ftol"]
        assert gui_fast["xtol"] == minpack_fast["xtol"]

        # Quality preset should match
        gui_quality = get_preset("quality")
        minpack_quality = WORKFLOW_PRESETS["quality"]
        assert gui_quality["gtol"] == minpack_quality["gtol"]
        assert gui_quality["ftol"] == minpack_quality["ftol"]
        assert gui_quality["xtol"] == minpack_quality["xtol"]
        assert gui_quality["n_starts"] == minpack_quality["n_starts"]

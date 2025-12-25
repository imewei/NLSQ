"""Tests for the advanced mode fitting options.

This module tests the advanced mode UI for the Fitting Options page,
including tabs for Fitting, Multi-start, Streaming, HPC, and Batch options.
"""

import pytest

from nlsq.gui.components.advanced_options import (
    FITTING_METHODS,
    LOSS_FUNCTIONS,
    SAMPLERS,
    SUMMARY_FORMATS,
    get_batch_tab_config,
    get_fitting_tab_config,
    get_hpc_tab_config,
    get_multistart_tab_config,
    get_streaming_tab_config,
    validate_chunk_size,
    validate_max_iterations,
    validate_n_starts,
)
from nlsq.gui.presets import (
    STREAMING_PRESETS,
    get_streaming_preset,
    get_streaming_preset_names,
)
from nlsq.gui.state import SessionState, initialize_state


class TestTabsRenderCorrectly:
    """Tests for tab availability and structure."""

    def test_fitting_methods_available(self) -> None:
        """Fitting methods should include trf, lm, dogbox."""
        assert "trf" in FITTING_METHODS
        assert "lm" in FITTING_METHODS
        assert "dogbox" in FITTING_METHODS

    def test_loss_functions_available(self) -> None:
        """Loss functions should include standard options."""
        assert "linear" in LOSS_FUNCTIONS
        assert "soft_l1" in LOSS_FUNCTIONS
        assert "huber" in LOSS_FUNCTIONS
        assert "cauchy" in LOSS_FUNCTIONS
        assert "arctan" in LOSS_FUNCTIONS

    def test_samplers_available(self) -> None:
        """Samplers should include lhs, sobol, halton."""
        assert "lhs" in SAMPLERS
        assert "sobol" in SAMPLERS
        assert "halton" in SAMPLERS

    def test_summary_formats_available(self) -> None:
        """Summary formats should include json, csv, txt."""
        assert "json" in SUMMARY_FORMATS
        assert "csv" in SUMMARY_FORMATS


class TestFittingTabFields:
    """Tests for Fitting tab field population."""

    def test_get_fitting_tab_config(self) -> None:
        """Fitting tab config should have all required fields."""
        state = initialize_state()
        config = get_fitting_tab_config(state)

        assert "method" in config
        assert "gtol" in config
        assert "ftol" in config
        assert "xtol" in config
        assert "max_iterations" in config
        assert "loss" in config

    def test_fitting_tab_default_values(self) -> None:
        """Fitting tab should have correct default values."""
        state = initialize_state()
        config = get_fitting_tab_config(state)

        assert config["method"] == "trf"
        assert config["gtol"] == 1e-8
        assert config["ftol"] == 1e-8
        assert config["xtol"] == 1e-8
        assert config["max_iterations"] == 200
        assert config["loss"] == "linear"

    def test_fitting_tab_custom_values(self) -> None:
        """Fitting tab should reflect custom state values."""
        state = initialize_state()
        state.method = "lm"
        state.gtol = 1e-12
        state.max_iterations = 1000
        state.loss = "huber"

        config = get_fitting_tab_config(state)

        assert config["method"] == "lm"
        assert config["gtol"] == 1e-12
        assert config["max_iterations"] == 1000
        assert config["loss"] == "huber"


class TestMultistartTabEnableDisable:
    """Tests for Multi-start tab enable/disable functionality."""

    def test_get_multistart_tab_config(self) -> None:
        """Multi-start tab config should have all required fields."""
        state = initialize_state()
        config = get_multistart_tab_config(state)

        assert "enabled" in config
        assert "n_starts" in config
        assert "sampler" in config
        assert "center_on_p0" in config
        assert "scale_factor" in config

    def test_multistart_disabled_by_default(self) -> None:
        """Multi-start should be disabled by default."""
        state = initialize_state()
        config = get_multistart_tab_config(state)

        assert config["enabled"] is False

    def test_multistart_enabled(self) -> None:
        """Multi-start can be enabled."""
        state = initialize_state()
        state.enable_multistart = True
        state.n_starts = 15

        config = get_multistart_tab_config(state)

        assert config["enabled"] is True
        assert config["n_starts"] == 15

    def test_multistart_sampler_options(self) -> None:
        """Multi-start sampler should be configurable."""
        state = initialize_state()
        state.sampler = "sobol"

        config = get_multistart_tab_config(state)

        assert config["sampler"] == "sobol"

    def test_validate_n_starts_valid(self) -> None:
        """Valid n_starts values should pass."""
        assert validate_n_starts(5) is True
        assert validate_n_starts(10) is True
        assert validate_n_starts(50) is True

    def test_validate_n_starts_invalid(self) -> None:
        """Invalid n_starts values should fail."""
        assert validate_n_starts(0) is False
        assert validate_n_starts(-1) is False


class TestStreamingTabChunkSize:
    """Tests for Streaming tab chunk size input."""

    def test_get_streaming_tab_config(self) -> None:
        """Streaming tab config should have all required fields."""
        state = initialize_state()
        config = get_streaming_tab_config(state)

        assert "chunk_size" in config
        assert "normalize" in config
        assert "warmup_iterations" in config
        assert "max_warmup_iterations" in config

    def test_streaming_default_values(self) -> None:
        """Streaming tab should have correct default values."""
        state = initialize_state()
        config = get_streaming_tab_config(state)

        assert config["chunk_size"] == 10000
        assert config["normalize"] is True
        assert config["warmup_iterations"] == 200

    def test_streaming_custom_chunk_size(self) -> None:
        """Custom chunk size should be reflected."""
        state = initialize_state()
        state.chunk_size = 50000

        config = get_streaming_tab_config(state)

        assert config["chunk_size"] == 50000

    def test_validate_chunk_size_valid(self) -> None:
        """Valid chunk sizes should pass."""
        assert validate_chunk_size(1000) is True
        assert validate_chunk_size(10000) is True
        assert validate_chunk_size(100000) is True

    def test_validate_chunk_size_invalid(self) -> None:
        """Invalid chunk sizes should fail."""
        assert validate_chunk_size(0) is False
        assert validate_chunk_size(-100) is False


class TestStreamingPresets:
    """Tests for streaming presets."""

    def test_streaming_preset_names(self) -> None:
        """Streaming presets should be available."""
        names = get_streaming_preset_names()

        assert "conservative" in names
        assert "aggressive" in names
        assert "memory_efficient" in names

    def test_streaming_preset_conservative(self) -> None:
        """Conservative streaming preset should have expected values."""
        preset = get_streaming_preset("conservative")

        assert preset["chunk_size"] == 10000
        assert preset["normalize"] is True

    def test_streaming_preset_aggressive(self) -> None:
        """Aggressive streaming preset should have larger chunk size."""
        preset = get_streaming_preset("aggressive")

        assert preset["chunk_size"] == 50000


class TestHPCTabOptions:
    """Tests for HPC tab options."""

    def test_get_hpc_tab_config(self) -> None:
        """HPC tab config should have all required fields."""
        state = initialize_state()
        config = get_hpc_tab_config(state)

        assert "enable_multi_device" in config
        assert "enable_checkpoints" in config
        assert "checkpoint_dir" in config

    def test_hpc_default_values(self) -> None:
        """HPC tab should have correct default values."""
        state = initialize_state()
        config = get_hpc_tab_config(state)

        assert config["enable_multi_device"] is False
        assert config["enable_checkpoints"] is False
        assert config["checkpoint_dir"] is None

    def test_hpc_custom_values(self) -> None:
        """HPC tab should reflect custom values."""
        state = initialize_state()
        state.enable_multi_device = True
        state.enable_checkpoints = True
        state.checkpoint_dir = "/tmp/checkpoints"

        config = get_hpc_tab_config(state)

        assert config["enable_multi_device"] is True
        assert config["enable_checkpoints"] is True
        assert config["checkpoint_dir"] == "/tmp/checkpoints"


class TestBatchTabOptions:
    """Tests for Batch tab options."""

    def test_get_batch_tab_config(self) -> None:
        """Batch tab config should have all required fields."""
        state = initialize_state()
        config = get_batch_tab_config(state)

        assert "max_workers" in config
        assert "continue_on_error" in config
        assert "summary_format" in config

    def test_batch_default_values(self) -> None:
        """Batch tab should have correct default values."""
        state = initialize_state()
        config = get_batch_tab_config(state)

        assert config["max_workers"] is None  # Auto
        assert config["continue_on_error"] is True
        assert config["summary_format"] == "json"

    def test_batch_custom_values(self) -> None:
        """Batch tab should reflect custom values."""
        state = initialize_state()
        state.batch_max_workers = 4
        state.batch_continue_on_error = False
        state.batch_summary_format = "csv"

        config = get_batch_tab_config(state)

        assert config["max_workers"] == 4
        assert config["continue_on_error"] is False
        assert config["summary_format"] == "csv"


class TestValidationFunctions:
    """Tests for input validation functions."""

    def test_validate_max_iterations_valid(self) -> None:
        """Valid max_iterations should pass."""
        assert validate_max_iterations(100) is True
        assert validate_max_iterations(1000) is True
        assert validate_max_iterations(10000) is True

    def test_validate_max_iterations_invalid(self) -> None:
        """Invalid max_iterations should fail."""
        assert validate_max_iterations(0) is False
        assert validate_max_iterations(-1) is False


class TestDefenseLayerSettings:
    """Tests for defense layer settings."""

    def test_defense_layer_defaults(self) -> None:
        """Defense layer settings should have defaults."""
        state = initialize_state()

        assert state.layer1_enabled is True
        assert state.layer2_enabled is True
        assert state.layer3_enabled is True
        assert state.layer4_enabled is True

    def test_defense_layer_customizable(self) -> None:
        """Defense layer settings should be customizable."""
        state = initialize_state()
        state.layer1_enabled = False
        state.layer3_tolerance = 0.1
        state.layer4_max_step = 0.05

        assert state.layer1_enabled is False
        assert state.layer3_tolerance == 0.1
        assert state.layer4_max_step == 0.05

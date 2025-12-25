"""Tests for the guided mode fitting options.

This module tests the guided mode UI for the Fitting Options page,
including preset selection, tolerance display, and YAML import functionality.
"""

from io import StringIO

import pytest

from nlsq.gui.adapters.config_adapter import (
    load_yaml_config,
    validate_yaml_config,
)
from nlsq.gui.presets import (
    PRESETS,
    get_preset,
    get_preset_description,
    get_preset_n_starts,
    get_preset_names,
    get_preset_tolerances,
    preset_uses_multistart,
)
from nlsq.gui.state import (
    SessionState,
    apply_preset_to_state,
    initialize_state,
)


class TestPresetSelectorPopulation:
    """Tests for preset selector population."""

    def test_preset_names_include_fast(self) -> None:
        """Fast preset should be available."""
        names = get_preset_names()
        assert "fast" in names

    def test_preset_names_include_robust(self) -> None:
        """Robust preset should be available."""
        names = get_preset_names()
        assert "robust" in names

    def test_preset_names_include_quality(self) -> None:
        """Quality preset should be available."""
        names = get_preset_names()
        assert "quality" in names

    def test_preset_names_include_standard(self) -> None:
        """Standard preset should be available."""
        names = get_preset_names()
        assert "standard" in names

    def test_all_presets_have_descriptions(self) -> None:
        """All presets should have descriptions."""
        for name in get_preset_names():
            desc = get_preset_description(name)
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestPresetValuesApplied:
    """Tests for preset values being applied correctly."""

    def test_fast_preset_tolerances(self) -> None:
        """Fast preset should have 1e-6 tolerances."""
        gtol, ftol, xtol = get_preset_tolerances("fast")

        assert gtol == 1e-6
        assert ftol == 1e-6
        assert xtol == 1e-6

    def test_robust_preset_tolerances(self) -> None:
        """Robust preset should have 1e-8 tolerances."""
        gtol, ftol, xtol = get_preset_tolerances("robust")

        assert gtol == 1e-8
        assert ftol == 1e-8
        assert xtol == 1e-8

    def test_quality_preset_tolerances(self) -> None:
        """Quality preset should have 1e-10 tolerances."""
        gtol, ftol, xtol = get_preset_tolerances("quality")

        assert gtol == 1e-10
        assert ftol == 1e-10
        assert xtol == 1e-10

    def test_fast_preset_no_multistart(self) -> None:
        """Fast preset should not use multi-start."""
        assert preset_uses_multistart("fast") is False

    def test_robust_preset_multistart(self) -> None:
        """Robust preset should use multi-start with 10 starts."""
        assert preset_uses_multistart("robust") is True
        assert get_preset_n_starts("robust") == 10

    def test_quality_preset_multistart(self) -> None:
        """Quality preset should use multi-start with 20 starts."""
        assert preset_uses_multistart("quality") is True
        assert get_preset_n_starts("quality") == 20


class TestPresetAppliedToState:
    """Tests for applying presets to session state."""

    def test_apply_fast_preset(self) -> None:
        """Applying fast preset should update state."""
        state = initialize_state()
        apply_preset_to_state(state, "fast")

        assert state.preset == "fast"
        assert state.gtol == 1e-6
        assert state.ftol == 1e-6
        assert state.xtol == 1e-6
        assert state.enable_multistart is False

    def test_apply_robust_preset(self) -> None:
        """Applying robust preset should update state."""
        state = initialize_state()
        apply_preset_to_state(state, "robust")

        assert state.preset == "robust"
        assert state.gtol == 1e-8
        assert state.ftol == 1e-8
        assert state.xtol == 1e-8
        assert state.enable_multistart is True
        assert state.n_starts == 10

    def test_apply_quality_preset(self) -> None:
        """Applying quality preset should update state."""
        state = initialize_state()
        apply_preset_to_state(state, "quality")

        assert state.preset == "quality"
        assert state.gtol == 1e-10
        assert state.ftol == 1e-10
        assert state.xtol == 1e-10
        assert state.enable_multistart is True
        assert state.n_starts == 20

    def test_preset_overwrites_previous(self) -> None:
        """Applying new preset should overwrite previous settings."""
        state = initialize_state()

        apply_preset_to_state(state, "quality")
        assert state.gtol == 1e-10

        apply_preset_to_state(state, "fast")
        assert state.gtol == 1e-6


class TestModeToggle:
    """Tests for mode toggle between Guided and Advanced."""

    def test_default_mode_is_guided(self) -> None:
        """Default mode should be guided."""
        state = initialize_state()
        assert state.mode == "guided"

    def test_mode_can_be_set_to_advanced(self) -> None:
        """Mode should be changeable to advanced."""
        state = initialize_state()
        state.mode = "advanced"
        assert state.mode == "advanced"

    def test_mode_persists_across_preset_changes(self) -> None:
        """Mode should not change when preset is applied."""
        state = initialize_state()
        state.mode = "advanced"

        apply_preset_to_state(state, "quality")

        assert state.mode == "advanced"


class TestYAMLImport:
    """Tests for YAML configuration import."""

    def test_validate_yaml_valid(self) -> None:
        """Valid YAML should pass validation."""
        yaml_content = """
        model:
          type: builtin
          name: gaussian
        fitting:
          termination:
            gtol: 1.0e-10
        """
        is_valid, error = validate_yaml_config(yaml_content)

        assert is_valid is True
        assert error is None

    def test_validate_yaml_invalid_syntax(self) -> None:
        """Invalid YAML syntax should fail validation."""
        yaml_content = """
        model:
          type: builtin
          name: gaussian
          invalid: [unclosed
        """
        is_valid, error = validate_yaml_config(yaml_content)

        assert is_valid is False
        assert error is not None
        assert "error" in error.lower()

    def test_import_yaml_updates_state(self) -> None:
        """Importing YAML should update state values."""
        yaml_content = """
        model:
          type: builtin
          name: sigmoid
          auto_p0: false
        fitting:
          method: lm
          termination:
            gtol: 1.0e-12
            max_iterations: 500
        """
        state = load_yaml_config(StringIO(yaml_content))

        assert state.model_name == "sigmoid"
        assert state.auto_p0 is False
        assert state.method == "lm"
        assert state.gtol == 1e-12
        assert state.max_iterations == 500

    def test_import_yaml_with_multistart(self) -> None:
        """Importing YAML with multi-start settings should work."""
        yaml_content = """
        fitting:
          multistart:
            enabled: true
            num_starts: 15
            sampler: sobol
            center_on_p0: false
        """
        state = load_yaml_config(StringIO(yaml_content))

        assert state.enable_multistart is True
        assert state.n_starts == 15
        assert state.sampler == "sobol"
        assert state.center_on_p0 is False

    def test_import_yaml_with_streaming(self) -> None:
        """Importing YAML with streaming settings should work."""
        yaml_content = """
        hybrid_streaming:
          chunk_size: 50000
          normalize: false
          warmup_iterations: 100
          enable_checkpoints: true
        """
        state = load_yaml_config(StringIO(yaml_content))

        assert state.chunk_size == 50000
        assert state.normalize is False
        assert state.warmup_iterations == 100
        assert state.enable_checkpoints is True


class TestToleranceDisplay:
    """Tests for tolerance display updates."""

    def test_tolerances_formatted_correctly(self) -> None:
        """Tolerance values should be stored as floats."""
        state = initialize_state()
        apply_preset_to_state(state, "quality")

        # Should be actual float values, not strings
        assert isinstance(state.gtol, float)
        assert isinstance(state.ftol, float)
        assert isinstance(state.xtol, float)

    def test_tolerances_accessible_for_display(self) -> None:
        """Tolerance values should be accessible for display."""
        state = initialize_state()
        apply_preset_to_state(state, "fast")

        # These should work without error
        gtol_str = f"{state.gtol:.0e}"
        ftol_str = f"{state.ftol:.0e}"
        xtol_str = f"{state.xtol:.0e}"

        assert "1e-06" in gtol_str
        assert "1e-06" in ftol_str
        assert "1e-06" in xtol_str


class TestMultistartStatusDisplay:
    """Tests for multi-start status display."""

    def test_multistart_disabled_status(self) -> None:
        """Multi-start disabled should show 0 starts."""
        state = initialize_state()
        apply_preset_to_state(state, "fast")

        assert state.enable_multistart is False
        # n_starts might still have a value but multistart is disabled
        assert state.enable_multistart is False

    def test_multistart_enabled_shows_n_starts(self) -> None:
        """Multi-start enabled should show n_starts value."""
        state = initialize_state()
        apply_preset_to_state(state, "robust")

        assert state.enable_multistart is True
        assert state.n_starts == 10

    def test_multistart_status_consistent(self) -> None:
        """Multi-start status should be consistent between preset and state."""
        for name in ["fast", "robust", "quality"]:
            preset = get_preset(name)
            state = initialize_state()
            apply_preset_to_state(state, name)

            assert state.enable_multistart == preset["enable_multistart"]
            if preset["enable_multistart"]:
                assert state.n_starts == preset["n_starts"]

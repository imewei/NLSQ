"""Tests for CMAESConfig dataclass and presets.

Tests cover:
- Configuration validation
- Preset configurations
- evosax availability checking
"""

from __future__ import annotations

import pytest

from nlsq.global_optimization.cmaes_config import (
    CMAES_PRESETS,
    CMAESConfig,
    get_evosax_import_error,
    is_evosax_available,
)


class TestCMAESConfig:
    """Tests for CMAESConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CMAESConfig()

        assert config.popsize is None  # Auto
        assert config.max_generations == 100
        assert config.sigma == 0.5
        assert config.tol_fun == 1e-8
        assert config.tol_x == 1e-8
        assert config.restart_strategy == "bipop"
        assert config.max_restarts == 9
        assert config.refine_with_nlsq is True
        assert config.seed is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CMAESConfig(
            popsize=32,
            max_generations=200,
            sigma=0.3,
            tol_fun=1e-10,
            tol_x=1e-10,
            restart_strategy="none",
            max_restarts=0,
            refine_with_nlsq=False,
            seed=42,
        )

        assert config.popsize == 32
        assert config.max_generations == 200
        assert config.sigma == 0.3
        assert config.tol_fun == 1e-10
        assert config.tol_x == 1e-10
        assert config.restart_strategy == "none"
        assert config.max_restarts == 0
        assert config.refine_with_nlsq is False
        assert config.seed == 42

    def test_validation_popsize_minimum(self) -> None:
        """Test that popsize must be >= 4 if specified."""
        with pytest.raises(ValueError, match="popsize must be >= 4"):
            CMAESConfig(popsize=3)

        # popsize=4 should work
        config = CMAESConfig(popsize=4)
        assert config.popsize == 4

    def test_validation_max_generations(self) -> None:
        """Test that max_generations must be >= 1."""
        with pytest.raises(ValueError, match="max_generations must be >= 1"):
            CMAESConfig(max_generations=0)

        # max_generations=1 should work
        config = CMAESConfig(max_generations=1)
        assert config.max_generations == 1

    def test_validation_sigma(self) -> None:
        """Test that sigma must be > 0."""
        with pytest.raises(ValueError, match="sigma must be > 0"):
            CMAESConfig(sigma=0.0)

        with pytest.raises(ValueError, match="sigma must be > 0"):
            CMAESConfig(sigma=-0.1)

    def test_validation_tol_fun(self) -> None:
        """Test that tol_fun must be > 0."""
        with pytest.raises(ValueError, match="tol_fun must be > 0"):
            CMAESConfig(tol_fun=0.0)

    def test_validation_tol_x(self) -> None:
        """Test that tol_x must be > 0."""
        with pytest.raises(ValueError, match="tol_x must be > 0"):
            CMAESConfig(tol_x=0.0)

    def test_validation_max_restarts(self) -> None:
        """Test that max_restarts must be >= 0."""
        with pytest.raises(ValueError, match="max_restarts must be >= 0"):
            CMAESConfig(max_restarts=-1)

        # max_restarts=0 should work (disables restarts)
        config = CMAESConfig(max_restarts=0)
        assert config.max_restarts == 0

    def test_validation_restart_strategy(self) -> None:
        """Test that restart_strategy must be 'none' or 'bipop'."""
        with pytest.raises(ValueError, match="restart_strategy must be"):
            CMAESConfig(restart_strategy="invalid")  # type: ignore

        # Valid strategies
        config_none = CMAESConfig(restart_strategy="none")
        assert config_none.restart_strategy == "none"

        config_bipop = CMAESConfig(restart_strategy="bipop")
        assert config_bipop.restart_strategy == "bipop"


class TestCMAESConfigPresets:
    """Tests for CMAESConfig.from_preset() and presets."""

    def test_presets_exist(self) -> None:
        """Test that expected presets exist."""
        expected_presets = {"cmaes-fast", "cmaes", "cmaes-global"}
        assert set(CMAES_PRESETS.keys()) == expected_presets

    def test_from_preset_cmaes_fast(self) -> None:
        """Test 'cmaes-fast' preset configuration."""
        config = CMAESConfig.from_preset("cmaes-fast")

        assert config.popsize is None  # Auto
        assert config.max_generations == 50
        assert config.restart_strategy == "none"
        assert config.max_restarts == 0

    def test_from_preset_cmaes(self) -> None:
        """Test 'cmaes' (default) preset configuration."""
        config = CMAESConfig.from_preset("cmaes")

        assert config.popsize is None  # Auto
        assert config.max_generations == 100
        assert config.restart_strategy == "bipop"
        assert config.max_restarts == 9

    def test_from_preset_cmaes_global(self) -> None:
        """Test 'cmaes-global' preset configuration."""
        config = CMAESConfig.from_preset("cmaes-global")

        assert config.popsize is None  # Will be doubled in optimizer
        assert config.max_generations == 200
        assert config.restart_strategy == "bipop"
        assert config.max_restarts == 9

    def test_from_preset_invalid(self) -> None:
        """Test that invalid preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            CMAESConfig.from_preset("invalid-preset")

    def test_from_preset_case_sensitive(self) -> None:
        """Test that preset names are case-sensitive."""
        with pytest.raises(ValueError, match="Unknown preset"):
            CMAESConfig.from_preset("CMAES")

        with pytest.raises(ValueError, match="Unknown preset"):
            CMAESConfig.from_preset("Cmaes-Fast")


class TestEvosaxAvailability:
    """Tests for evosax availability checking."""

    def test_is_evosax_available_returns_bool(self) -> None:
        """Test that is_evosax_available returns a boolean."""
        result = is_evosax_available()
        assert isinstance(result, bool)

    def test_is_evosax_available_cached(self) -> None:
        """Test that availability check is cached."""
        # Call twice - should return same result
        result1 = is_evosax_available()
        result2 = is_evosax_available()
        assert result1 == result2

    def test_get_evosax_import_error_when_available(self) -> None:
        """Test that import error is None when evosax is available."""
        if is_evosax_available():
            assert get_evosax_import_error() is None

    def test_get_evosax_import_error_when_unavailable(self) -> None:
        """Test that import error is set when evosax is unavailable."""
        if not is_evosax_available():
            error = get_evosax_import_error()
            assert error is not None
            assert isinstance(error, str)


class TestCMAESConfigSlots:
    """Tests for CMAESConfig __slots__ optimization."""

    def test_has_slots(self) -> None:
        """Test that CMAESConfig uses __slots__ for memory efficiency."""
        config = CMAESConfig()
        assert hasattr(config, "__slots__") or (
            hasattr(config.__class__, "__slots__")
            and config.__class__.__slots__ is not None
        )

    def test_no_dict_attribute(self) -> None:
        """Test that instances don't have __dict__ (slots optimization)."""
        config = CMAESConfig()
        # With slots=True, instances shouldn't have __dict__
        # unless they inherit from a class with __dict__
        # This test verifies the dataclass was created with slots=True
        assert not hasattr(config, "__dict__") or config.__dict__ == {}

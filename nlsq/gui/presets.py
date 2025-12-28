"""Preset configurations for the NLSQ GUI.

This module defines preset configurations for common curve fitting scenarios.
The presets are designed to be consistent with nlsq.minpack.WORKFLOW_PRESETS
while providing a GUI-friendly interface.

Presets:
    - fast: Speed-optimized with looser tolerances, no multi-start
    - robust: Balanced precision/speed, multi-start enabled (n_starts=10)
    - quality: Highest precision, tighter tolerances, multi-start (n_starts=20)
    - standard: Default settings (same as NLSQ defaults)
"""

from typing import Any

# GUI Preset configurations
# These are designed to match nlsq.minpack.WORKFLOW_PRESETS for consistency
PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "description": "Speed-optimized with looser tolerances",
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "enable_multistart": False,
        "n_starts": 0,
        "tier": "STANDARD",
    },
    "robust": {
        "description": "Balanced precision and speed with multi-start",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_multistart": True,
        "n_starts": 10,
        "tier": "STANDARD",
    },
    "quality": {
        "description": "Highest precision with multi-start for best results",
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
        "enable_multistart": True,
        "n_starts": 20,
        "tier": "STANDARD",
    },
    "standard": {
        "description": "Default NLSQ settings",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_multistart": False,
        "n_starts": 0,
        "tier": "STANDARD",
    },
}


def get_preset(name: str) -> dict[str, Any]:
    """Get a preset configuration by name.

    Parameters
    ----------
    name : str
        The preset name ("fast", "robust", "quality", "standard").

    Returns
    -------
    dict
        The preset configuration dictionary.

    Raises
    ------
    ValueError
        If the preset name is not recognized.

    Examples
    --------
    >>> preset = get_preset("fast")
    >>> preset["gtol"]
    1e-06
    >>> preset["enable_multistart"]
    False

    >>> preset = get_preset("quality")
    >>> preset["n_starts"]
    20
    """
    name_lower = name.lower()
    if name_lower not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
    return PRESETS[name_lower].copy()


def get_preset_names() -> list[str]:
    """Get a list of available preset names.

    Returns
    -------
    list[str]
        A list of preset names.

    Examples
    --------
    >>> names = get_preset_names()
    >>> "fast" in names
    True
    >>> "quality" in names
    True
    """
    return list(PRESETS.keys())


def get_preset_description(name: str) -> str:
    """Get the description for a preset.

    Parameters
    ----------
    name : str
        The preset name.

    Returns
    -------
    str
        A human-readable description of the preset.

    Raises
    ------
    ValueError
        If the preset name is not recognized.

    Examples
    --------
    >>> get_preset_description("fast")
    'Speed-optimized with looser tolerances'
    """
    preset = get_preset(name)
    return preset.get("description", "")


def get_preset_tolerances(name: str) -> tuple[float, float, float]:
    """Get the tolerances for a preset.

    Parameters
    ----------
    name : str
        The preset name.

    Returns
    -------
    tuple[float, float, float]
        A tuple of (gtol, ftol, xtol).

    Examples
    --------
    >>> gtol, ftol, xtol = get_preset_tolerances("quality")
    >>> gtol
    1e-10
    """
    preset = get_preset(name)
    return preset["gtol"], preset["ftol"], preset["xtol"]


def preset_uses_multistart(name: str) -> bool:
    """Check if a preset uses multi-start optimization.

    Parameters
    ----------
    name : str
        The preset name.

    Returns
    -------
    bool
        True if the preset enables multi-start.

    Examples
    --------
    >>> preset_uses_multistart("fast")
    False
    >>> preset_uses_multistart("robust")
    True
    """
    preset = get_preset(name)
    return preset.get("enable_multistart", False)


def get_preset_n_starts(name: str) -> int:
    """Get the number of multi-start points for a preset.

    Parameters
    ----------
    name : str
        The preset name.

    Returns
    -------
    int
        The number of starting points (0 if multi-start disabled).

    Examples
    --------
    >>> get_preset_n_starts("quality")
    20
    >>> get_preset_n_starts("fast")
    0
    """
    preset = get_preset(name)
    if not preset.get("enable_multistart", False):
        return 0
    return preset.get("n_starts", 10)


def validate_preset_consistency() -> bool:
    """Validate that GUI presets are consistent with WORKFLOW_PRESETS.

    This function checks that the GUI presets match the corresponding
    WORKFLOW_PRESETS in nlsq.minpack.

    Returns
    -------
    bool
        True if all presets are consistent.

    Raises
    ------
    AssertionError
        If any preset is inconsistent.
    """
    from nlsq.core.minpack import WORKFLOW_PRESETS

    # Check fast preset
    gui_fast = PRESETS["fast"]
    minpack_fast = WORKFLOW_PRESETS["fast"]
    assert gui_fast["gtol"] == minpack_fast["gtol"], "Fast gtol mismatch"
    assert gui_fast["ftol"] == minpack_fast["ftol"], "Fast ftol mismatch"
    assert gui_fast["xtol"] == minpack_fast["xtol"], "Fast xtol mismatch"
    assert gui_fast["enable_multistart"] == minpack_fast["enable_multistart"], (
        "Fast multistart mismatch"
    )

    # Check quality preset
    gui_quality = PRESETS["quality"]
    minpack_quality = WORKFLOW_PRESETS["quality"]
    assert gui_quality["gtol"] == minpack_quality["gtol"], "Quality gtol mismatch"
    assert gui_quality["ftol"] == minpack_quality["ftol"], "Quality ftol mismatch"
    assert gui_quality["xtol"] == minpack_quality["xtol"], "Quality xtol mismatch"
    assert gui_quality["enable_multistart"] == minpack_quality["enable_multistart"], (
        "Quality multistart mismatch"
    )
    assert gui_quality["n_starts"] == minpack_quality["n_starts"], (
        "Quality n_starts mismatch"
    )

    # Check standard preset
    gui_standard = PRESETS["standard"]
    minpack_standard = WORKFLOW_PRESETS["standard"]
    assert gui_standard["gtol"] == minpack_standard["gtol"], "Standard gtol mismatch"
    assert gui_standard["ftol"] == minpack_standard["ftol"], "Standard ftol mismatch"
    assert gui_standard["xtol"] == minpack_standard["xtol"], "Standard xtol mismatch"

    return True


# Streaming presets for large datasets
STREAMING_PRESETS: dict[str, dict[str, Any]] = {
    "conservative": {
        "description": "Conservative streaming with all defense layers enabled",
        "chunk_size": 10000,
        "normalize": True,
        "warmup_iterations": 200,
        "max_warmup_iterations": 500,
        "defense_preset": "default",
    },
    "aggressive": {
        "description": "Aggressive streaming for faster convergence",
        "chunk_size": 50000,
        "normalize": True,
        "warmup_iterations": 100,
        "max_warmup_iterations": 300,
        "defense_preset": "relaxed",
    },
    "memory_efficient": {
        "description": "Minimal memory footprint streaming",
        "chunk_size": 5000,
        "normalize": True,
        "warmup_iterations": 200,
        "max_warmup_iterations": 500,
        "defense_preset": "default",
        "enable_checkpoints": True,
    },
}


def get_streaming_preset(name: str) -> dict[str, Any]:
    """Get a streaming preset configuration by name.

    Parameters
    ----------
    name : str
        The streaming preset name.

    Returns
    -------
    dict
        The streaming preset configuration.

    Raises
    ------
    ValueError
        If the preset name is not recognized.
    """
    name_lower = name.lower()
    if name_lower not in STREAMING_PRESETS:
        available = ", ".join(STREAMING_PRESETS.keys())
        raise ValueError(f"Unknown streaming preset '{name}'. Available: {available}")
    return STREAMING_PRESETS[name_lower].copy()


def get_streaming_preset_names() -> list[str]:
    """Get a list of available streaming preset names.

    Returns
    -------
    list[str]
        A list of streaming preset names.
    """
    return list(STREAMING_PRESETS.keys())

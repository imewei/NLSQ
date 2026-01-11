"""Preset configurations for the NLSQ GUI.

This module defines preset configurations for common curve fitting scenarios.
The presets use the 3-workflow system (v0.6.3):

GUI Presets:
    - Fast: Local optimization with looser tolerances (workflow='auto')
    - Robust: Global optimization with default tolerances (workflow='auto_global')
    - Quality: Global optimization with tighter tolerances (workflow='auto_global')

Workflows (v0.6.3):
    - 'auto': Memory-aware local optimization (default)
    - 'auto_global': Memory-aware global optimization (requires bounds)
    - 'hpc': auto_global + checkpointing for HPC environments
"""

from typing import Any

# GUI Preset configurations using 3-workflow system (v0.6.3)
PRESETS: dict[str, dict[str, Any]] = {
    "Fast": {
        "description": "Speed-optimized local optimization with looser tolerances",
        "workflow": "auto",
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "max_iterations": 100,
    },
    "Robust": {
        "description": "Global optimization with multi-start for reliable results",
        "workflow": "auto_global",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "max_iterations": 200,
        "n_starts": 5,
    },
    "Quality": {
        "description": "Highest precision global optimization with tighter tolerances",
        "workflow": "auto_global",
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
        "max_iterations": 500,
        "n_starts": 10,
    },
}


def get_preset(name: str) -> dict[str, Any]:
    """Get a preset configuration by name.

    Parameters
    ----------
    name : str
        The preset name. Available presets: "Fast", "Robust", "Quality".

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
    >>> preset = get_preset("Fast")
    >>> preset["gtol"]
    1e-06
    >>> preset["workflow"]
    'auto'

    >>> preset = get_preset("Quality")
    >>> preset["n_starts"]
    10
    """
    # Try exact match first
    if name in PRESETS:
        return PRESETS[name].copy()

    # Try case-insensitive match
    name_map = {k.lower(): k for k in PRESETS}
    if name.lower() in name_map:
        return PRESETS[name_map[name.lower()]].copy()

    available = ", ".join(PRESETS.keys())
    raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")


def get_preset_names() -> list[str]:
    """Get a list of available preset names.

    Returns
    -------
    list[str]
        A list of preset names.

    Examples
    --------
    >>> names = get_preset_names()
    >>> "Fast" in names
    True
    >>> "Quality" in names
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
    >>> get_preset_description("Fast")
    'Speed-optimized local optimization with looser tolerances'
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
    >>> gtol, ftol, xtol = get_preset_tolerances("Quality")
    >>> gtol
    1e-10
    """
    preset = get_preset(name)
    return preset["gtol"], preset["ftol"], preset["xtol"]


def preset_uses_global_optimization(name: str) -> bool:
    """Check if a preset uses global optimization (auto_global workflow).

    Parameters
    ----------
    name : str
        The preset name.

    Returns
    -------
    bool
        True if the preset uses workflow='auto_global'.

    Examples
    --------
    >>> preset_uses_global_optimization("Fast")
    False
    >>> preset_uses_global_optimization("Robust")
    True
    """
    preset = get_preset(name)
    return preset.get("workflow") == "auto_global"


def get_preset_n_starts(name: str) -> int:
    """Get the number of global optimization starts for a preset.

    Parameters
    ----------
    name : str
        The preset name.

    Returns
    -------
    int
        The number of starting points (0 if using local optimization).

    Examples
    --------
    >>> get_preset_n_starts("Quality")
    10
    >>> get_preset_n_starts("Fast")
    0
    """
    preset = get_preset(name)
    if preset.get("workflow") != "auto_global":
        return 0
    return preset.get("n_starts", 10)


# Streaming presets for large datasets (used with workflow='auto')
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

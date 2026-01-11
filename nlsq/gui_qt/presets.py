"""Preset configurations for the NLSQ GUI.

This module defines preset configurations for common curve fitting scenarios.
The presets are synchronized with nlsq.core.minpack.WORKFLOW_PRESETS to ensure
consistent behavior between CLI and GUI interfaces.

Core Presets (matching WORKFLOW_PRESETS):
    - standard: Default curve_fit() with default tolerances (1e-8)
    - fast: Speed-optimized with looser tolerances (1e-6), no multi-start
    - quality: Highest precision, tighter tolerances (1e-10), multi-start (n_starts=20)
    - large_robust: Chunked processing with multi-start for large datasets
    - streaming: AdaptiveHybridStreamingOptimizer for huge datasets
    - hpc_distributed: Multi-GPU/node configuration for HPC clusters

Global Optimization Presets:
    - cmaes: CMA-ES global optimization with BIPOP restarts (100 gens)
    - cmaes-global: Thorough CMA-ES exploration (200 gens, 2x population)
    - global_auto: Auto-selects CMA-ES or Multi-Start based on parameter scale
"""

from typing import Any

# GUI Preset configurations
# Synchronized with nlsq.core.minpack.WORKFLOW_PRESETS for consistency
PRESETS: dict[str, dict[str, Any]] = {
    # === Core Presets (matching WORKFLOW_PRESETS) ===
    "standard": {
        "description": "Standard curve_fit() with default tolerances",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_multistart": False,
        "n_starts": 0,
        "tier": "STANDARD",
    },
    "fast": {
        "description": "Speed-optimized with looser tolerances",
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "enable_multistart": False,
        "n_starts": 0,
        "tier": "STANDARD",
    },
    "quality": {
        "description": "Highest precision with multi-start and tighter tolerances",
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
        "enable_multistart": True,
        "n_starts": 20,
        "tier": "STANDARD",
    },
    "large_robust": {
        "description": "Chunked processing with multi-start for large datasets",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_multistart": True,
        "n_starts": 10,
        "tier": "CHUNKED",
    },
    "streaming": {
        "description": "AdaptiveHybridStreamingOptimizer for huge datasets",
        "gtol": 1e-7,
        "ftol": 1e-7,
        "xtol": 1e-7,
        "enable_multistart": False,
        "n_starts": 0,
        "tier": "STREAMING",
    },
    "hpc_distributed": {
        "description": "Multi-GPU/node configuration for HPC clusters",
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "enable_multistart": True,
        "n_starts": 10,
        "enable_checkpoints": True,
        "tier": "STREAMING_CHECKPOINT",
    },
    # === Global Optimization Presets ===
    "cmaes": {
        "description": "CMA-ES global optimization with BIPOP restarts",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_multistart": False,
        "n_starts": 0,
        "method": "cmaes",
        "cmaes_preset": "cmaes",
        "tier": "STANDARD",
    },
    "cmaes-global": {
        "description": "Thorough CMA-ES exploration with extended generations",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_multistart": False,
        "n_starts": 0,
        "method": "cmaes",
        "cmaes_preset": "cmaes-global",
        "tier": "STANDARD",
    },
    "global_auto": {
        "description": "Auto-selects CMA-ES or Multi-Start based on parameter scale",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_multistart": True,
        "n_starts": 10,
        "method": "auto",
        "cmaes_preset": "cmaes",
        "tier": "STANDARD",
    },
}


def get_preset(name: str) -> dict[str, Any]:
    """Get a preset configuration by name.

    Parameters
    ----------
    name : str
        The preset name. Available presets:
        - Core: "standard", "fast", "quality", "large_robust", "streaming", "hpc_distributed"
        - Global: "cmaes", "cmaes-global", "global_auto"

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

    >>> preset = get_preset("cmaes")
    >>> preset["method"]
    'cmaes'
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
    WORKFLOW_PRESETS in nlsq.core.minpack. Presets that exist only in GUI
    (like "robust") are not validated.

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

    # Presets that should match between GUI and WORKFLOW_PRESETS
    presets_to_validate = [
        "standard",
        "fast",
        "quality",
        "large_robust",
        "streaming",
        "hpc_distributed",
        "cmaes",
        "cmaes-global",
        "global_auto",
    ]

    for name in presets_to_validate:
        gui_preset = PRESETS[name]
        minpack_preset = WORKFLOW_PRESETS[name]

        # Check tolerances
        assert gui_preset["gtol"] == minpack_preset["gtol"], f"{name} gtol mismatch"
        assert gui_preset["ftol"] == minpack_preset["ftol"], f"{name} ftol mismatch"
        assert gui_preset["xtol"] == minpack_preset["xtol"], f"{name} xtol mismatch"

        # Check multistart settings
        assert gui_preset["enable_multistart"] == minpack_preset["enable_multistart"], (
            f"{name} enable_multistart mismatch"
        )

        # Check n_starts if multistart is enabled
        if minpack_preset.get("n_starts") is not None:
            assert gui_preset["n_starts"] == minpack_preset["n_starts"], (
                f"{name} n_starts mismatch"
            )

        # Check tier
        assert gui_preset["tier"] == minpack_preset["tier"], f"{name} tier mismatch"

        # Check method if specified
        if "method" in minpack_preset:
            assert gui_preset.get("method") == minpack_preset["method"], (
                f"{name} method mismatch"
            )

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

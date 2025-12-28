"""
Converted from 05_yaml_configuration.ipynb

This script was automatically generated from a Jupyter notebook.

Features demonstrated:
- Create an nlsq.yaml configuration file
- Configure tolerances, memory limits, and checkpointing via YAML
- Use load_yaml_config() and get_custom_workflow() functions
- Override YAML settings with environment variables

Run this example:
    python examples/scripts/08_workflow_system/05_yaml_configuration.py
"""

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

try:
    import yaml
except ImportError:
    print("pyyaml is required. Install with: pip install pyyaml")
    raise

from nlsq import curve_fit

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
from nlsq.core.workflow import (
    WorkflowConfig,
    get_custom_workflow,
    get_env_overrides,
    load_config_with_overrides,
    load_yaml_config,
)


def exponential_decay(x, a, b, c):
    """Exponential decay model."""
    return a * jnp.exp(-b * x) + c


def fit_with_yaml_config(
    f,
    xdata,
    ydata,
    p0=None,
    bounds=(-np.inf, np.inf),
    workflow_name=None,
    config_path="nlsq.yaml",
):
    """Curve fit using YAML-defined workflow configuration.

    Parameters
    ----------
    f : callable
        Model function
    xdata, ydata : array_like
        Data to fit
    p0 : array_like, optional
        Initial parameters
    bounds : tuple, optional
        Parameter bounds
    workflow_name : str, optional
        Name of workflow from nlsq.yaml. If None, uses default_workflow.
    config_path : str, optional
        Path to YAML config file

    Returns
    -------
    popt, pcov : tuple
        Fitted parameters and covariance
    """
    # Load YAML config
    yaml_config = load_yaml_config(config_path)

    if yaml_config is None:
        print("No YAML config found, using defaults")
        return curve_fit(f, xdata, ydata, p0=p0, bounds=bounds)

    # Determine workflow to use
    if workflow_name is None:
        workflow_name = yaml_config.get("default_workflow", "standard")

    # Try to get custom workflow
    workflow = get_custom_workflow(workflow_name, config_path)

    if workflow is None:
        # Fall back to preset
        try:
            workflow = WorkflowConfig.from_preset(workflow_name)
        except ValueError:
            workflow = WorkflowConfig()  # Default

    # Apply workflow settings
    return curve_fit(
        f,
        xdata,
        ydata,
        p0=p0,
        bounds=bounds,
        gtol=workflow.gtol,
        ftol=workflow.ftol,
        xtol=workflow.xtol,
        multistart=workflow.enable_multistart,
        n_starts=workflow.n_starts if workflow.enable_multistart else 0,
        sampler=workflow.sampler,
    )


def main():
    print("=" * 70)
    print("YAML Configuration for NLSQ Workflows")
    print("=" * 70)
    print()

    # Set random seed
    np.random.seed(42)

    # =========================================================================
    # 1. Create example nlsq.yaml
    # =========================================================================
    print("1. Creating example nlsq.yaml...")

    complete_config = {
        "default_workflow": "standard",
        "memory_limit_gb": 16.0,
        "workflows": {
            "high_precision": {
                "tier": "STANDARD",
                "goal": "QUALITY",
                "gtol": 1e-10,
                "ftol": 1e-10,
                "xtol": 1e-10,
                "enable_multistart": True,
                "n_starts": 4 if QUICK else 20,
                "sampler": "lhs",
            },
            "quick_explore": {
                "tier": "STANDARD",
                "goal": "FAST",
                "gtol": 1e-5,
                "ftol": 1e-5,
                "xtol": 1e-5,
                "enable_multistart": False,
            },
            "large_data": {
                "tier": "CHUNKED",
                "goal": "ROBUST",
                "gtol": 1e-8,
                "ftol": 1e-8,
                "xtol": 1e-8,
                "memory_limit_gb": 8.0,
                "chunk_size": 100000,
                "enable_multistart": True,
                "n_starts": 4 if QUICK else 10,
            },
            "hpc_checkpoint": {
                "tier": "STREAMING_CHECKPOINT",
                "goal": "ROBUST",
                "gtol": 1e-7,
                "ftol": 1e-7,
                "xtol": 1e-7,
                "enable_checkpoints": True,
                "checkpoint_dir": "./checkpoints",
                "enable_multistart": True,
                "n_starts": 4 if QUICK else 10,
            },
        },
    }

    config_path = Path("nlsq.yaml")
    with open(config_path, "w") as f:
        yaml.dump(complete_config, f, default_flow_style=False, sort_keys=False)

    print("  Created nlsq.yaml")
    print()
    print("  Contents:")
    print("  " + "-" * 40)
    for line in config_path.read_text().split("\n")[:20]:
        print(f"  {line}")
    print("  ...")

    # =========================================================================
    # 2. Load YAML configuration
    # =========================================================================
    print()
    print("2. Loading YAML configuration...")

    config = load_yaml_config()

    print(f"  default_workflow: {config.get('default_workflow')}")
    print(f"  memory_limit_gb: {config.get('memory_limit_gb')}")
    print(f"  workflows defined: {list(config.get('workflows', {}).keys())}")

    # =========================================================================
    # 3. Get custom workflows
    # =========================================================================
    print()
    print("3. Getting custom workflows:")

    for wf_name in ["high_precision", "large_data", "hpc_checkpoint"]:
        workflow = get_custom_workflow(wf_name)
        if workflow:
            print(f"\n  {wf_name}:")
            print(f"    tier: {workflow.tier}")
            print(f"    goal: {workflow.goal}")
            print(f"    gtol: {workflow.gtol}")
            if workflow.enable_multistart:
                print(f"    n_starts: {workflow.n_starts}")
            if workflow.enable_checkpoints:
                print(f"    checkpoint_dir: {workflow.checkpoint_dir}")

    # =========================================================================
    # 4. Environment variable overrides
    # =========================================================================
    print()
    print("4. Environment variable overrides:")

    # Set environment variables
    os.environ["NLSQ_WORKFLOW_GOAL"] = "quality"
    os.environ["NLSQ_MEMORY_LIMIT_GB"] = "32.0"

    env_overrides = get_env_overrides()

    print("  Set environment variables:")
    print("    NLSQ_WORKFLOW_GOAL=quality")
    print("    NLSQ_MEMORY_LIMIT_GB=32.0")
    print()
    print("  Detected overrides:")
    for key, value in env_overrides.items():
        print(f"    {key}: {value}")

    # Load merged config
    merged_config = load_config_with_overrides()
    print()
    print("  Merged configuration:")
    print(f"    goal: {merged_config.get('goal')}")
    print(f"    memory_limit_gb: {merged_config.get('memory_limit_gb')}")

    # Clean up
    del os.environ["NLSQ_WORKFLOW_GOAL"]
    del os.environ["NLSQ_MEMORY_LIMIT_GB"]

    # =========================================================================
    # 5. Using custom workflows with curve_fit
    # =========================================================================
    print()
    print("5. Using custom workflows with curve_fit:")

    # Generate test data
    x_data = np.linspace(0, 5, 100 if QUICK else 300)
    true_a, true_b, true_c = 2.5, 1.2, 0.5
    y_true = true_a * np.exp(-true_b * x_data) + true_c
    y_data = y_true + 0.1 * np.random.randn(len(x_data))

    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}")

    # Load and use high_precision workflow
    workflow = get_custom_workflow("high_precision")

    if workflow:
        popt, _ = curve_fit(
            exponential_decay,
            x_data,
            y_data,
            p0=[1.0, 1.0, 0.0],
            bounds=([0, 0, -1], [10, 5, 2]),
            gtol=workflow.gtol,
            ftol=workflow.ftol,
            xtol=workflow.xtol,
            multistart=workflow.enable_multistart,
            n_starts=workflow.n_starts if workflow.enable_multistart else 0,
            sampler=workflow.sampler,
        )

        print()
        print("  high_precision workflow result:")
        print(f"    a={popt[0]:.6f}, b={popt[1]:.6f}, c={popt[2]:.6f}")
        print(f"    Settings: gtol={workflow.gtol}, n_starts={workflow.n_starts}")

    # =========================================================================
    # 6. Using helper function
    # =========================================================================
    print()
    print("6. Using fit_with_yaml_config() helper:")

    popt1, _ = fit_with_yaml_config(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=([0, 0, -1], [10, 5, 2]),
        workflow_name="high_precision",
    )
    print(f"  high_precision: a={popt1[0]:.4f}, b={popt1[1]:.4f}, c={popt1[2]:.4f}")

    popt2, _ = fit_with_yaml_config(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=([0, 0, -1], [10, 5, 2]),
        workflow_name="quick_explore",
    )
    print(f"  quick_explore:  a={popt2[0]:.4f}, b={popt2[1]:.4f}, c={popt2[2]:.4f}")

    # =========================================================================
    # Cleanup and Summary
    # =========================================================================
    if config_path.exists():
        config_path.unlink()
        print()
        print("  Cleaned up nlsq.yaml")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("YAML configuration enables:")
    print("  - Reproducible workflow settings")
    print("  - Easy sharing between collaborators")
    print("  - Environment-specific overrides")
    print()
    print("Key functions:")
    print("  - load_yaml_config(path)")
    print("  - get_custom_workflow(name)")
    print("  - load_config_with_overrides()")
    print("  - get_env_overrides()")
    print()
    print("Environment variables:")
    print("  - NLSQ_WORKFLOW_GOAL")
    print("  - NLSQ_MEMORY_LIMIT_GB")
    print("  - NLSQ_DEFAULT_WORKFLOW")
    print("  - NLSQ_CHECKPOINT_DIR")


if __name__ == "__main__":
    main()

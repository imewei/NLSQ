"""XPCS (X-ray Photon Correlation Spectroscopy) Domain Preset Example.

This example demonstrates how to create a custom preset for XPCS data analysis
using NLSQ's WorkflowConfig.from_preset().with_overrides() pattern.

XPCS analysis typically involves:
- Fitting correlation functions (g2) to extract relaxation times
- Multi-scale parameters (tau can span nanoseconds to hours)
- High precision requirements for publication-quality results
- Potential for large datasets from modern 2D detectors

Run this example:
    python examples/scripts/domain_presets/xpcs_preset.py
"""

import jax.numpy as jnp
import numpy as np

from nlsq import fit
from nlsq.core.workflow import WorkflowConfig


def create_xpcs_preset() -> WorkflowConfig:
    """Create a workflow configuration optimized for XPCS correlation function fitting.

    XPCS-specific considerations:
    - Tolerances: 1e-8 provides sufficient precision for correlation functions
    - Multi-start: Enabled to avoid local minima in stretched exponential fits
    - Normalization: Important for parameters spanning many orders of magnitude

    Returns
    -------
    WorkflowConfig
        Configuration optimized for XPCS analysis.

    Example
    -------
    >>> config = create_xpcs_preset()
    >>> config.gtol
    1e-08
    >>> config.enable_multistart
    True
    """
    # Start from precision_standard preset and customize for XPCS
    # precision_standard provides:
    #   - gtol=ftol=xtol=1e-8 (appropriate for correlation analysis)
    #   - enable_multistart=True (helps with stretched exponential fits)
    #   - n_starts=10 (reasonable for 3-4 parameter fits)
    #   - sampler='lhs' (good coverage of parameter space)

    config = WorkflowConfig.from_preset("precision_standard").with_overrides(
        # XPCS-specific overrides:

        # Increase n_starts for stretched exponential fits which can have
        # multiple local minima depending on the stretching exponent
        n_starts=15,

        # Use Sobol sampling for better coverage of multi-dimensional
        # parameter spaces (tau, beta, baseline, contrast)
        sampler="sobol",
    )

    return config


def xpcs_g2_model(t, tau, beta, baseline, contrast):
    """XPCS correlation function model (g2 - 1).

    The normalized intensity autocorrelation function:
        g2(t) - 1 = contrast * exp(-2 * (t/tau)^beta)

    Parameters
    ----------
    t : array_like
        Delay times (typically logarithmically spaced)
    tau : float
        Relaxation time (characteristic decay time)
    beta : float
        Stretching exponent (0 < beta <= 1 for subdiffusive, beta > 1 for superdiffusive)
    baseline : float
        Baseline offset (ideally 0, but may drift in practice)
    contrast : float
        Speckle contrast (ideally close to theoretical maximum)

    Returns
    -------
    array
        g2(t) - 1 values at each delay time
    """
    return baseline + contrast * jnp.exp(-2.0 * (t / tau) ** beta)


def main():
    print("=" * 70)
    print("XPCS Domain Preset Example")
    print("=" * 70)
    print()

    # Create the XPCS preset
    config = create_xpcs_preset()

    print("XPCS Preset Configuration:")
    print("-" * 40)
    print(f"  Tier:              {config.tier.name}")
    print(f"  Goal:              {config.goal.name}")
    print(f"  gtol:              {config.gtol}")
    print(f"  ftol:              {config.ftol}")
    print(f"  xtol:              {config.xtol}")
    print(f"  enable_multistart: {config.enable_multistart}")
    print(f"  n_starts:          {config.n_starts}")
    print(f"  sampler:           {config.sampler}")
    print()

    # Generate synthetic XPCS data
    print("Generating synthetic XPCS data...")
    np.random.seed(42)

    # Logarithmically spaced delay times (typical for correlation functions)
    t_data = np.logspace(-6, 2, 100)  # 1 us to 100 s

    # True parameters
    true_params = {
        "tau": 0.01,       # 10 ms relaxation time
        "beta": 0.8,       # Stretched exponential (subdiffusive dynamics)
        "baseline": 0.0,   # Ideal baseline
        "contrast": 0.3,   # Typical speckle contrast
    }

    # Generate noisy data
    y_true = xpcs_g2_model(
        t_data,
        true_params["tau"],
        true_params["beta"],
        true_params["baseline"],
        true_params["contrast"],
    )
    noise = 0.01 * np.random.randn(len(t_data))
    y_data = y_true + noise

    print(f"  Data points: {len(t_data)}")
    print(f"  True tau:    {true_params['tau']:.4f} s")
    print(f"  True beta:   {true_params['beta']:.2f}")
    print()

    # Initial guesses and bounds
    # Note: Multi-scale parameters require careful bounds
    p0 = [0.1, 0.9, 0.0, 0.25]  # [tau, beta, baseline, contrast]

    bounds = (
        [1e-8, 0.1, -0.1, 0.01],   # Lower bounds
        [1e3, 2.0, 0.1, 1.0],      # Upper bounds
    )

    # Fit using the XPCS preset
    print("Fitting with XPCS preset...")
    popt, pcov = fit(
        xpcs_g2_model,
        t_data,
        y_data,
        p0=p0,
        bounds=bounds,
        workflow_config=config,
    )

    # Results
    print()
    print("Fit Results:")
    print("-" * 40)
    print(f"  tau:      {popt[0]:.6f} s (true: {true_params['tau']:.6f})")
    print(f"  beta:     {popt[1]:.4f} (true: {true_params['beta']:.4f})")
    print(f"  baseline: {popt[2]:.6f} (true: {true_params['baseline']:.6f})")
    print(f"  contrast: {popt[3]:.4f} (true: {true_params['contrast']:.4f})")

    # Parameter uncertainties from covariance
    if pcov is not None:
        perr = np.sqrt(np.diag(pcov))
        print()
        print("Parameter Uncertainties (1-sigma):")
        print(f"  tau:      +/- {perr[0]:.6f} s")
        print(f"  beta:     +/- {perr[1]:.6f}")
        print(f"  baseline: +/- {perr[2]:.6f}")
        print(f"  contrast: +/- {perr[3]:.6f}")

    print()
    print("=" * 70)
    print("Notes on XPCS Preset Customization")
    print("=" * 70)
    print()
    print("The XPCS preset builds on 'precision_standard' with adjustments for:")
    print()
    print("1. Multi-start optimization (n_starts=15)")
    print("   - Stretched exponential fits often have multiple local minima")
    print("   - Especially important when beta is unknown a priori")
    print()
    print("2. Sobol sampling")
    print("   - Better coverage of the tau/beta parameter space")
    print("   - More uniform exploration than random sampling")
    print()
    print("3. Parameter bounds guidance:")
    print("   - tau: Set bounds based on experimental time window")
    print("   - beta: [0.1, 2.0] covers sub- and super-diffusive regimes")
    print("   - contrast: [0, 1] for normalized correlation functions")
    print()


if __name__ == "__main__":
    main()

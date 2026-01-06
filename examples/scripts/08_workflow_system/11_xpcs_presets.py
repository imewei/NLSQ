"""XPCS (X-ray Photon Correlation Spectroscopy) Domain Preset Example.

This example demonstrates how to create a custom preset for XPCS data analysis
using NLSQ's fit() function with custom kwargs.

XPCS analysis typically involves:
- Fitting correlation functions (g2) to extract relaxation times
- Multi-scale parameters (tau can span nanoseconds to hours)
- High precision requirements for publication-quality results
- Potential for large datasets from modern 2D detectors

Run this example:
    python examples/scripts/08_workflow_system/11_xpcs_presets.py
"""

import jax.numpy as jnp
import numpy as np

from nlsq import fit


def create_xpcs_preset() -> dict:
    """Create fit kwargs optimized for XPCS correlation function fitting.

    XPCS-specific considerations:
    - Tolerances: 1e-8 provides sufficient precision for correlation functions
    - Multi-start: Enabled to avoid local minima in stretched exponential fits
    - Sobol sampler: Better coverage for multi-scale parameter spaces

    Returns
    -------
    dict
        Keyword arguments for fit() optimized for XPCS analysis.

    Example
    -------
    >>> kwargs = create_xpcs_preset()
    >>> popt, pcov = fit(model, t, g2, **kwargs)
    """
    # XPCS typically needs multi-start due to stretched exponential fits
    # which can have multiple local minima depending on the stretching exponent
    return {
        "workflow": "standard",
        # Multi-start is critical for:
        # 1. Stretched exponential fits with unknown beta
        # 2. Multi-tau correlation functions
        "multistart": True,
        "n_starts": 15,  # More starts for stretched exponential fits
        # Sobol sampling for better coverage of multi-dimensional
        # parameter spaces (tau, beta, baseline, contrast)
        "sampler": "sobol",
        # Tolerances appropriate for correlation analysis
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
    }


def create_xpcs_high_precision_preset() -> dict:
    """Create fit kwargs for high-precision XPCS analysis.

    Use this for:
    - Publication-quality relaxation time determination
    - Small stretching exponent differences
    - Multi-component dynamics analysis

    Returns
    -------
    dict
        Keyword arguments for fit() with high precision settings.
    """
    return {
        "workflow": "quality",
        "multistart": True,
        "n_starts": 25,
        "sampler": "sobol",
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
    }


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

    np.random.seed(42)

    # =========================================================================
    # 1. Standard XPCS Fitting
    # =========================================================================
    print("1. Standard XPCS Correlation Function Fitting:")
    print("-" * 50)

    # Logarithmically spaced delay times (typical for correlation functions)
    t_data = np.logspace(-6, 2, 100)  # 1 us to 100 s

    # True parameters
    true_tau = 0.01  # 10 ms relaxation time
    true_beta = 0.8  # Stretched exponential (subdiffusive dynamics)
    true_baseline = 0.0  # Ideal baseline
    true_contrast = 0.3  # Typical speckle contrast

    # Generate noisy data
    y_true = (
        true_contrast * np.exp(-2.0 * (t_data / true_tau) ** true_beta) + true_baseline
    )
    y_data = y_true + 0.01 * np.random.randn(len(t_data))

    print("  True parameters:")
    print(f"    tau: {true_tau:.4f} s")
    print(f"    beta: {true_beta:.2f}")
    print(f"    contrast: {true_contrast:.2f}")

    # Get XPCS preset kwargs
    kwargs = create_xpcs_preset()
    print(f"\n  XPCS preset: {kwargs}")

    # Initial guesses and bounds
    p0 = [0.1, 0.9, 0.0, 0.25]

    bounds = (
        [1e-8, 0.1, -0.1, 0.01],  # Lower bounds
        [1e3, 2.0, 0.1, 1.0],  # Upper bounds
    )

    # Fit
    popt, pcov = fit(
        xpcs_g2_model,
        t_data,
        y_data,
        p0=p0,
        bounds=bounds,
        **kwargs,
    )

    print("\n  Fitted parameters:")
    print(f"    tau: {popt[0]:.6f} s (true: {true_tau})")
    print(f"    beta: {popt[1]:.4f} (true: {true_beta})")
    print(f"    baseline: {popt[2]:.6f} (true: {true_baseline})")
    print(f"    contrast: {popt[3]:.4f} (true: {true_contrast})")

    # =========================================================================
    # 2. High-Precision XPCS Analysis
    # =========================================================================
    print()
    print("2. High-Precision XPCS Analysis:")
    print("-" * 50)

    kwargs_hp = create_xpcs_high_precision_preset()
    print(f"  High-precision preset: {kwargs_hp}")

    popt_hp, pcov_hp = fit(
        xpcs_g2_model,
        t_data,
        y_data,
        p0=p0,
        bounds=bounds,
        **kwargs_hp,
    )

    perr = np.sqrt(np.diag(pcov_hp))
    print("\n  Fitted with uncertainties:")
    print(f"    tau: {popt_hp[0]:.6f} ± {perr[0]:.6f} s")
    print(f"    beta: {popt_hp[1]:.5f} ± {perr[1]:.5f}")
    print(f"    baseline: {popt_hp[2]:.6f} ± {perr[2]:.6f}")
    print(f"    contrast: {popt_hp[3]:.5f} ± {perr[3]:.5f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("XPCS preset characteristics:")
    print("  - Multi-start enabled: Critical for stretched exponential fits")
    print("  - n_starts=15: More starts for complex dynamics")
    print("  - Sobol sampler: Better coverage of tau/beta space")
    print("  - gtol=1e-8: Sufficient for correlation analysis")
    print()
    print("Use cases:")
    print("  - Colloidal dynamics (diffusion, aging)")
    print("  - Polymer relaxation studies")
    print("  - Liquid crystal dynamics")
    print("  - Two-time correlation analysis")
    print()
    print("Parameter bounds guidance:")
    print("  - tau: Set based on experimental time window")
    print("  - beta: [0.1, 2.0] covers sub- and super-diffusive regimes")
    print("  - contrast: [0, 1] for normalized correlation functions")
    print()
    print("Usage:")
    print("  kwargs = create_xpcs_preset()")
    print("  popt, pcov = fit(xpcs_g2_model, t, g2, **kwargs)")


if __name__ == "__main__":
    main()

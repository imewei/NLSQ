"""SAXS (Small-Angle X-ray Scattering) Domain Preset Example.

This example demonstrates how to create a custom preset for SAXS form factor
fitting using NLSQ's WorkflowConfig.from_preset().with_overrides() pattern.

SAXS analysis typically involves:
- Fitting form factor models (spheres, cylinders, core-shell, etc.)
- Parameters spanning many orders of magnitude (radius in nm, intensity in counts)
- High precision requirements for structural characterization
- Moderate dataset sizes (typically hundreds to thousands of q-points)

Run this example:
    python examples/scripts/domain_presets/saxs_preset.py
"""

import jax.numpy as jnp
import numpy as np

from nlsq import fit
from nlsq.core.workflow import WorkflowConfig


def create_saxs_preset() -> WorkflowConfig:
    """Create a workflow configuration optimized for SAXS form factor fitting.

    SAXS-specific considerations:
    - Tolerances: 1e-8 to 1e-10 for accurate structural parameters
    - Multi-start: Helpful for complex form factors with shape ambiguity
    - Normalization: Critical due to intensity/size parameter scale differences

    Returns
    -------
    WorkflowConfig
        Configuration optimized for SAXS analysis.

    Example
    -------
    >>> config = create_saxs_preset()
    >>> config.gtol
    1e-09
    >>> config.enable_multistart
    True
    """
    # Start from precision_standard and customize for SAXS
    # For SAXS, we often need tighter tolerances than default for
    # accurate size determination, but not as tight as precision_high
    # since form factor oscillations provide good gradient information

    config = WorkflowConfig.from_preset("precision_standard").with_overrides(
        # SAXS-specific overrides:

        # Tighter tolerances for accurate structural parameters
        # Form factor fitting benefits from higher precision
        gtol=1e-9,
        ftol=1e-9,
        xtol=1e-9,

        # Moderate n_starts - form factors are usually well-behaved
        # but can have local minima for polydisperse or multi-component systems
        n_starts=12,

        # LHS sampling works well for SAXS parameter spaces
        sampler="lhs",
    )

    return config


def sphere_form_factor(q, radius, scale, background):
    """Sphere form factor for SAXS fitting.

    The form factor amplitude for a uniform sphere:
        F(q) = 3 * [sin(qR) - qR*cos(qR)] / (qR)^3

    Intensity: I(q) = scale * |F(q)|^2 + background

    Parameters
    ----------
    q : array_like
        Scattering vector magnitude (typically in nm^-1 or A^-1)
    radius : float
        Sphere radius (same units as 1/q)
    scale : float
        Intensity scaling factor (proportional to concentration and contrast)
    background : float
        Flat background intensity

    Returns
    -------
    array
        Scattering intensity at each q value
    """
    qr = q * radius

    # Avoid division by zero at q=0
    # Use Taylor expansion for small qr: F(qr) -> 1 - (qr)^2/10 + ...
    form_factor = jnp.where(
        jnp.abs(qr) < 1e-6,
        1.0 - qr**2 / 10.0,
        3.0 * (jnp.sin(qr) - qr * jnp.cos(qr)) / qr**3
    )

    return scale * form_factor**2 + background


def core_shell_form_factor(q, r_core, r_shell, scale, background):
    """Core-shell sphere form factor for SAXS fitting.

    A simplified model with fixed contrast ratios.

    Parameters
    ----------
    q : array_like
        Scattering vector magnitude
    r_core : float
        Core radius
    r_shell : float
        Total radius (core + shell thickness)
    scale : float
        Intensity scaling factor
    background : float
        Flat background intensity

    Returns
    -------
    array
        Scattering intensity at each q value
    """
    def sphere_amplitude(q_val, r):
        qr = q_val * r
        return jnp.where(
            jnp.abs(qr) < 1e-6,
            1.0 - qr**2 / 10.0,
            3.0 * (jnp.sin(qr) - qr * jnp.cos(qr)) / qr**3
        )

    # Simplified core-shell with fixed contrast ratio (core=1, shell=0.5)
    v_core = (4.0 / 3.0) * jnp.pi * r_core**3
    v_shell = (4.0 / 3.0) * jnp.pi * r_shell**3

    f_core = v_core * sphere_amplitude(q, r_core)
    f_shell = v_shell * sphere_amplitude(q, r_shell)

    # Combined amplitude (assuming shell contrast is 0.5 of core)
    f_total = f_core + 0.5 * (f_shell - f_core)

    return scale * (f_total / v_shell)**2 + background


def main():
    print("=" * 70)
    print("SAXS Domain Preset Example")
    print("=" * 70)
    print()

    # Create the SAXS preset
    config = create_saxs_preset()

    print("SAXS Preset Configuration:")
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

    # Generate synthetic SAXS data for sphere form factor
    print("Generating synthetic SAXS data (spheres)...")
    np.random.seed(42)

    # q-range typical for SAXS (nm^-1)
    q_data = np.logspace(-2, 0, 200)  # 0.01 to 1 nm^-1

    # True parameters
    true_params = {
        "radius": 10.0,         # 10 nm radius
        "scale": 1e6,           # Intensity scale
        "background": 10.0,     # Background counts
    }

    # Generate noisy data with realistic Poisson-like noise
    y_true = sphere_form_factor(
        q_data,
        true_params["radius"],
        true_params["scale"],
        true_params["background"],
    )
    # Add relative noise that increases at low intensity
    noise = 0.05 * np.sqrt(np.maximum(y_true, 1)) * np.random.randn(len(q_data))
    y_data = np.maximum(y_true + noise, 0.1)  # Ensure positive intensities

    print(f"  Data points: {len(q_data)}")
    print(f"  True radius: {true_params['radius']:.2f} nm")
    print(f"  True scale:  {true_params['scale']:.2e}")
    print()

    # Initial guesses and bounds
    # Note: Scale and radius have very different magnitudes
    p0 = [8.0, 5e5, 5.0]  # [radius, scale, background]

    bounds = (
        [1.0, 1e3, 0.0],      # Lower bounds
        [50.0, 1e9, 100.0],   # Upper bounds
    )

    # Fit using the SAXS preset
    print("Fitting sphere form factor with SAXS preset...")
    popt, pcov = fit(
        sphere_form_factor,
        q_data,
        y_data,
        p0=p0,
        bounds=bounds,
        workflow_config=config,
    )

    # Results
    print()
    print("Fit Results:")
    print("-" * 40)
    print(f"  radius:     {popt[0]:.4f} nm (true: {true_params['radius']:.4f})")
    print(f"  scale:      {popt[1]:.4e} (true: {true_params['scale']:.4e})")
    print(f"  background: {popt[2]:.4f} (true: {true_params['background']:.4f})")

    # Parameter uncertainties
    if pcov is not None:
        perr = np.sqrt(np.diag(pcov))
        print()
        print("Parameter Uncertainties (1-sigma):")
        print(f"  radius:     +/- {perr[0]:.4f} nm")
        print(f"  scale:      +/- {perr[1]:.4e}")
        print(f"  background: +/- {perr[2]:.4f}")

    # Calculate relative error
    rel_error = 100 * np.abs(popt[0] - true_params["radius"]) / true_params["radius"]
    print()
    print(f"  Radius relative error: {rel_error:.3f}%")

    print()
    print("=" * 70)
    print("Notes on SAXS Preset Customization")
    print("=" * 70)
    print()
    print("The SAXS preset builds on 'precision_standard' with adjustments for:")
    print()
    print("1. Tighter tolerances (1e-9)")
    print("   - Form factor oscillations provide good gradient information")
    print("   - Accurate size determination requires higher precision")
    print("   - Still computationally efficient for typical SAXS datasets")
    print()
    print("2. Multi-start optimization (n_starts=12)")
    print("   - Helps with polydisperse systems")
    print("   - Useful for complex form factors (core-shell, ellipsoids)")
    print("   - Moderate number sufficient for well-conditioned problems")
    print()
    print("3. Parameter scaling considerations:")
    print("   - Intensity scales span many orders of magnitude")
    print("   - NLSQ's automatic normalization handles this well")
    print("   - Use bounds to constrain physically reasonable values")
    print()
    print("4. Common SAXS models:")
    print("   - Sphere, cylinder, ellipsoid form factors")
    print("   - Core-shell and multi-shell structures")
    print("   - Structure factors for concentrated systems")
    print()


if __name__ == "__main__":
    main()

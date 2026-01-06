"""SAXS (Small-Angle X-ray Scattering) Domain Preset Example.

This example demonstrates how to create a custom preset for SAXS form factor
fitting using NLSQ's fit() function with custom kwargs.

SAXS analysis typically involves:
- Fitting form factor models (spheres, cylinders, core-shell, etc.)
- Parameters spanning many orders of magnitude (radius in nm, intensity in counts)
- High precision requirements for structural characterization
- Moderate dataset sizes (typically hundreds to thousands of q-points)

Run this example:
    python examples/scripts/08_workflow_system/10_saxs_presets.py
"""

import jax.numpy as jnp
import numpy as np

from nlsq import fit


def create_saxs_preset() -> dict:
    """Create fit kwargs optimized for SAXS form factor fitting.

    SAXS-specific considerations:
    - Tolerances: 1e-8 to 1e-10 for accurate structural parameters
    - Multi-start: Helpful for complex form factors with shape ambiguity
    - Normalization: Critical due to intensity/size parameter scale differences

    Returns
    -------
    dict
        Keyword arguments for fit() optimized for SAXS analysis.

    Example
    -------
    >>> kwargs = create_saxs_preset()
    >>> popt, pcov = fit(model, q, I, **kwargs)
    """
    # SAXS typically needs tighter tolerances than default for
    # accurate size determination, but form factor oscillations
    # provide good gradient information
    return {
        "workflow": "standard",
        # Multi-start is helpful for:
        # 1. Complex form factors (core-shell, polydisperse)
        # 2. Shape ambiguity (sphere vs oblate ellipsoid)
        "multistart": True,
        "n_starts": 10,  # Sufficient for most form factors
        "sampler": "lhs",
        # Tighter tolerances for structural accuracy
        "gtol": 1e-9,
        "ftol": 1e-9,
        "xtol": 1e-9,
    }


def create_saxs_high_precision_preset() -> dict:
    """Create fit kwargs for high-precision SAXS analysis.

    Use this for:
    - Publication-quality structural parameters
    - Polydispersity analysis
    - Complex multi-component systems

    Returns
    -------
    dict
        Keyword arguments for fit() with high precision settings.
    """
    return {
        "workflow": "quality",
        "multistart": True,
        "n_starts": 20,
        "sampler": "sobol",  # Better coverage for high-dimensional problems
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
    }


def sphere_form_factor(q, intensity_scale, radius, background):
    """Spherical form factor for SAXS.

    I(q) = I0 * [3(sin(qR) - qR*cos(qR)) / (qR)^3]^2 + background

    Parameters
    ----------
    q : array_like
        Scattering vector (Å^-1 or nm^-1)
    intensity_scale : float
        Overall intensity scaling factor
    radius : float
        Sphere radius (same units as 1/q)
    background : float
        Flat background level

    Returns
    -------
    array
        Scattered intensity I(q)
    """
    qr = q * radius

    # Avoid division by zero at q=0
    # Use Taylor expansion near qr=0
    def safe_form_factor(qr):
        # Form factor: 3*(sin(qr) - qr*cos(qr)) / qr^3
        small = qr < 1e-6
        large_qr = jnp.where(small, 1.0, qr)

        # For large qr, use exact formula
        ff_large = (
            3.0 * (jnp.sin(large_qr) - large_qr * jnp.cos(large_qr)) / large_qr**3
        )

        # For small qr, use Taylor expansion: 1 - qr^2/10 + O(qr^4)
        ff_small = 1.0 - qr**2 / 10.0

        return jnp.where(small, ff_small, ff_large)

    form_factor = safe_form_factor(qr)
    return intensity_scale * form_factor**2 + background


def core_shell_sphere(
    q, scale, r_core, thickness, sld_core, sld_shell, sld_solvent, bg
):
    """Core-shell sphere form factor for SAXS.

    Models particles with a core of one density surrounded by a shell
    of different density.

    Parameters
    ----------
    q : array_like
        Scattering vector
    scale : float
        Overall scale factor
    r_core : float
        Core radius
    thickness : float
        Shell thickness (total radius = r_core + thickness)
    sld_core : float
        Core scattering length density
    sld_shell : float
        Shell scattering length density
    sld_solvent : float
        Solvent scattering length density
    bg : float
        Background

    Returns
    -------
    array
        Scattered intensity
    """
    r_total = r_core + thickness

    def sphere_amplitude(q, r):
        qr = q * r
        small = qr < 1e-6
        large_qr = jnp.where(small, 1.0, qr)
        amp_large = (
            3.0 * (jnp.sin(large_qr) - large_qr * jnp.cos(large_qr)) / large_qr**3
        )
        amp_small = 1.0 - qr**2 / 10.0
        return jnp.where(small, amp_small, amp_large)

    # Volume weighted amplitudes
    v_core = (4.0 / 3.0) * jnp.pi * r_core**3
    v_total = (4.0 / 3.0) * jnp.pi * r_total**3

    f_core = v_core * (sld_core - sld_shell) * sphere_amplitude(q, r_core)
    f_shell = v_total * (sld_shell - sld_solvent) * sphere_amplitude(q, r_total)

    form_factor = f_core + f_shell
    return scale * form_factor**2 + bg


def main():
    print("=" * 70)
    print("SAXS (Small-Angle X-ray Scattering) Domain Preset")
    print("=" * 70)
    print()

    np.random.seed(42)

    # =========================================================================
    # 1. Simple Sphere Form Factor
    # =========================================================================
    print("1. Simple Sphere Form Factor:")
    print("-" * 50)

    true_scale = 100.0
    true_radius = 50.0  # nm
    true_bg = 0.1

    # q-range typical for SAXS (nm^-1)
    q = np.linspace(0.01, 0.5, 200)

    # Generate synthetic SAXS data
    I_true = (
        true_scale
        * (
            (
                3
                * (np.sin(q * true_radius) - q * true_radius * np.cos(q * true_radius))
                / (q * true_radius) ** 3
            )
            ** 2
        )
        + true_bg
    )
    I_data = I_true * (1 + 0.02 * np.random.randn(len(q)))

    print("  True parameters:")
    print(f"    Intensity scale: {true_scale}")
    print(f"    Radius: {true_radius} nm")
    print(f"    Background: {true_bg}")

    # Fit using SAXS preset
    kwargs = create_saxs_preset()
    print(f"\n  SAXS preset: {kwargs}")

    popt, pcov = fit(
        sphere_form_factor,
        q,
        I_data,
        p0=[50.0, 30.0, 0.5],
        bounds=([0.1, 1.0, 0.0], [1000.0, 200.0, 10.0]),
        **kwargs,
    )

    print("\n  Fitted parameters:")
    print(f"    Intensity scale: {popt[0]:.2f} (true: {true_scale})")
    print(f"    Radius: {popt[1]:.2f} nm (true: {true_radius})")
    print(f"    Background: {popt[2]:.4f} (true: {true_bg})")

    # =========================================================================
    # 2. High-Precision SAXS Analysis
    # =========================================================================
    print()
    print("2. High-Precision SAXS Analysis:")
    print("-" * 50)

    kwargs_hp = create_saxs_high_precision_preset()
    print(f"  High-precision preset: {kwargs_hp}")

    popt_hp, pcov_hp = fit(
        sphere_form_factor,
        q,
        I_data,
        p0=[50.0, 30.0, 0.5],
        bounds=([0.1, 1.0, 0.0], [1000.0, 200.0, 10.0]),
        **kwargs_hp,
    )

    perr = np.sqrt(np.diag(pcov_hp))
    print("\n  Fitted with uncertainties:")
    print(f"    Intensity: {popt_hp[0]:.3f} ± {perr[0]:.3f}")
    print(f"    Radius: {popt_hp[1]:.3f} ± {perr[1]:.3f} nm")
    print(f"    Background: {popt_hp[2]:.5f} ± {perr[2]:.5f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("SAXS preset characteristics:")
    print("  - Multi-start enabled: Helps with shape ambiguity")
    print("  - n_starts=10: Sufficient for typical form factors")
    print("  - gtol=1e-9: Tighter tolerance for structural accuracy")
    print("  - LHS sampler: Good coverage of parameter space")
    print()
    print("Use cases:")
    print("  - Nanoparticle size determination")
    print("  - Protein shape analysis")
    print("  - Core-shell structures")
    print("  - Polydispersity analysis")
    print()
    print("Usage:")
    print("  kwargs = create_saxs_preset()")
    print("  popt, pcov = fit(form_factor, q, I, **kwargs)")


if __name__ == "__main__":
    main()

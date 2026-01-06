"""Chemical/Enzyme Kinetics Domain Preset Example.

This example demonstrates how to create a custom preset for chemical and
enzyme kinetics fitting using NLSQ's fit() function with custom kwargs.

Kinetics analysis typically involves:
- Fitting rate equations (exponential decays, Michaelis-Menten, etc.)
- Rate constants spanning many orders of magnitude
- High precision for accurate rate determination
- Often limited data points requiring robust convergence

Run this example:
    python examples/scripts/08_workflow_system/09_kinetics_presets.py
"""

import jax.numpy as jnp
import numpy as np

from nlsq import fit


def create_kinetics_preset() -> dict:
    """Create fit kwargs optimized for kinetics rate constant fitting.

    Kinetics-specific considerations:
    - Rate constants can span many orders of magnitude (10^-6 to 10^6 s^-1)
    - Exponential models are sensitive to initial guesses
    - Multi-start is essential for avoiding local minima
    - Moderate precision usually sufficient for rate constants

    Returns
    -------
    dict
        Keyword arguments for fit() optimized for kinetics analysis.

    Example
    -------
    >>> kwargs = create_kinetics_preset()
    >>> popt, pcov = fit(model, x, y, **kwargs)
    """
    # Kinetics fitting often has multiple local minima, especially
    # for multi-exponential and complex reaction schemes
    return {
        "workflow": "standard",
        # Multi-start is critical for kinetics due to:
        # 1. Exponential sensitivity to rate constants
        # 2. Multiple timescales in complex reactions
        "multistart": True,
        "n_starts": 20,  # More starts for exponential models
        "sampler": "lhs",  # Good for spanning rate constant ranges
        # Moderate tolerances sufficient for rate constants
        # (rate constants typically reported to 2-3 significant figures)
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
    }


def create_kinetics_high_precision_preset() -> dict:
    """Create fit kwargs for high-precision kinetics analysis.

    Use this for:
    - Activation energy determination (Arrhenius plots)
    - Isotope effect studies requiring precise kH/kD ratios
    - Competitive kinetics with similar rate constants

    Returns
    -------
    dict
        Keyword arguments for fit() with high precision settings.
    """
    return {
        "workflow": "quality",
        "multistart": True,
        "n_starts": 30,  # More starts for high precision
        "sampler": "sobol",  # Better space coverage
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
    }


def michaelis_menten(substrate, vmax, km):
    """Michaelis-Menten enzyme kinetics model.

    v = Vmax * [S] / (Km + [S])

    Parameters
    ----------
    substrate : array_like
        Substrate concentration [S]
    vmax : float
        Maximum reaction velocity Vmax
    km : float
        Michaelis constant Km

    Returns
    -------
    array
        Reaction velocity v
    """
    return vmax * substrate / (km + substrate)


def first_order_decay(t, a0, k):
    """First-order decay kinetics.

    [A] = [A]0 * exp(-k*t)

    Parameters
    ----------
    t : array_like
        Time points
    a0 : float
        Initial concentration [A]0
    k : float
        Rate constant k

    Returns
    -------
    array
        Concentration at time t
    """
    return a0 * jnp.exp(-k * t)


def biexponential_decay(t, a1, k1, a2, k2, offset):
    """Biexponential decay kinetics.

    [A] = A1 * exp(-k1*t) + A2 * exp(-k2*t) + offset

    Useful for:
    - Protein folding with intermediate states
    - Parallel reaction pathways
    - Relaxation with fast and slow components
    """
    return a1 * jnp.exp(-k1 * t) + a2 * jnp.exp(-k2 * t) + offset


def main():
    print("=" * 70)
    print("Chemical/Enzyme Kinetics Domain Preset")
    print("=" * 70)
    print()

    np.random.seed(42)

    # =========================================================================
    # 1. Michaelis-Menten Enzyme Kinetics
    # =========================================================================
    print("1. Michaelis-Menten Enzyme Kinetics:")
    print("-" * 50)

    # Generate synthetic enzyme kinetics data
    true_vmax = 100.0  # max velocity (units/min)
    true_km = 5.0  # Michaelis constant (mM)

    substrate_conc = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0])
    velocity_true = true_vmax * substrate_conc / (true_km + substrate_conc)
    velocity_data = velocity_true * (1 + 0.05 * np.random.randn(len(substrate_conc)))

    print(f"  True Vmax: {true_vmax} units/min")
    print(f"  True Km: {true_km} mM")

    # Fit using kinetics preset
    kwargs = create_kinetics_preset()
    print(f"\n  Kinetics preset: {kwargs}")

    popt, pcov = fit(
        michaelis_menten,
        substrate_conc,
        velocity_data,
        p0=[50.0, 10.0],  # Initial guess
        bounds=([0.1, 0.01], [500.0, 100.0]),
        **kwargs,
    )

    print(f"\n  Fitted Vmax: {popt[0]:.2f} units/min (true: {true_vmax})")
    print(f"  Fitted Km: {popt[1]:.2f} mM (true: {true_km})")

    # =========================================================================
    # 2. First-Order Decay Kinetics
    # =========================================================================
    print()
    print("2. First-Order Decay Kinetics:")
    print("-" * 50)

    true_a0 = 1.0  # Initial concentration
    true_k = 0.05  # Rate constant (s^-1)

    time = np.linspace(0, 100, 50)
    conc_true = true_a0 * np.exp(-true_k * time)
    conc_data = conc_true + 0.02 * np.random.randn(len(time))

    print(f"  True [A]0: {true_a0}")
    print(f"  True k: {true_k} s^-1")

    popt, pcov = fit(
        first_order_decay,
        time,
        conc_data,
        p0=[0.5, 0.1],
        bounds=([0.01, 0.001], [10.0, 1.0]),
        **kwargs,
    )

    perr = np.sqrt(np.diag(pcov))
    print(f"\n  Fitted [A]0: {popt[0]:.4f} ± {perr[0]:.4f}")
    print(f"  Fitted k: {popt[1]:.5f} ± {perr[1]:.5f} s^-1")

    # =========================================================================
    # 3. Biexponential Decay (Complex Kinetics)
    # =========================================================================
    print()
    print("3. Biexponential Decay (Complex Kinetics):")
    print("-" * 50)

    # Parameters: fast phase + slow phase
    true_a1, true_k1 = 0.6, 0.5  # Fast component
    true_a2, true_k2 = 0.4, 0.02  # Slow component
    true_offset = 0.0

    time = np.linspace(0, 200, 100)
    signal_true = (
        true_a1 * np.exp(-true_k1 * time)
        + true_a2 * np.exp(-true_k2 * time)
        + true_offset
    )
    signal_data = signal_true + 0.02 * np.random.randn(len(time))

    print("  True parameters:")
    print(f"    Fast: A1={true_a1}, k1={true_k1} s^-1")
    print(f"    Slow: A2={true_a2}, k2={true_k2} s^-1")

    # Biexponential fits are notoriously sensitive to initial guesses
    # Multi-start is critical here
    popt, pcov = fit(
        biexponential_decay,
        time,
        signal_data,
        p0=[0.5, 0.3, 0.3, 0.01, 0.0],  # Initial guess
        bounds=([0.0, 0.001, 0.0, 0.0001, -0.1], [2.0, 10.0, 2.0, 10.0, 0.1]),
        **kwargs,
    )

    print("\n  Fitted parameters:")
    print(f"    Fast: A1={popt[0]:.3f}, k1={popt[1]:.4f} s^-1")
    print(f"    Slow: A2={popt[2]:.3f}, k2={popt[3]:.5f} s^-1")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Kinetics preset characteristics:")
    print("  - Multi-start enabled: Essential for exponential models")
    print("  - n_starts=20: More starts for rate constant fitting")
    print("  - LHS sampler: Good coverage of rate constant ranges")
    print("  - Standard tolerances: Sufficient for typical rate constants")
    print()
    print("Use cases:")
    print("  - Enzyme kinetics (Michaelis-Menten)")
    print("  - First-order reactions")
    print("  - Multi-exponential decays")
    print("  - Activation energy determination")
    print()
    print("Usage:")
    print("  kwargs = create_kinetics_preset()")
    print("  popt, pcov = fit(model, x, y, **kwargs)")


if __name__ == "__main__":
    main()

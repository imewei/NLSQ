"""Chemical/Enzyme Kinetics Domain Preset Example.

This example demonstrates how to create a custom preset for chemical and
enzyme kinetics fitting using NLSQ's WorkflowConfig.from_preset().with_overrides()
pattern.

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
from nlsq.core.workflow import WorkflowConfig


def create_kinetics_preset() -> WorkflowConfig:
    """Create a workflow configuration optimized for kinetics rate constant fitting.

    Kinetics-specific considerations:
    - Rate constants can span many orders of magnitude (10^-6 to 10^6 s^-1)
    - Exponential models are sensitive to initial guesses
    - Multi-start is essential for avoiding local minima
    - Moderate precision usually sufficient for rate constants

    Returns
    -------
    WorkflowConfig
        Configuration optimized for kinetics analysis.

    Example
    -------
    >>> config = create_kinetics_preset()
    >>> config.enable_multistart
    True
    >>> config.n_starts
    20
    """
    # Start from precision_standard and customize for kinetics
    # Kinetics fitting often has multiple local minima, especially
    # for multi-exponential and complex reaction schemes

    config = WorkflowConfig.from_preset("precision_standard").with_overrides(
        # Kinetics-specific overrides:
        # More aggressive multi-start for rate constant fitting
        # Multi-exponential models are notorious for local minima
        n_starts=20,
        # Sobol sampling provides better coverage for rate constant
        # parameter spaces which often span many orders of magnitude
        sampler="sobol",
        # Standard tolerances are usually sufficient for rate constants
        # since experimental uncertainty dominates
        gtol=1e-8,
        ftol=1e-8,
        xtol=1e-8,
    )

    return config


def first_order_decay(t, k, A0, offset):
    """First-order decay kinetics: A(t) = A0 * exp(-k*t) + offset.

    Parameters
    ----------
    t : array_like
        Time points
    k : float
        Rate constant (s^-1 or appropriate time units)
    A0 : float
        Initial amplitude
    offset : float
        Baseline offset (product concentration at equilibrium)

    Returns
    -------
    array
        Concentration or signal at each time point
    """
    return A0 * jnp.exp(-k * t) + offset


def biexponential_decay(t, k1, k2, A1, A2, offset):
    """Bi-exponential decay: sum of two first-order processes.

    Parameters
    ----------
    t : array_like
        Time points
    k1 : float
        Fast rate constant
    k2 : float
        Slow rate constant
    A1 : float
        Amplitude of fast component
    A2 : float
        Amplitude of slow component
    offset : float
        Baseline offset

    Returns
    -------
    array
        Concentration or signal at each time point
    """
    return A1 * jnp.exp(-k1 * t) + A2 * jnp.exp(-k2 * t) + offset


def michaelis_menten(s, vmax, km):
    """Michaelis-Menten enzyme kinetics: v = Vmax * [S] / (Km + [S]).

    Parameters
    ----------
    s : array_like
        Substrate concentration
    vmax : float
        Maximum reaction velocity
    km : float
        Michaelis constant (substrate concentration at half Vmax)

    Returns
    -------
    array
        Reaction velocity at each substrate concentration
    """
    return vmax * s / (km + s)


def main():
    print("=" * 70)
    print("Chemical/Enzyme Kinetics Domain Preset Example")
    print("=" * 70)
    print()

    # Create the kinetics preset
    config = create_kinetics_preset()

    print("Kinetics Preset Configuration:")
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

    # =========================================================================
    # Example 1: First-order decay
    # =========================================================================
    print("=" * 70)
    print("Example 1: First-Order Decay Kinetics")
    print("=" * 70)
    print()

    np.random.seed(42)

    # Time points
    t_data = np.linspace(0, 10, 50)  # 0-10 seconds

    # True parameters
    true_k = 0.5  # 0.5 s^-1 (half-life = 1.4 s)
    true_A0 = 100.0  # Initial concentration
    true_offset = 10.0  # Baseline

    # Generate noisy data
    y_true = first_order_decay(t_data, true_k, true_A0, true_offset)
    noise = 3.0 * np.random.randn(len(t_data))
    y_data = y_true + noise

    print(f"  True rate constant k = {true_k} s^-1")
    print(f"  True half-life = {np.log(2) / true_k:.3f} s")
    print()

    # Initial guesses (deliberately off)
    p0 = [0.1, 80.0, 5.0]  # [k, A0, offset]
    bounds = ([0.001, 0.1, 0.0], [10.0, 200.0, 50.0])

    # Fit using kinetics preset
    print("Fitting first-order decay...")
    popt, pcov = fit(
        first_order_decay,
        t_data,
        y_data,
        p0=p0,
        bounds=bounds,
        workflow_config=config,
    )

    print()
    print("First-Order Fit Results:")
    print("-" * 40)
    print(f"  k:      {popt[0]:.4f} s^-1 (true: {true_k})")
    print(f"  A0:     {popt[1]:.2f} (true: {true_A0})")
    print(f"  offset: {popt[2]:.2f} (true: {true_offset})")
    print(f"  Fitted half-life: {np.log(2) / popt[0]:.3f} s")

    if pcov is not None:
        perr = np.sqrt(np.diag(pcov))
        print()
        print("  Uncertainties:")
        print(f"    k:      +/- {perr[0]:.4f} s^-1")
        print(f"    A0:     +/- {perr[1]:.2f}")
        print(f"    offset: +/- {perr[2]:.2f}")

    # =========================================================================
    # Example 2: Bi-exponential decay (challenging case)
    # =========================================================================
    print()
    print("=" * 70)
    print("Example 2: Bi-Exponential Decay (Challenging Case)")
    print("=" * 70)
    print()

    np.random.seed(123)

    # Time points (more dense to resolve two phases)
    t_data2 = np.linspace(0, 20, 100)

    # True parameters (well-separated rates)
    true_k1 = 1.0  # Fast rate (1.0 s^-1)
    true_k2 = 0.1  # Slow rate (0.1 s^-1)
    true_A1 = 60.0  # Fast amplitude
    true_A2 = 40.0  # Slow amplitude
    true_offset2 = 5.0

    # Generate noisy data
    y_true2 = biexponential_decay(
        t_data2, true_k1, true_k2, true_A1, true_A2, true_offset2
    )
    noise2 = 2.0 * np.random.randn(len(t_data2))
    y_data2 = y_true2 + noise2

    print(f"  True fast rate k1 = {true_k1} s^-1")
    print(f"  True slow rate k2 = {true_k2} s^-1")
    print(f"  Rate ratio k1/k2 = {true_k1 / true_k2:.1f}")
    print()

    # Initial guesses (this is a challenging fit)
    p0_bi = [0.5, 0.05, 50.0, 50.0, 0.0]  # [k1, k2, A1, A2, offset]
    bounds_bi = (
        [0.01, 0.001, 1.0, 1.0, 0.0],  # Lower bounds
        [10.0, 1.0, 200.0, 200.0, 20.0],  # Upper bounds
    )

    print("Fitting bi-exponential decay...")
    print("(This is challenging - multi-start helps avoid local minima)")
    popt2, pcov2 = fit(
        biexponential_decay,
        t_data2,
        y_data2,
        p0=p0_bi,
        bounds=bounds_bi,
        workflow_config=config,
    )

    print()
    print("Bi-Exponential Fit Results:")
    print("-" * 40)
    print(f"  k1 (fast):  {popt2[0]:.4f} s^-1 (true: {true_k1})")
    print(f"  k2 (slow):  {popt2[1]:.4f} s^-1 (true: {true_k2})")
    print(f"  A1:         {popt2[2]:.2f} (true: {true_A1})")
    print(f"  A2:         {popt2[3]:.2f} (true: {true_A2})")
    print(f"  offset:     {popt2[4]:.2f} (true: {true_offset2})")

    # =========================================================================
    # Example 3: Michaelis-Menten enzyme kinetics
    # =========================================================================
    print()
    print("=" * 70)
    print("Example 3: Michaelis-Menten Enzyme Kinetics")
    print("=" * 70)
    print()

    np.random.seed(456)

    # Substrate concentrations (log-spaced for better coverage)
    s_data = np.logspace(-2, 2, 30)  # 0.01 to 100 mM

    # True parameters
    true_vmax = 100.0  # Maximum velocity (arbitrary units)
    true_km = 5.0  # Michaelis constant (mM)

    # Generate noisy data
    v_true = michaelis_menten(s_data, true_vmax, true_km)
    v_noise = 3.0 * np.random.randn(len(s_data))
    v_data = np.maximum(v_true + v_noise, 0.1)

    print(f"  True Vmax = {true_vmax}")
    print(f"  True Km = {true_km} mM")
    print()

    # Initial guesses
    p0_mm = [50.0, 1.0]  # [Vmax, Km]
    bounds_mm = ([1.0, 0.01], [500.0, 100.0])

    print("Fitting Michaelis-Menten kinetics...")
    popt3, pcov3 = fit(
        michaelis_menten,
        s_data,
        v_data,
        p0=p0_mm,
        bounds=bounds_mm,
        workflow_config=config,
    )

    print()
    print("Michaelis-Menten Fit Results:")
    print("-" * 40)
    print(f"  Vmax: {popt3[0]:.2f} (true: {true_vmax})")
    print(f"  Km:   {popt3[1]:.2f} mM (true: {true_km})")

    if pcov3 is not None:
        perr3 = np.sqrt(np.diag(pcov3))
        print()
        print("  Uncertainties:")
        print(f"    Vmax: +/- {perr3[0]:.2f}")
        print(f"    Km:   +/- {perr3[1]:.2f} mM")

    # =========================================================================
    # Best practices summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Best Practices for Kinetics Fitting")
    print("=" * 70)
    print()
    print("1. Multi-start optimization (n_starts=20)")
    print("   - Essential for multi-exponential models")
    print("   - Helps escape local minima")
    print("   - Sobol sampling for rate constant spaces")
    print()
    print("2. Parameter bounds:")
    print("   - Rate constants: set physically reasonable limits")
    print("   - Order k1 > k2 if rates should be distinguishable")
    print("   - Use log-scale bounds for rate constants")
    print()
    print("3. Data considerations:")
    print("   - Sample time points to capture all phases")
    print("   - For bi-exponential: need ~10x rate separation")
    print("   - More data points improve parameter correlation")
    print()
    print("4. Common pitfalls:")
    print("   - Swapped rate constants (k1 <-> k2)")
    print("   - Correlated amplitude/rate parameters")
    print("   - Insufficient time range for slow phases")
    print()


if __name__ == "__main__":
    main()

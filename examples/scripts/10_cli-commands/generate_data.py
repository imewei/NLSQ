#!/usr/bin/env python
"""Generate sample data files for CLI demonstrations.

This script creates CSV data files that can be used with the NLSQ CLI
to demonstrate curve fitting workflows.

Generated datasets:
1. Radioactive decay (exponential decay)
2. Enzyme kinetics (Michaelis-Menten)
3. Reaction kinetics (first-order decay)

Usage:
    python generate_data.py

Output:
    data/radioactive_decay.csv
    data/enzyme_kinetics.csv
    data/reaction_kinetics.csv
"""

from pathlib import Path

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_radioactive_decay():
    """Generate radioactive decay data (Carbon-14 example).

    Model: N(t) = N0 * exp(-lambda * t)

    True parameters:
        N0 = 1000 counts/min
        lambda = 1.21e-4 yr^-1 (half-life = 5730 years)
    """
    # True parameters
    N0_true = 1000.0
    half_life = 5730.0  # years (C-14)
    lambda_true = np.log(2) / half_life

    # Time points (0 to 20000 years)
    time = np.linspace(0, 20000, 50)

    # True decay curve
    N_true = N0_true * np.exp(-lambda_true * time)

    # Add realistic noise (~5% relative + counting statistics)
    noise = np.sqrt(N_true) + 0.03 * N_true
    N_measured = N_true + np.random.normal(0, 1, size=len(time)) * noise

    # Uncertainties
    sigma = noise

    # Save to CSV (no comment lines for CLI compatibility)
    output_file = DATA_DIR / "radioactive_decay.csv"
    with open(output_file, "w") as f:
        f.write("time,activity,sigma\n")
        for t, n, s in zip(time, N_measured, sigma):
            f.write(f"{t:.2f},{n:.4f},{s:.4f}\n")

    print(f"Generated: {output_file}")
    print(f"  Points: {len(time)}")
    print(f"  True N0: {N0_true}, True lambda: {lambda_true:.6e}")


def generate_enzyme_kinetics():
    """Generate enzyme kinetics data (Michaelis-Menten model).

    Model: v = Vmax * [S] / (Km + [S])

    True parameters:
        Vmax = 100 uM/min
        Km = 50 uM
    """
    # True parameters
    Vmax_true = 100.0  # uM/min
    Km_true = 50.0  # uM

    # Substrate concentrations (log-spaced for better coverage)
    S = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])

    # True velocity
    v_true = Vmax_true * S / (Km_true + S)

    # Add noise (~5% relative error)
    noise = 0.05 * v_true + 0.5
    v_measured = v_true + np.random.normal(0, 1, size=len(S)) * noise

    # Ensure positive velocities
    v_measured = np.maximum(v_measured, 0.1)

    # Uncertainties
    sigma = noise

    # Save to CSV (no comment lines for CLI compatibility)
    output_file = DATA_DIR / "enzyme_kinetics.csv"
    with open(output_file, "w") as f:
        f.write("substrate,velocity,sigma\n")
        for s, v, sig in zip(S, v_measured, sigma):
            f.write(f"{s:.2f},{v:.4f},{sig:.4f}\n")

    print(f"Generated: {output_file}")
    print(f"  Points: {len(S)}")
    print(f"  True Vmax: {Vmax_true}, True Km: {Km_true}")


def generate_reaction_kinetics():
    """Generate first-order reaction kinetics data.

    Model: C(t) = C0 * exp(-k * t)

    True parameters:
        C0 = 1.0 M
        k = 0.005 s^-1
    """
    # True parameters
    C0_true = 1.0  # M
    k_true = 0.005  # s^-1

    # Time points (0 to 1000 seconds)
    time = np.linspace(0, 1000, 100)

    # True concentration
    C_true = C0_true * np.exp(-k_true * time)

    # Add noise (~2% relative error)
    noise = 0.02 * C_true + 0.001
    C_measured = C_true + np.random.normal(0, 1, size=len(time)) * noise

    # Ensure positive concentrations
    C_measured = np.maximum(C_measured, 0.001)

    # Uncertainties
    sigma = noise

    # Save to CSV (no comment lines for CLI compatibility)
    output_file = DATA_DIR / "reaction_kinetics.csv"
    with open(output_file, "w") as f:
        f.write("time,concentration,sigma\n")
        for t, c, s in zip(time, C_measured, sigma):
            f.write(f"{t:.2f},{c:.6f},{s:.6f}\n")

    print(f"Generated: {output_file}")
    print(f"  Points: {len(time)}")
    print(f"  True C0: {C0_true}, True k: {k_true}")


def generate_damped_oscillation():
    """Generate damped oscillation data for custom model demo.

    Model: y = A * exp(-gamma * t) * cos(omega * t + phi)

    True parameters:
        A = 5.0
        gamma = 0.1 (decay rate)
        omega = 2.0 (angular frequency)
        phi = 0.5 (phase)
    """
    # True parameters
    A_true = 5.0
    gamma_true = 0.1
    omega_true = 2.0
    phi_true = 0.5

    # Time points
    time = np.linspace(0, 30, 150)

    # True signal
    y_true = A_true * np.exp(-gamma_true * time) * np.cos(omega_true * time + phi_true)

    # Add noise
    noise_level = 0.1
    y_measured = y_true + np.random.normal(0, noise_level, size=len(time))

    # Uncertainties
    sigma = np.full_like(time, noise_level)

    # Save to CSV (no comment lines for CLI compatibility)
    output_file = DATA_DIR / "damped_oscillation.csv"
    with open(output_file, "w") as f:
        f.write("time,amplitude,sigma\n")
        for t, y, s in zip(time, y_measured, sigma):
            f.write(f"{t:.4f},{y:.6f},{s:.6f}\n")

    print(f"Generated: {output_file}")
    print(f"  Points: {len(time)}")
    print(f"  True A: {A_true}, gamma: {gamma_true}, omega: {omega_true}, phi: {phi_true}")


def main():
    """Generate all sample data files."""
    print("=" * 60)
    print("NLSQ CLI Example Data Generation")
    print("=" * 60)
    print()

    generate_radioactive_decay()
    print()

    generate_enzyme_kinetics()
    print()

    generate_reaction_kinetics()
    print()

    generate_damped_oscillation()
    print()

    print("=" * 60)
    print("All data files generated successfully!")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

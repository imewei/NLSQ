#!/usr/bin/env python3
"""Generate sample data files for NLSQ CLI demonstrations.

This script creates various sample datasets in different formats to demonstrate
NLSQ CLI capabilities. Run this script to regenerate all sample data files.

Usage:
    python generate_sample_data.py
"""

from pathlib import Path

import h5py
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent


def damped_oscillator(t, A0, gamma, omega, phi):
    """Damped harmonic oscillator model."""
    return A0 * np.exp(-gamma * t) * np.cos(omega * t + phi)


def exponential_decay(x, a, b, c):
    """Exponential decay model: y = a * exp(-b*x) + c."""
    return a * np.exp(-b * x) + c


def gaussian(x, amplitude, mu, sigma):
    """Gaussian model."""
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def michaelis_menten(S, Vmax, Km):
    """Michaelis-Menten enzyme kinetics model."""
    return Vmax * S / (Km + S)


def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian surface model."""
    return (
        amplitude
        * np.exp(-((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2)))
        + offset
    )


def polynomial(x, *coeffs):
    """Polynomial model."""
    result = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        result += c * x**i
    return result


def main():
    """Generate all sample data files."""
    print("=" * 60)
    print("Generating NLSQ CLI Sample Data Files")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Damped Oscillation (CSV format)
    # -------------------------------------------------------------------------
    print("\n1. Generating damped_oscillation.csv...")
    t = np.linspace(0, 60, 300)
    # True parameters
    A0_true, gamma_true, omega_true, phi_true = 15.0, 0.05, np.pi, 0.0
    y_true = damped_oscillator(t, A0_true, gamma_true, omega_true, phi_true)
    noise = np.random.normal(0, 0.2, size=len(t))
    y_measured = y_true + noise
    sigma = 0.2 * np.ones_like(t)

    with open(OUTPUT_DIR / "damped_oscillation.csv", "w") as f:
        f.write("# Damped oscillation data for NLSQ CLI demonstration\n")
        f.write("# True parameters: A0=15.0, gamma=0.05, omega=pi, phi=0.0\n")
        f.write("time,displacement,sigma\n")
        f.writelines(
            f"{ti:.6f},{yi:.6f},{si:.6f}\n"
            for ti, yi, si in zip(t, y_measured, sigma, strict=False)
        )
    print(f"   Created: damped_oscillation.csv ({len(t)} points)")

    # -------------------------------------------------------------------------
    # 2. Exponential Decay (CSV format, no header)
    # -------------------------------------------------------------------------
    print("\n2. Generating exponential_decay.csv...")
    x = np.linspace(0, 10, 100)
    # True parameters: a=5.0, b=0.5, c=1.0
    y_true = exponential_decay(x, 5.0, 0.5, 1.0)
    noise = np.random.normal(0, 0.1, size=len(x))
    y_measured = y_true + noise
    sigma = 0.1 * np.ones_like(x)

    with open(OUTPUT_DIR / "exponential_decay.csv", "w") as f:
        f.write("x,y,sigma\n")
        f.writelines(
            f"{xi:.6f},{yi:.6f},{si:.6f}\n"
            for xi, yi, si in zip(x, y_measured, sigma, strict=False)
        )
    print(f"   Created: exponential_decay.csv ({len(x)} points)")

    # -------------------------------------------------------------------------
    # 3. Gaussian Peak (ASCII text format)
    # -------------------------------------------------------------------------
    print("\n3. Generating gaussian_peak.txt...")
    x = np.linspace(-5, 5, 150)
    # True parameters: amplitude=10.0, mu=0.5, sigma=1.2
    y_true = gaussian(x, 10.0, 0.5, 1.2)
    noise = np.random.normal(0, 0.2, size=len(x))
    y_measured = y_true + noise

    with open(OUTPUT_DIR / "gaussian_peak.txt", "w") as f:
        f.write("# Gaussian peak data (ASCII format)\n")
        f.write("# Columns: x  y\n")
        f.write("# True parameters: amplitude=10.0, mu=0.5, sigma=1.2\n")
        f.writelines(
            f"{xi:12.6f}  {yi:12.6f}\n" for xi, yi in zip(x, y_measured, strict=False)
        )
    print(f"   Created: gaussian_peak.txt ({len(x)} points)")

    # -------------------------------------------------------------------------
    # 4. Enzyme Kinetics (CSV format)
    # -------------------------------------------------------------------------
    print("\n4. Generating enzyme_kinetics.csv...")
    S = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    # True parameters: Vmax=100.0, Km=50.0
    v_true = michaelis_menten(S, 100.0, 50.0)
    noise = np.random.normal(0, 0.05 * v_true, size=len(S))
    v_measured = v_true + noise
    sigma_v = 0.05 * v_measured + 1.0

    with open(OUTPUT_DIR / "enzyme_kinetics.csv", "w") as f:
        f.write("substrate_conc,velocity,sigma\n")
        f.writelines(
            f"{si:.6f},{vi:.6f},{sigi:.6f}\n"
            for si, vi, sigi in zip(S, v_measured, sigma_v, strict=False)
        )
    print(f"   Created: enzyme_kinetics.csv ({len(S)} points)")

    # -------------------------------------------------------------------------
    # 5. Polynomial Data (NPZ format)
    # -------------------------------------------------------------------------
    print("\n5. Generating polynomial_data.npz...")
    x = np.linspace(-2, 2, 80)
    # True coefficients: a0=1.0, a1=0.5, a2=-0.3, a3=0.1
    y_true = polynomial(x, 1.0, 0.5, -0.3, 0.1)
    noise = np.random.normal(0, 0.05, size=len(x))
    y_measured = y_true + noise
    sigma = 0.05 * np.ones_like(x)

    np.savez(
        OUTPUT_DIR / "polynomial_data.npz",
        x=x,
        y=y_measured,
        sigma=sigma,
    )
    print(f"   Created: polynomial_data.npz ({len(x)} points)")

    # -------------------------------------------------------------------------
    # 6. 2D Surface Data (ASCII text format for 2D fitting)
    # -------------------------------------------------------------------------
    print("\n6. Generating surface_2d.txt...")
    nx, ny = 30, 30
    x_1d = np.linspace(-3, 3, nx)
    y_1d = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x_1d, y_1d)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # True parameters: amplitude=100, x0=0.3, y0=-0.2, sigma_x=1.0, sigma_y=0.8, offset=10
    z_true = gaussian_2d(x_flat, y_flat, 100.0, 0.3, -0.2, 1.0, 0.8, 10.0)
    noise = np.random.normal(0, np.sqrt(z_true + 5))
    z_measured = z_true + noise
    sigma_z = np.sqrt(z_measured + 5)

    with open(OUTPUT_DIR / "surface_2d.txt", "w") as f:
        f.write("# 2D Gaussian surface data\n")
        f.write("# Columns: x  y  z  sigma\n")
        f.write(
            "# True parameters: amplitude=100, x0=0.3, y0=-0.2, sigma_x=1.0, sigma_y=0.8, offset=10\n"
        )
        f.writelines(
            f"{xi:10.6f}  {yi:10.6f}  {zi:12.6f}  {si:10.6f}\n"
            for xi, yi, zi, si in zip(x_flat, y_flat, z_measured, sigma_z, strict=False)
        )
    print(f"   Created: surface_2d.txt ({len(x_flat)} points)")

    # -------------------------------------------------------------------------
    # 7. Large Dataset (HDF5 format)
    # -------------------------------------------------------------------------
    print("\n7. Generating large_dataset.h5...")
    x = np.linspace(0, 100, 10000)
    # True parameters: a=10.0, b=0.1, c=2.0
    y_true = exponential_decay(x, 10.0, 0.1, 2.0)
    noise = np.random.normal(0, 0.2, size=len(x))
    y_measured = y_true + noise
    sigma = 0.2 * np.ones_like(x)

    with h5py.File(OUTPUT_DIR / "large_dataset.h5", "w") as f:
        data_group = f.create_group("data")
        data_group.create_dataset("x", data=x)
        data_group.create_dataset("y", data=y_measured)
        data_group.create_dataset("sigma", data=sigma)
        # Add metadata
        data_group.attrs["true_params"] = "a=10.0, b=0.1, c=2.0"
        data_group.attrs["model"] = "exponential_decay"
    print(f"   Created: large_dataset.h5 ({len(x)} points)")

    # -------------------------------------------------------------------------
    # 8. Multiple datasets for batch processing
    # -------------------------------------------------------------------------
    print("\n8. Generating batch processing datasets...")
    for i in range(1, 4):
        x = np.linspace(0, 10, 50)
        # Vary parameters slightly for each dataset
        a = 5.0 + i * 0.5
        b = 0.3 + i * 0.1
        c = 1.0
        y_true = exponential_decay(x, a, b, c)
        noise = np.random.normal(0, 0.15, size=len(x))
        y_measured = y_true + noise
        sigma = 0.15 * np.ones_like(x)

        with open(OUTPUT_DIR / f"batch_data_{i}.csv", "w") as f:
            f.write(f"# Batch dataset {i}: a={a}, b={b}, c={c}\n")
            f.write("x,y,sigma\n")
            f.writelines(
                f"{xi:.6f},{yi:.6f},{si:.6f}\n"
                for xi, yi, si in zip(x, y_measured, sigma, strict=False)
            )
        print(f"   Created: batch_data_{i}.csv ({len(x)} points)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Sample Data Generation Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")
    for f in sorted(OUTPUT_DIR.glob("*.txt")):
        print(f"  - {f.name}")
    for f in sorted(OUTPUT_DIR.glob("*.npz")):
        print(f"  - {f.name}")
    for f in sorted(OUTPUT_DIR.glob("*.h5")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

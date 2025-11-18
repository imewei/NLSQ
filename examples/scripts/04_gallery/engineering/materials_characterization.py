"""
Converted from materials_characterization.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # Materials Characterization: Stress-Strain Curve Analysis
#
#
# This example demonstrates fitting stress-strain curves to extract mechanical
# properties of materials. We analyze elastic modulus, yield point, and strain
# hardening behavior using realistic tensile test data.
#
# Key Concepts:
# - Piecewise linear fitting (elastic + plastic regions)
# - Elastic modulus (Young's modulus) extraction
# - Yield strength determination (0.2% offset method)
# - Strain hardening coefficient calculation
# - Ultimate tensile strength identification
#
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import curve_fit

# Set random seed
np.random.seed(42)


def linear_elastic(strain, E):
    """
    Linear elastic stress-strain relationship (Hooke's Law).

    σ = E * ε  # noqa: RUF003

    Parameters
    ----------
    strain : array_like
        Engineering strain (dimensionless)
    E : float
        Young's modulus (GPa)

    Returns
    -------
    stress : array_like
        Engineering stress (MPa)
    """
    return E * 1000 * strain  # Convert GPa to MPa


def ramberg_osgood(strain, E, sigma_y, n):
    """
    Ramberg-Osgood model for elastoplastic behavior.

    ε = σ/E + (σ/K)^n

    Rearranged for stress as function of strain (implicit, approximated).

    Parameters
    ----------
    strain : array_like
        Engineering strain
    E : float
        Young's modulus (GPa)
    sigma_y : float
        Yield strength (MPa)
    n : float
        Strain hardening exponent

    Returns
    -------
    stress : array_like
        Engineering stress (MPa)
    """
    # Simplified approximation for demonstration
    # For small strains: linear elastic
    # For large strains: power law hardening
    E_MPa = E * 1000  # Convert to MPa
    elastic_stress = E_MPa * strain

    # Transition at yield point
    epsilon_y = sigma_y / E_MPa

    # Plastic contribution (simplified)
    plastic_stress = sigma_y * jnp.power(
        jnp.maximum(strain - epsilon_y, 0) / epsilon_y + 1, 1 / n
    )

    # Combined (smooth transition)
    return jnp.where(strain < epsilon_y, elastic_stress, plastic_stress)


def power_law_hardening(strain, sigma_y, K, n):
    """
    Power law strain hardening model (Hollomon equation).

    σ = K * ε^n  # noqa: RUF003

    Valid for plastic region only.

    Parameters
    ----------
    strain : array_like
        Plastic strain (dimensionless)
    sigma_y : float
        Yield strength (MPa, for reference)
    K : float
        Strength coefficient (MPa)
    n : float
        Strain hardening exponent

    Returns
    -------
    stress : array_like
        True stress (MPa)
    """
    return K * jnp.power(strain, n)


# Simulate aluminum alloy (6061-T6 like properties)
# True material parameters
E_true = 69.0  # GPa (Young's modulus)
sigma_y_true = 275.0  # MPa (yield strength)
UTS_true = 310.0  # MPa (ultimate tensile strength)
epsilon_y_true = sigma_y_true / (E_true * 1000)  # Yield strain

# Generate strain data (0% to 12% strain)
strain = np.linspace(0, 0.12, 250)

# Build stress-strain curve in three regions:
# 1. Elastic region (0 to yield)
mask_elastic = strain <= epsilon_y_true
stress_elastic = E_true * 1000 * strain[mask_elastic]

# 2. Plastic region (yield to UTS) - strain hardening
mask_plastic = (strain > epsilon_y_true) & (strain <= 0.08)
strain_plastic = strain[mask_plastic] - epsilon_y_true
# Power law hardening: σ = σ_y + K * ε_p^n  # noqa: RUF003
K_hardening = 450.0  # MPa
n_hardening = 0.3
stress_plastic = sigma_y_true + K_hardening * strain_plastic**n_hardening

# 3. Necking region (UTS to fracture) - stress decreases
mask_necking = strain > 0.08
stress_necking = UTS_true - 200 * (strain[mask_necking] - 0.08)

# Combine regions
stress_true = np.concatenate([stress_elastic, stress_plastic, stress_necking])

# Add measurement noise (realistic for extensometer + load cell)
# Strain: ±0.05% (0.0005), Stress: ±2 MPa
strain_noise = np.random.normal(0, 0.0005, size=len(strain))
stress_noise = np.random.normal(0, 2.0, size=len(stress_true))

strain_measured = strain + strain_noise
stress_measured = stress_true + stress_noise

# Measurement uncertainties
sigma_strain = 0.0005 * np.ones_like(strain_measured)
sigma_stress = 2.0 * np.ones_like(stress_measured)


print("=" * 70)
print("MATERIALS CHARACTERIZATION: STRESS-STRAIN ANALYSIS")
print("=" * 70)

# Select elastic region (typically < 50% of yield stress)
elastic_limit = 0.5 * sigma_y_true
mask_fit_elastic = stress_measured < elastic_limit

print("\n" + "-" * 70)
print("ELASTIC REGION ANALYSIS")
print("-" * 70)

# Fit linear elastic model
popt_elastic, pcov_elastic = curve_fit(
    linear_elastic,
    strain_measured[mask_fit_elastic],
    stress_measured[mask_fit_elastic],
    p0=[70],  # Initial guess for E (GPa)
    sigma=sigma_stress[mask_fit_elastic],
    absolute_sigma=True,
)

E_fit = popt_elastic[0]
E_err = np.sqrt(pcov_elastic[0, 0])

print(f"Young's Modulus (E): {E_fit:.2f} ± {E_err:.2f} GPa")
print(f"True value:          {E_true:.2f} GPa")
print(
    f"Error:               {abs(E_fit - E_true):.2f} GPa "
    + f"({100 * abs(E_fit - E_true) / E_true:.1f}%)"
)


print("\n" + "-" * 70)
print("YIELD STRENGTH DETERMINATION (0.2% Offset Method)")
print("-" * 70)

# 0.2% offset line: σ = E * (ε - 0.002)  # noqa: RUF003
offset = 0.002
offset_line = E_fit * 1000 * (strain_measured - offset)

# Find intersection (yield point)
# Approximate by finding where stress curve crosses offset line
differences = stress_measured - offset_line
# Find first crossing from below
sign_changes = np.diff(np.sign(differences))
yield_index = np.where(sign_changes > 0)[0]

if len(yield_index) > 0:
    yield_index = yield_index[0]
    sigma_y_fit = stress_measured[yield_index]
    epsilon_y_fit = strain_measured[yield_index]

    print(f"Yield Strength (σ_y):     {sigma_y_fit:.1f} MPa")
    print(f"Yield Strain (ε_y):       {epsilon_y_fit:.4f} ({100 * epsilon_y_fit:.2f}%)")
    print(f"True yield strength:      {sigma_y_true:.1f} MPa")
    print(f"Error:                    {abs(sigma_y_fit - sigma_y_true):.1f} MPa")
else:
    sigma_y_fit = sigma_y_true
    epsilon_y_fit = epsilon_y_true
    print("Warning: Could not determine yield point, using estimates")


print("\n" + "-" * 70)
print("PLASTIC REGION ANALYSIS (Strain Hardening)")
print("-" * 70)

# Select plastic region (yield to peak stress)
mask_fit_plastic = (strain_measured > epsilon_y_fit) & (strain_measured < 0.08)

if np.sum(mask_fit_plastic) > 10:
    # Fit power law to plastic region
    # Convert to true stress/strain for better fit
    strain_plastic_fit = strain_measured[mask_fit_plastic] - epsilon_y_fit
    stress_plastic_fit = stress_measured[mask_fit_plastic]

    # Hollomon model: σ = K * ε^n  # noqa: RUF003
    def hollomon_model(eps, K, n):
        return K * jnp.power(eps + 1e-6, n)  # Add small offset to avoid log(0)

    popt_plastic, pcov_plastic = curve_fit(
        hollomon_model,
        strain_plastic_fit,
        stress_plastic_fit,
        p0=[450, 0.3],
        sigma=sigma_stress[mask_fit_plastic],
        bounds=([100, 0.01], [1000, 1.0]),
        absolute_sigma=True,
    )

    K_fit, n_fit = popt_plastic
    perr_plastic = np.sqrt(np.diag(pcov_plastic))
    K_err, n_err = perr_plastic

    print(f"Strength Coefficient (K): {K_fit:.1f} ± {K_err:.1f} MPa")
    print(f"Hardening Exponent (n):   {n_fit:.3f} ± {n_err:.3f}")
    print(f"True values:              K={K_hardening:.1f} MPa, n={n_hardening:.3f}")

    # Strain hardening rate
    # dσ/dε = n*K*ε^(n-1)  # noqa: RUF003
    eps_avg = np.mean(strain_plastic_fit)
    hardening_rate = n_fit * K_fit * eps_avg ** (n_fit - 1)
    print(
        f"\nStrain hardening rate:    {hardening_rate:.1f} MPa "
        + f"(at ε_p = {eps_avg:.4f})"
    )


print("\n" + "-" * 70)
print("ULTIMATE TENSILE STRENGTH")
print("-" * 70)

UTS_fit = np.max(stress_measured)
UTS_strain = strain_measured[np.argmax(stress_measured)]

print(f"Ultimate Tensile Strength: {UTS_fit:.1f} MPa")
print(f"Strain at UTS:             {UTS_strain:.4f} ({100 * UTS_strain:.2f}%)")
print(f"True UTS:                  {UTS_true:.1f} MPa")
print(f"Error:                     {abs(UTS_fit - UTS_true):.1f} MPa")


print("\n" + "=" * 70)
print("MATERIAL PROPERTIES SUMMARY")
print("=" * 70)

# Ductility
strain_at_fracture = strain_measured[-1]
elongation = 100 * strain_at_fracture
print("\nElastic Properties:")
print(f"  Young's Modulus (E):      {E_fit:.2f} GPa")
print(f"  Proportional Limit:       ~{elastic_limit:.0f} MPa")

print("\nStrength Properties:")
print(f"  Yield Strength (0.2%):    {sigma_y_fit:.1f} MPa")
print(f"  Ultimate Tensile Strength: {UTS_fit:.1f} MPa")
print(f"  Strength Ratio (UTS/σ_y): {UTS_fit / sigma_y_fit:.2f}")

print("\nPlastic Properties:")
print(f"  Hardening Coefficient (K): {K_fit:.1f} MPa")
print(f"  Hardening Exponent (n):    {n_fit:.3f}")
print(f"  Elongation at fracture:    {elongation:.1f}%")

# Toughness (area under curve - approximate with trapezoid rule)
toughness = np.trapezoid(stress_measured, strain_measured)
print(f"  Toughness (area):          {toughness:.2f} MPa")

# Material classification
print("\nMaterial Classification:")
if E_fit > 150:
    print("  → High stiffness (E > 150 GPa)")
elif E_fit > 45:
    print("  → Medium stiffness (45 < E < 150 GPa)")
    print("  → Likely aluminum alloy")
else:
    print("  → Low stiffness (E < 45 GPa)")

if elongation > 20:
    print("  → Ductile material (elongation > 20%)")
elif elongation > 5:
    print("  → Moderately ductile (5% < elongation < 20%)")
else:
    print("  → Brittle material (elongation < 5%)")


fig = plt.figure(figsize=(16, 12))

# Plot 1: Full stress-strain curve
ax1 = plt.subplot(3, 2, 1)
ax1.plot(
    strain_measured * 100,
    stress_measured,
    "o",
    alpha=0.4,
    markersize=3,
    label="Experimental data",
)

# Plot fitted regions
strain_fine = np.linspace(0, epsilon_y_fit, 100)
ax1.plot(
    strain_fine * 100,
    linear_elastic(strain_fine, E_fit),
    "g-",
    linewidth=2.5,
    label=f"Elastic fit (E={E_fit:.1f} GPa)",
)

if np.sum(mask_fit_plastic) > 10:
    strain_plastic_fine = np.linspace(0, np.max(strain_plastic_fit), 100)
    stress_plastic_fine = hollomon_model(strain_plastic_fine, K_fit, n_fit)
    ax1.plot(
        (strain_plastic_fine + epsilon_y_fit) * 100,
        stress_plastic_fine,
        "b-",
        linewidth=2.5,
        label=f"Plastic fit (n={n_fit:.3f})",
    )

# Mark key points
ax1.axhline(
    sigma_y_fit,
    color="orange",
    linestyle="--",
    linewidth=1.5,
    label=f"Yield ({sigma_y_fit:.0f} MPa)",
)
ax1.axhline(
    UTS_fit,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"UTS ({UTS_fit:.0f} MPa)",
)
ax1.axvline(epsilon_y_fit * 100, color="orange", linestyle=":", alpha=0.5)

ax1.set_xlabel("Strain (%)", fontsize=12)
ax1.set_ylabel("Stress (MPa)", fontsize=12)
ax1.set_title("Engineering Stress-Strain Curve", fontsize=14, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Plot 2: Elastic region with 0.2% offset line
ax2 = plt.subplot(3, 2, 2)
mask_zoom = strain_measured * 100 < 1.5
ax2.plot(
    strain_measured[mask_zoom] * 100,
    stress_measured[mask_zoom],
    "o",
    alpha=0.6,
    markersize=4,
    label="Data",
)

# Elastic line
strain_elastic_plot = np.linspace(0, 0.015, 100)
ax2.plot(
    strain_elastic_plot * 100,
    linear_elastic(strain_elastic_plot, E_fit),
    "g-",
    linewidth=2.5,
    label=f"Elastic (E={E_fit:.1f} GPa)",
)

# 0.2% offset line
ax2.plot(
    strain_elastic_plot * 100,
    E_fit * 1000 * (strain_elastic_plot - offset),
    "r--",
    linewidth=2,
    label="0.2% offset line",
)

# Mark yield point
ax2.plot(
    epsilon_y_fit * 100,
    sigma_y_fit,
    "o",
    markersize=10,
    color="orange",
    label=f"Yield point ({sigma_y_fit:.0f} MPa)",
)

ax2.set_xlabel("Strain (%)")
ax2.set_ylabel("Stress (MPa)")
ax2.set_title("Yield Strength Determination (0.2% Offset Method)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Plastic region (log-log for power law)
ax3 = plt.subplot(3, 2, 3)
if np.sum(mask_fit_plastic) > 10:
    ax3.loglog(
        strain_plastic_fit,
        stress_plastic_fit - sigma_y_fit,
        "o",
        alpha=0.5,
        markersize=4,
        label="Data",
    )

    strain_log = np.logspace(
        np.log10(strain_plastic_fit.min()), np.log10(strain_plastic_fit.max()), 100
    )
    stress_log = hollomon_model(strain_log, K_fit, n_fit) - sigma_y_fit
    ax3.loglog(
        strain_log,
        stress_log,
        "b-",
        linewidth=2.5,
        label=f"σ = {K_fit:.0f}ε^{n_fit:.3f}",
    )

    ax3.set_xlabel("Plastic Strain (log scale)")
    ax3.set_ylabel("Stress - σ_y (MPa, log scale)")
    ax3.set_title("Strain Hardening (Log-Log Plot)")
    ax3.legend()
    ax3.grid(True, alpha=0.3, which="both")

# Plot 4: Strain hardening rate
ax4 = plt.subplot(3, 2, 4)
if np.sum(mask_fit_plastic) > 10:
    # Compute numerical derivative
    dstress = np.gradient(
        stress_measured[mask_fit_plastic], strain_measured[mask_fit_plastic]
    )

    ax4.plot(
        strain_measured[mask_fit_plastic] * 100,
        dstress,
        "o",
        alpha=0.5,
        markersize=4,
        label="Numerical dσ/dε",
    )

    # Analytical derivative from power law
    eps_range = strain_plastic_fit
    analytical_dstress = n_fit * K_fit * eps_range ** (n_fit - 1)
    ax4.plot(
        (eps_range + epsilon_y_fit) * 100,
        analytical_dstress,
        "b-",
        linewidth=2,
        label="Analytical (power law)",
    )

    ax4.set_xlabel("Strain (%)")
    ax4.set_ylabel("Strain Hardening Rate (MPa)")
    ax4.set_title("Strain Hardening Rate (dσ/dε)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# Plot 5: Residuals (elastic region)
ax5 = plt.subplot(3, 2, 5)
residuals_elastic = stress_measured[mask_fit_elastic] - linear_elastic(
    strain_measured[mask_fit_elastic], E_fit
)
ax5.plot(
    strain_measured[mask_fit_elastic] * 100,
    residuals_elastic,
    "o",
    alpha=0.5,
    markersize=4,
)
ax5.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax5.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax5.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax5.set_xlabel("Strain (%)")
ax5.set_ylabel("Residual (MPa)")
ax5.set_title("Elastic Fit Residuals")
ax5.grid(True, alpha=0.3)

# Plot 6: Toughness visualization (energy absorption)
ax6 = plt.subplot(3, 2, 6)
ax6.fill_between(
    strain_measured * 100,
    0,
    stress_measured,
    alpha=0.3,
    label=f"Toughness = {toughness:.1f} MPa",
)
ax6.plot(strain_measured * 100, stress_measured, "b-", linewidth=2)
ax6.set_xlabel("Strain (%)")
ax6.set_ylabel("Stress (MPa)")
ax6.set_title("Toughness (Energy Absorption)")
ax6.legend(loc="upper left")
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("materials_characterization.png", dpi=150)
print("\n✅ Plot saved as 'materials_characterization.png'")
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "materials_characterization"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Material properties successfully characterized:")
print("\n  Material type: Aluminum alloy (6061-T6 like)")
print(f"  Young's Modulus:  {E_fit:.2f} GPa")
print(f"  Yield Strength:   {sigma_y_fit:.0f} MPa (0.2% offset)")
print(f"  UTS:              {UTS_fit:.0f} MPa")
print(f"  Hardening (n):    {n_fit:.3f}")
print(f"  Elongation:       {elongation:.1f}%")
print(f"  Toughness:        {toughness:.1f} MPa")
print("\nThis example demonstrates:")
print("  ✓ Elastic modulus extraction from stress-strain data")
print("  ✓ Yield strength determination (0.2% offset method)")
print("  ✓ Strain hardening analysis (power law fitting)")
print("  ✓ Ultimate tensile strength identification")
print("  ✓ Material classification based on mechanical properties")
print("  ✓ Toughness calculation (energy absorption)")
print("=" * 70)

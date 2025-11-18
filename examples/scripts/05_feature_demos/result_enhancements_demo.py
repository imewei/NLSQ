"""
Converted from result_enhancements_demo.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # Demo: Enhanced Result Objects with Statistical Analysis
#
# This example demonstrates how to use NLSQ's enhanced CurveFitResult class
# to access statistical properties, confidence intervals, and visualization.
#
# ======================================================================

import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit

# ======================================================================
# ## Example 1: Basic Statistical Properties
#
# ======================================================================


def example1_statistical_properties():
    """Demonstrate basic statistical properties of curve fit result."""
    print("\n" + "=" * 70)
    print("Example 1: Statistical Properties")
    print("=" * 70)

    # Define exponential decay model
    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 10 * np.exp(-0.5 * x) + 2
    y = y_true + np.random.normal(0, 0.5, size=len(x))

    # Fit model
    result = curve_fit(exponential, x, y, p0=[10, 0.5, 2])

    # Access statistical properties
    print("\nFitted parameters:")
    print(f"  a = {result.popt[0]:.4f}")
    print(f"  b = {result.popt[1]:.4f}")
    print(f"  c = {result.popt[2]:.4f}")

    print("\nGoodness of fit:")
    print(f"  R² = {result.r_squared:.6f}")
    print(f"  Adjusted R² = {result.adj_r_squared:.6f}")
    print(f"  RMSE = {result.rmse:.6f}")
    print(f"  MAE = {result.mae:.6f}")

    print("\nModel selection criteria:")
    print(f"  AIC = {result.aic:.2f}")
    print(f"  BIC = {result.bic:.2f}")

    print("\n✓ Statistical properties accessed successfully!")


# ======================================================================
# ## Example 2: Backward Compatibility
#
# ======================================================================


def example2_backward_compatibility():
    """Demonstrate backward compatibility with tuple unpacking."""
    print("\n" + "=" * 70)
    print("Example 2: Backward Compatibility")
    print("=" * 70)

    def linear(x, a, b):
        return a * x + b

    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 0.5, size=len(x))

    # Pattern 1: Traditional tuple unpacking (backward compatible)
    popt, pcov = curve_fit(linear, x, y, p0=[1, 1])
    print("\nPattern 1 (tuple unpacking):")
    print(f"  popt = {popt}")
    print(f"  pcov shape = {pcov.shape}")

    # Pattern 2: Enhanced result object
    result = curve_fit(linear, x, y, p0=[1, 1])
    print("\nPattern 2 (enhanced result):")
    print(f"  result.popt = {result.popt}")
    print(f"  result.r_squared = {result.r_squared:.6f}")

    # Pattern 3: Can still unpack enhanced result
    popt2, pcov2 = result
    print("\nPattern 3 (unpack enhanced result):")
    print(f"  popt = {popt2}")
    print(f"  Same as Pattern 1? {np.allclose(popt, popt2)}")

    print("\n✓ All usage patterns work seamlessly!")


# ======================================================================
# ## Example 3: Confidence Intervals
#
# ======================================================================


def example3_confidence_intervals():
    """Demonstrate parameter confidence intervals."""
    print("\n" + "=" * 70)
    print("Example 3: Confidence Intervals")
    print("=" * 70)

    def power_law(x, a, b):
        return a * x**b

    np.random.seed(42)
    x = np.linspace(1, 10, 50)
    y = 2 * x**1.5 + np.random.normal(0, 2, size=len(x))

    # Fit model
    result = curve_fit(power_law, x, y, p0=[2, 1.5])

    # Get confidence intervals
    ci_95 = result.confidence_intervals(alpha=0.95)
    ci_99 = result.confidence_intervals(alpha=0.99)

    print("\nFitted parameters with confidence intervals:")
    print("\nParameter    Value       95% CI                    99% CI")
    print("-" * 70)
    for i, (val, ci95, ci99) in enumerate(zip(result.popt, ci_95, ci_99, strict=False)):
        print(
            f"p{i:<11} {val:>8.4f}    [{ci95[0]:>7.4f}, {ci95[1]:>7.4f}]    "
            f"[{ci99[0]:>7.4f}, {ci99[1]:>7.4f}]"
        )

    print("\n✓ Confidence intervals computed!")


# ======================================================================
# ## Example 4: Prediction Intervals
#
# ======================================================================


def example4_prediction_intervals():
    """Demonstrate prediction intervals."""
    print("\n" + "=" * 70)
    print("Example 4: Prediction Intervals")
    print("=" * 70)

    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    np.random.seed(42)
    x = np.linspace(0, 5, 30)
    y = 0.5 * x**2 - 2 * x + 1 + np.random.normal(0, 0.3, size=len(x))

    # Fit model
    result = curve_fit(quadratic, x, y, p0=[1, -2, 1])

    # Get prediction intervals at fitted x values
    pi = result.prediction_interval()

    print("\nPrediction intervals at first 5 data points:")
    print("\n  x       y_data    y_pred    Lower     Upper")
    print("  " + "-" * 50)
    for i in range(5):
        print(
            f"  {x[i]:.2f}    {y[i]:>6.3f}    {result.predictions[i]:>6.3f}    "
            f"{pi[i, 0]:>6.3f}    {pi[i, 1]:>6.3f}"
        )

    # Get prediction intervals at new x values
    x_new = np.array([1.5, 3.0, 4.5])
    pi_new = result.prediction_interval(x=x_new)

    print("\nPrediction intervals at new x values:")
    print("\n  x_new    y_pred    Lower     Upper    Width")
    print("  " + "-" * 50)
    for i, x_val in enumerate(x_new):
        width = pi_new[i, 1] - pi_new[i, 0]
        y_pred = result.model(x_val, *result.popt)
        print(
            f"  {x_val:.2f}     {y_pred:>6.3f}    {pi_new[i, 0]:>6.3f}    "
            f"{pi_new[i, 1]:>6.3f}    {width:>6.3f}"
        )

    print("\n✓ Prediction intervals computed!")


# ======================================================================
# ## Example 5: Visualization
#
# ======================================================================


# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib


def example5_visualization():
    """Demonstrate built-in plotting functionality."""
    print("\n" + "=" * 70)
    print("Example 5: Visualization")
    print("=" * 70)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠ matplotlib not installed - skipping visualization example")
        return

    def gaussian(x, a, mu, sigma):
        return a * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

    np.random.seed(42)
    x = np.linspace(-5, 5, 100)
    y = 10 * np.exp(-((x - 1) ** 2) / (2 * 1.5**2)) + np.random.normal(
        0, 0.5, size=len(x)
    )

    # Fit model
    result = curve_fit(gaussian, x, y, p0=[10, 1, 1.5])

    # Plot with residuals
    print("\nGenerating plot with residuals...")
    result.plot(show_residuals=True)
    plt.savefig("curve_fit_result.png", dpi=150, bbox_inches="tight")
    print("  ✓ Plot saved to 'curve_fit_result.png'")

    # Plot without residuals
    print("\nGenerating plot without residuals...")
    fig, ax = plt.subplots(figsize=(8, 6))
    result.plot(ax=ax, show_residuals=False, color="blue", alpha=0.5)
    plt.savefig("curve_fit_result_no_residuals.png", dpi=150, bbox_inches="tight")
    print("  ✓ Plot saved to 'curve_fit_result_no_residuals.png'")

    plt.close("all")
    print("\n✓ Visualization completed!")


# ======================================================================
# ## Example 6: Summary Report
#
# ======================================================================


def example6_summary_report():
    """Demonstrate statistical summary report."""
    print("\n" + "=" * 70)
    print("Example 6: Summary Report")
    print("=" * 70)

    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 10 * np.exp(-0.5 * x) + 2
    y = y_true + np.random.normal(0, 0.5, size=len(x))

    # Fit model
    result = curve_fit(exponential, x, y, p0=[10, 0.5, 2])

    # Print summary
    print("\nGenerating statistical summary report...\n")
    result.summary()

    print("\n✓ Summary report generated!")


# ======================================================================
# ## Example 7: Model Comparison
#
# ======================================================================


def example7_model_comparison():
    """Demonstrate comparing multiple models using AIC/BIC."""
    print("\n" + "=" * 70)
    print("Example 7: Model Comparison")
    print("=" * 70)

    # Generate data with exponential decay
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 10 * np.exp(-0.5 * x) + 2
    y = y_true + np.random.normal(0, 0.5, size=len(x))

    # Model 1: Exponential (correct model)
    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    # Model 2: Linear (wrong model)
    def linear(x, a, b):
        return a * x + b

    # Model 3: Quadratic (overfitted)
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    # Fit all models
    result_exp = curve_fit(exponential, x, y, p0=[10, 0.5, 2])
    result_lin = curve_fit(linear, x, y, p0=[-1, 10])
    result_quad = curve_fit(quadratic, x, y, p0=[0, -1, 10])

    # Compare models
    print("\nModel Comparison:")
    print(
        f"\n{'Model':<15} {'Params':<8} {'R²':<10} {'RMSE':<10} {'AIC':<10} {'BIC':<10}"
    )
    print("-" * 70)

    models = [
        ("Exponential", result_exp),
        ("Linear", result_lin),
        ("Quadratic", result_quad),
    ]

    for name, res in models:
        print(
            f"{name:<15} {len(res.popt):<8} {res.r_squared:<10.6f} "
            f"{res.rmse:<10.6f} {res.aic:<10.2f} {res.bic:<10.2f}"
        )

    print("\nBest model (lowest AIC): ", end="")
    best_idx = np.argmin([r.aic for _, r in models])
    print(f"{models[best_idx][0]}")

    print("\n✓ Model comparison completed!")
    print("✓ Exponential model has the lowest AIC/BIC (correct model)")


# ======================================================================
# ## Example 8: Accessing Residuals and Predictions
#
# ======================================================================


def example8_residuals_predictions():
    """Demonstrate accessing residuals and predictions."""
    print("\n" + "=" * 70)
    print("Example 8: Residuals and Predictions")
    print("=" * 70)

    def sine_wave(x, a, f, phi):
        return a * jnp.sin(2 * jnp.pi * f * x + phi)

    np.random.seed(42)
    x = np.linspace(0, 2, 50)
    y = 3 * np.sin(2 * np.pi * 2 * x + 0.5) + np.random.normal(0, 0.3, size=len(x))

    # Fit model
    result = curve_fit(sine_wave, x, y, p0=[3, 2, 0.5])

    # Access residuals
    residuals = result.residuals
    print("\nResiduals:")
    print(f"  Mean: {np.mean(residuals):.6f} (should be ~0)")
    print(f"  Std:  {np.std(residuals):.6f}")
    print(f"  Min:  {np.min(residuals):.6f}")
    print(f"  Max:  {np.max(residuals):.6f}")

    # Access predictions
    predictions = result.predictions
    print("\nPredictions:")
    print(f"  Shape: {predictions.shape}")
    print(f"  Correlation with data: {np.corrcoef(predictions, y)[0, 1]:.6f}")

    # Verify residuals = data - predictions
    manual_residuals = y - predictions
    print("\nVerification:")
    print(
        f"  Residuals match (data - predictions)? "
        f"{np.allclose(residuals, manual_residuals)}"
    )

    print("\n✓ Residuals and predictions accessed successfully!")


# ======================================================================
# ## Main Demo
#
# ======================================================================


def main():
    """Run all result enhancement examples."""
    print("\n" + "=" * 70)
    print("NLSQ Result Object Enhancements Demo")
    print("=" * 70)
    print("\nDemonstrating enhanced CurveFitResult with statistical analysis,")
    print("confidence intervals, and visualization capabilities.\n")

    # Run examples
    example1_statistical_properties()
    example2_backward_compatibility()
    example3_confidence_intervals()
    example4_prediction_intervals()
    example5_visualization()
    example6_summary_report()
    example7_model_comparison()
    example8_residuals_predictions()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("  • Statistical properties: R², RMSE, MAE, AIC, BIC")
    print("  • Confidence intervals: Parameter uncertainty quantification")
    print("  • Prediction intervals: Future data point uncertainty")
    print("  • Visualization: Built-in plot() method with residuals")
    print("  • Summary reports: Comprehensive fit statistics")
    print("  • Model comparison: AIC/BIC for selecting best model")
    print("  • Backward compatible: Supports tuple unpacking")
    print("\nUsage:")
    print("  # Traditional usage (still works)")
    print("  popt, pcov = curve_fit(f, x, y)")
    print()
    print("  # Enhanced usage")
    print("  result = curve_fit(f, x, y)")
    print("  print(f'R² = {result.r_squared:.4f}')")
    print("  result.plot()")
    print("  result.summary()")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

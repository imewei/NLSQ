"""Custom Preset Guide: Building Domain-Specific Configurations.

This guide demonstrates patterns for creating domain-specific fitting
configurations in NLSQ. The library provides built-in presets that can be
used directly or customized for specific scientific or engineering domains.

Key patterns covered:
1. Using built-in WORKFLOW_PRESETS
2. Creating custom configurations with fit()
3. Building preset factories for your domain
4. Common parameter adjustments by use case

Run this example:
    python examples/scripts/08_workflow_system/08_custom_presets.py
"""

import jax.numpy as jnp
import numpy as np

from nlsq import fit
from nlsq.core.minpack import WORKFLOW_PRESETS


def main():
    print("=" * 70)
    print("Custom Preset Guide")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Available Base Presets
    # =========================================================================
    print("1. Available Base Presets")
    print("-" * 60)
    print()
    print("NLSQ provides these presets as starting points:")
    print()

    for name, config in WORKFLOW_PRESETS.items():
        desc = config.get("description", "No description")
        print(f"  {name:<20} - {desc}")

    # =========================================================================
    # 2. Preset Details
    # =========================================================================
    print()
    print()
    print("2. Preset Details")
    print("-" * 60)

    presets_to_show = ["standard", "quality", "fast", "streaming"]

    for preset_name in presets_to_show:
        if preset_name in WORKFLOW_PRESETS:
            print(f"\n  '{preset_name}' preset:")
            for key, value in WORKFLOW_PRESETS[preset_name].items():
                print(f"    {key}: {value}")

    # =========================================================================
    # 3. Using Presets with fit()
    # =========================================================================
    print()
    print()
    print("3. Using Presets with fit()")
    print("-" * 60)
    print()
    print("  Basic usage:")
    print()
    print("    # Use built-in preset")
    print("    popt, pcov = fit(model, x, y, workflow='quality')")
    print()
    print("    # Use automatic memory-based selection")
    print("    popt, pcov = fit(model, x, y, workflow='auto')")
    print()
    print("  With additional parameters:")
    print()
    print("    popt, pcov = fit(")
    print("        model, x, y,")
    print("        workflow='standard',")
    print("        multistart=True,      # Override preset setting")
    print("        n_starts=20,          # Custom number of starts")
    print("        sampler='sobol',      # Different sampler")
    print("    )")

    # =========================================================================
    # 4. Common Override Patterns
    # =========================================================================
    print()
    print()
    print("4. Common Override Patterns")
    print("-" * 60)

    patterns = [
        {
            "name": "Pattern A: Increase multi-start coverage",
            "use_case": "Models with multiple local minima",
            "code": "fit(f, x, y, workflow='standard', multistart=True, n_starts=30, sampler='sobol')",
        },
        {
            "name": "Pattern B: Tighten tolerances",
            "use_case": "High-precision structural parameters",
            "code": "fit(f, x, y, workflow='standard', gtol=1e-12, ftol=1e-12, xtol=1e-12)",
        },
        {
            "name": "Pattern C: Memory-constrained fitting",
            "use_case": "Systems with limited RAM",
            "code": "fit(f, x, y, workflow='streaming', memory_limit_gb=8.0)",
        },
        {
            "name": "Pattern D: Fast exploration",
            "use_case": "Quick iterative fitting",
            "code": "fit(f, x, y, workflow='fast')",
        },
    ]

    for pattern in patterns:
        print(f"\n  {pattern['name']}")
        print(f"  Use case: {pattern['use_case']}")
        print(f"  Code: {pattern['code']}")

    # =========================================================================
    # 5. Creating Reusable Preset Factories
    # =========================================================================
    print()
    print()
    print("5. Creating Reusable Preset Factories")
    print("-" * 60)
    print()
    print("  Define functions that return customized fit kwargs for your domain:")
    print()

    def create_spectroscopy_preset(high_resolution: bool = False) -> dict:
        """Create preset kwargs for spectroscopic peak fitting.

        Parameters
        ----------
        high_resolution : bool
            If True, use tighter tolerances for high-resolution spectra.
        """
        if high_resolution:
            return {
                "workflow": "quality",
                "multistart": True,
                "n_starts": 15,
                "sampler": "lhs",
            }
        else:
            return {
                "workflow": "standard",
                "multistart": True,
                "n_starts": 10,
                "sampler": "lhs",
            }

    def create_timeseries_preset(n_points: int) -> dict:
        """Create preset kwargs for time series analysis.

        Parameters
        ----------
        n_points : int
            Number of data points (affects workflow selection).
        """
        if n_points > 1_000_000:
            return {"workflow": "streaming"}
        else:
            return {
                "workflow": "standard",
                "multistart": True,
                "n_starts": 10,
            }

    def create_optimization_preset(n_params: int) -> dict:
        """Create preset kwargs based on parameter count.

        More parameters -> more multi-start coverage needed.
        """
        n_starts = max(10, n_params * 3)  # Scale with complexity
        return {
            "workflow": "standard",
            "multistart": True,
            "n_starts": n_starts,
            "sampler": "sobol",
        }

    # Demonstrate factories
    print("  Example factories:")
    print()
    print("    def create_spectroscopy_preset(high_resolution=False):")
    print("        if high_resolution:")
    print("            return {'workflow': 'quality', 'multistart': True, ...}")
    print("        else:")
    print("            return {'workflow': 'standard', ...}")
    print()
    print("  Usage:")
    print("    kwargs = create_spectroscopy_preset(high_resolution=True)")
    print("    popt, pcov = fit(model, x, y, **kwargs)")

    # =========================================================================
    # 6. Complete Example: Domain-Specific Fitting
    # =========================================================================
    print()
    print()
    print("6. Complete Example: Domain-Specific Fitting")
    print("-" * 60)
    print()

    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    np.random.seed(42)
    x_data = np.linspace(0, 5, 100)
    y_true = 2.5 * np.exp(-1.3 * x_data) + 0.5
    y_data = y_true + 0.1 * np.random.randn(100)

    print("  Testing with exponential decay fit...")
    print("  True parameters: a=2.5, b=1.3, c=0.5")

    # Use factory
    kwargs = create_spectroscopy_preset(high_resolution=True)
    print(f"  Preset kwargs: {kwargs}")

    popt, pcov = fit(
        exponential,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=([0.1, 0.1, -1.0], [10.0, 10.0, 2.0]),
        **kwargs,
    )

    print(f"  Fitted parameters: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")

    # =========================================================================
    # 7. Defense Preset Examples
    # =========================================================================
    print()
    print()
    print("7. Defense Presets for Streaming (v0.3.6+)")
    print("-" * 60)
    print()
    print("  For streaming workflows, use HybridStreamingConfig presets:")
    print()
    print("    from nlsq import HybridStreamingConfig")
    print()
    print("    # Warm-start refinement (checkpoint resume)")
    print("    config = HybridStreamingConfig.defense_strict()")
    print()
    print("    # Exploration (rough initial guesses)")
    print("    config = HybridStreamingConfig.defense_relaxed()")
    print()
    print("    # Production scientific computing")
    print("    config = HybridStreamingConfig.scientific_default()")
    print()
    print("  Then pass to fit():")
    print("    popt, pcov = fit(model, x, y, workflow=config)")

    # =========================================================================
    # 8. Summary: Key Takeaways
    # =========================================================================
    print()
    print()
    print("=" * 70)
    print("Summary: Key Takeaways")
    print("=" * 70)
    print()
    print("1. Use built-in presets for common scenarios:")
    print("   fit(model, x, y, workflow='quality')  # High precision")
    print("   fit(model, x, y, workflow='fast')     # Quick exploration")
    print("   fit(model, x, y, workflow='auto')     # Memory-based selection")
    print()
    print("2. Override preset settings as needed:")
    print("   fit(model, x, y, workflow='standard', n_starts=30, sampler='sobol')")
    print()
    print("3. Common customizations:")
    print("   - multistart=True, n_starts=N: Enable global search")
    print("   - sampler='sobol': Better space coverage")
    print("   - gtol/ftol/xtol: Control convergence tolerances")
    print()
    print("4. Create factory functions for your domain:")
    print("   def create_my_preset(**kwargs) -> dict: ...")
    print("   popt, pcov = fit(model, x, y, **create_my_preset())")
    print()
    print("5. For streaming, use HybridStreamingConfig presets:")
    print("   fit(model, x, y, workflow=HybridStreamingConfig.defense_strict())")


if __name__ == "__main__":
    main()

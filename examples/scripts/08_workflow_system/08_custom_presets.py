"""Custom Preset Guide: Building Domain-Specific Configurations.

This guide demonstrates the with_overrides() pattern for creating domain-specific
workflow configurations in NLSQ. The library provides generic presets (precision_high,
precision_standard, streaming_large, global_multimodal) that can be customized
for any scientific or engineering domain.

Key patterns covered:
1. Basic with_overrides() usage
2. Building on different base presets
3. Chaining multiple overrides
4. Creating reusable preset factories
5. Common parameter adjustments by use case

Run this example:
    python examples/scripts/08_workflow_system/08_custom_presets.py
"""

import jax.numpy as jnp
import numpy as np

from nlsq import fit
from nlsq.core.workflow import WORKFLOW_PRESETS, WorkflowConfig


def main():
    print("=" * 70)
    print("Custom Preset Guide: The with_overrides() Pattern")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Available Base Presets
    # =========================================================================
    print("1. Available Base Presets")
    print("-" * 60)
    print()
    print("NLSQ provides these generic presets as starting points:")
    print()

    # Group presets by category
    precision_presets = ["precision_high", "precision_standard"]
    scale_presets = ["streaming_large"]
    global_presets = ["global_multimodal", "multimodal"]
    core_presets = ["standard", "quality", "fast", "large_robust", "streaming"]

    print("  Precision presets (for numerical accuracy):")
    for name in precision_presets:
        if name in WORKFLOW_PRESETS:
            desc = WORKFLOW_PRESETS[name].get("description", "")
            print(f"    - {name}: {desc}")

    print()
    print("  Scale presets (for large datasets):")
    for name in scale_presets:
        if name in WORKFLOW_PRESETS:
            desc = WORKFLOW_PRESETS[name].get("description", "")
            print(f"    - {name}: {desc}")

    print()
    print("  Global optimization presets:")
    for name in global_presets:
        if name in WORKFLOW_PRESETS:
            desc = WORKFLOW_PRESETS[name].get("description", "")
            print(f"    - {name}: {desc}")

    print()
    print("  Core presets:")
    for name in core_presets:
        if name in WORKFLOW_PRESETS:
            desc = WORKFLOW_PRESETS[name].get("description", "")
            print(f"    - {name}: {desc}")

    # =========================================================================
    # 2. Basic with_overrides() Usage
    # =========================================================================
    print()
    print()
    print("2. Basic with_overrides() Usage")
    print("-" * 60)
    print()
    print("The with_overrides() method creates a new config with modified settings:")
    print()
    print("  # Start from a base preset")
    print("  base_config = WorkflowConfig.from_preset('precision_standard')")
    print()
    print("  # Create customized version")
    print("  custom_config = base_config.with_overrides(")
    print("      n_starts=20,          # More starting points")
    print("      sampler='sobol',      # Different sampling strategy")
    print("      gtol=1e-10,           # Tighter tolerance")
    print("  )")
    print()

    # Demonstrate
    base_config = WorkflowConfig.from_preset("precision_standard")
    custom_config = base_config.with_overrides(
        n_starts=20,
        sampler="sobol",
        gtol=1e-10,
    )

    print("  Result comparison:")
    print(f"    Base n_starts:   {base_config.n_starts}")
    print(f"    Custom n_starts: {custom_config.n_starts}")
    print(f"    Base sampler:    {base_config.sampler}")
    print(f"    Custom sampler:  {custom_config.sampler}")
    print(f"    Base gtol:       {base_config.gtol}")
    print(f"    Custom gtol:     {custom_config.gtol}")

    # =========================================================================
    # 3. Choosing the Right Base Preset
    # =========================================================================
    print()
    print()
    print("3. Choosing the Right Base Preset")
    print("-" * 60)
    print()
    print("  Use case                          -> Base preset")
    print("  " + "-" * 55)
    print("  High precision fitting            -> precision_high")
    print("  Standard scientific analysis      -> precision_standard")
    print("  Large datasets (>1M points)       -> streaming_large")
    print("  Multiple local minima expected    -> global_multimodal")
    print("  Quick exploratory fitting         -> fast")
    print("  Publication-quality results       -> quality")
    print()

    # =========================================================================
    # 4. Common Override Patterns
    # =========================================================================
    print()
    print("4. Common Override Patterns")
    print("-" * 60)

    # Pattern A: Increase multi-start coverage
    print()
    print("  Pattern A: Increase multi-start coverage")
    print("  (for models with multiple local minima)")
    print()
    config_a = WorkflowConfig.from_preset("precision_standard").with_overrides(
        n_starts=30,  # More starting points
        sampler="sobol",  # Better space coverage
    )
    print(f"    n_starts: {config_a.n_starts}")
    print(f"    sampler:  {config_a.sampler}")

    # Pattern B: Tighten tolerances
    print()
    print("  Pattern B: Tighten tolerances")
    print("  (for high-precision structural parameters)")
    print()
    config_b = WorkflowConfig.from_preset("precision_standard").with_overrides(
        gtol=1e-12,
        ftol=1e-12,
        xtol=1e-12,
    )
    print(f"    gtol: {config_b.gtol}")
    print(f"    ftol: {config_b.ftol}")
    print(f"    xtol: {config_b.xtol}")

    # Pattern C: Enable checkpointing for long runs
    print()
    print("  Pattern C: Enable checkpointing")
    print("  (for long-running fits that need fault tolerance)")
    print()
    config_c = WorkflowConfig.from_preset("streaming_large").with_overrides(
        enable_checkpoints=True,
        checkpoint_dir="./my_checkpoints",
    )
    print(f"    enable_checkpoints: {config_c.enable_checkpoints}")
    print(f"    checkpoint_dir:     {config_c.checkpoint_dir}")

    # Pattern D: Memory-constrained environment
    print()
    print("  Pattern D: Memory-constrained fitting")
    print("  (for systems with limited RAM)")
    print()
    config_d = WorkflowConfig.from_preset("streaming_large").with_overrides(
        chunk_size=5000,  # Smaller chunks
        memory_limit_gb=8.0,  # Explicit memory limit
    )
    print(f"    chunk_size:      {config_d.chunk_size}")
    print(f"    memory_limit_gb: {config_d.memory_limit_gb}")

    # =========================================================================
    # 5. Creating Reusable Preset Factories
    # =========================================================================
    print()
    print()
    print("5. Creating Reusable Preset Factories")
    print("-" * 60)
    print()
    print("  Define functions that return customized configs for your domain:")
    print()

    def create_spectroscopy_preset(high_resolution: bool = False) -> WorkflowConfig:
        """Create preset for spectroscopic peak fitting.

        Parameters
        ----------
        high_resolution : bool
            If True, use tighter tolerances for high-resolution spectra.
        """
        base = "precision_high" if high_resolution else "precision_standard"
        return WorkflowConfig.from_preset(base).with_overrides(
            n_starts=15,  # Multi-peak fits need global search
            sampler="lhs",  # Good for peak position/width spaces
        )

    def create_timeseries_preset(n_points: int) -> WorkflowConfig:
        """Create preset for time series analysis.

        Parameters
        ----------
        n_points : int
            Number of data points (affects tier selection).
        """
        if n_points > 1_000_000:
            base = "streaming_large"
        else:
            base = "precision_standard"

        return WorkflowConfig.from_preset(base).with_overrides(
            enable_multistart=True,
            n_starts=10,
        )

    def create_optimization_preset(n_params: int) -> WorkflowConfig:
        """Create preset based on parameter count.

        More parameters -> more multi-start coverage needed.
        """
        n_starts = max(10, n_params * 3)  # Scale with complexity
        return WorkflowConfig.from_preset("global_multimodal").with_overrides(
            n_starts=n_starts,
        )

    # Demonstrate factories
    print("  Examples:")
    spec_config = create_spectroscopy_preset(high_resolution=True)
    print(
        f"    Spectroscopy (high-res): gtol={spec_config.gtol}, n_starts={spec_config.n_starts}"
    )

    ts_config = create_timeseries_preset(n_points=500_000)
    print(f"    Time series (500K pts):  tier={ts_config.tier.name}")

    opt_config = create_optimization_preset(n_params=8)
    print(f"    8-param optimization:    n_starts={opt_config.n_starts}")

    # =========================================================================
    # 6. Complete Example: Domain-Specific Fitting
    # =========================================================================
    print()
    print()
    print("6. Complete Example: Domain-Specific Fitting")
    print("-" * 60)
    print()

    # Define a custom preset for a hypothetical domain
    def create_my_domain_preset() -> WorkflowConfig:
        """Create a preset for my specific application.

        This example shows all the considerations for a custom domain.
        """
        return WorkflowConfig.from_preset("precision_standard").with_overrides(
            # Tolerance settings
            gtol=1e-9,  # Tighter for accurate parameters
            ftol=1e-9,
            xtol=1e-9,
            # Multi-start settings
            enable_multistart=True,
            n_starts=15,
            sampler="sobol",  # Better coverage
            # Memory settings (if needed)
            # chunk_size=50000,
            # memory_limit_gb=16.0,
        )

    # Use the custom preset
    config = create_my_domain_preset()

    print("  Custom domain preset configuration:")
    print(f"    tier:              {config.tier.name}")
    print(f"    goal:              {config.goal.name}")
    print(f"    gtol:              {config.gtol}")
    print(f"    enable_multistart: {config.enable_multistart}")
    print(f"    n_starts:          {config.n_starts}")
    print(f"    sampler:           {config.sampler}")
    print()

    # Simple test fit
    print("  Testing with exponential decay fit...")

    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    np.random.seed(42)
    x_data = np.linspace(0, 5, 100)
    y_true = 2.5 * np.exp(-1.3 * x_data) + 0.5
    y_data = y_true + 0.1 * np.random.randn(100)

    popt, pcov = fit(
        exponential,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=([0.1, 0.1, -1.0], [10.0, 10.0, 2.0]),
        workflow_config=config,
    )

    print(f"    Fitted parameters: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
    print("    True parameters:   a=2.5000, b=1.3000, c=0.5000")

    # =========================================================================
    # 7. Summary: Key Takeaways
    # =========================================================================
    print()
    print()
    print("=" * 70)
    print("Summary: Key Takeaways")
    print("=" * 70)
    print()
    print("1. Start from an appropriate base preset:")
    print("   - precision_high/standard for numerical accuracy")
    print("   - streaming_large for big data")
    print("   - global_multimodal for complex optimization landscapes")
    print()
    print("2. Use with_overrides() to customize:")
    print("   config = WorkflowConfig.from_preset('base').with_overrides(...)")
    print()
    print("3. Common customizations:")
    print("   - n_starts: More for complex models, fewer for simple ones")
    print("   - sampler: 'sobol' for better coverage, 'lhs' for efficiency")
    print("   - tolerances: Tighter for structural params, looser for rates")
    print()
    print("4. Create reusable factory functions for your domain:")
    print("   def create_my_preset(**kwargs) -> WorkflowConfig: ...")
    print()
    print("5. The original base preset is never modified:")
    print("   with_overrides() returns a new config instance")
    print()


if __name__ == "__main__":
    main()

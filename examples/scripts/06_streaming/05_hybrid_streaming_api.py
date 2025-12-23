#!/usr/bin/env python3
"""
Demonstration of Hybrid Streaming Optimizer API Integration

This example shows how to use method='hybrid_streaming' with both
curve_fit() and curve_fit_large() functions.

The hybrid streaming optimizer provides:
- Parameter normalization for better gradient signals
- Adam warmup for robust initial convergence
- 4-layer defense strategy for warmup divergence prevention (v0.3.6+)
- Streaming Gauss-Newton for exact covariance computation
- Automatic memory management for large datasets
"""

import os

import jax.numpy as jnp
import numpy as np

from nlsq import (
    HybridStreamingConfig,
    curve_fit,
    curve_fit_large,
    get_defense_telemetry,
    reset_defense_telemetry,
)


def exponential_decay(x, a, b, c):
    """Three-parameter exponential decay model."""
    return a * jnp.exp(-b * x) + c


def main():
    if os.environ.get("NLSQ_EXAMPLES_QUICK"):
        print(
            "Quick mode: skipping hybrid streaming demo (full run requires more time)."
        )
        return

    print("=" * 70)
    print("Hybrid Streaming Optimizer API Demo")
    print("=" * 70)
    print()

    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(0, 10, 2000)
    true_params = np.array([5.0, 0.5, 1.0])
    y_true = exponential_decay(x, *true_params)
    y = y_true + np.random.normal(0, 0.1, len(x))

    # Example 1: Basic usage with curve_fit()
    print("Example 1: Basic usage with curve_fit()")
    print("-" * 70)

    result = curve_fit(
        exponential_decay,
        x,
        y,
        p0=np.array([4.0, 0.4, 0.8]),
        method="hybrid_streaming",
        verbose=1,
    )

    # Unpack result
    popt, pcov = result

    print(f"\nFitted parameters: {popt}")
    print(f"True parameters:   {true_params}")
    print(f"Parameter errors:  {np.abs(popt - true_params)}")
    print(f"\nCovariance matrix diagonal: {np.diag(pcov)}")
    print(f"Parameter std errors: {np.sqrt(np.diag(pcov))}")
    print()

    # Example 2: With parameter bounds
    print("Example 2: With parameter bounds")
    print("-" * 70)

    result = curve_fit(
        exponential_decay,
        x,
        y,
        p0=np.array([4.0, 0.4, 0.8]),
        bounds=([0, 0, 0], [10, 2, 5]),
        method="hybrid_streaming",
        verbose=0,  # Silent mode
    )

    popt, pcov = result
    print(f"Fitted parameters (bounded): {popt}")
    print(f"Within bounds: {np.all(popt >= [0, 0, 0]) and np.all(popt <= [10, 2, 5])}")
    print()

    # Example 3: Large dataset with curve_fit_large()
    print("Example 3: Large dataset with curve_fit_large()")
    print("-" * 70)

    # Generate larger dataset
    x_large = np.linspace(0, 10, 10000)
    y_large = exponential_decay(x_large, *true_params) + np.random.normal(
        0, 0.1, len(x_large)
    )

    popt, pcov = curve_fit_large(
        exponential_decay,
        x_large,
        y_large,
        p0=np.array([4.0, 0.4, 0.8]),
        method="hybrid_streaming",
        verbose=1,
    )

    print(f"\nFitted parameters (large dataset): {popt}")
    print(f"True parameters:                   {true_params}")
    print(f"Parameter errors:                  {np.abs(popt - true_params)}")
    print()

    # Example 4: Config overrides via kwargs
    print("Example 4: Custom configuration via kwargs")
    print("-" * 70)

    result = curve_fit(
        exponential_decay,
        x,
        y,
        p0=np.array([4.0, 0.4, 0.8]),
        method="hybrid_streaming",
        verbose=0,
        # HybridStreamingConfig overrides:
        warmup_iterations=300,
        normalization_strategy="p0",  # 'bounds' requires explicit bounds
        phase2_max_iterations=100,
    )

    popt, pcov = result
    print(f"Fitted parameters (custom config): {popt}")
    print(f"Result attributes available: {list(result.keys())[:10]}")
    print()

    # Example 5: Accessing full result details
    print("Example 5: Accessing full result details")
    print("-" * 70)

    result = curve_fit(
        exponential_decay,
        x,
        y,
        p0=np.array([4.0, 0.4, 0.8]),
        method="hybrid_streaming",
        verbose=0,
    )

    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Final cost: {result.get('cost', 'N/A')}")

    if "streaming_diagnostics" in result:
        diag = result["streaming_diagnostics"]
        print("\nStreaming diagnostics available:")
        print(f"  Keys: {list(diag.keys())}")
    print()

    # =========================================================================
    # Example 6: Defense Layers (v0.3.6+)
    # =========================================================================
    print("Example 6: 4-Layer Defense Strategy (v0.3.6+)")
    print("-" * 70)

    # Reset telemetry for this demo
    reset_defense_telemetry()

    # Near-optimal initial guess - defense layers should trigger
    near_optimal_p0 = true_params * np.array([1.01, 0.99, 1.005])
    print(f"Near-optimal p0: {near_optimal_p0}")

    popt, pcov = curve_fit(
        exponential_decay,
        x,
        y,
        p0=near_optimal_p0,
        method="hybrid_streaming",
        verbose=0,
    )

    telemetry = get_defense_telemetry()
    summary = telemetry.get_summary()
    rates = telemetry.get_trigger_rates()

    print(f"Fitted params: {popt}")
    print("\nDefense Layer Telemetry:")
    print(f"  Layer 1 (warm start) rate: {rates.get('layer1_warm_start_rate', 0):.1f}%")
    print(f"  Layer 2 LR modes: {summary['layer2_lr_mode_counts']}")
    print(f"  Layer 3 (cost guard) triggers: {summary['layer3_cost_guard_triggers']}")
    print(f"  Layer 4 (step clip) triggers: {summary['layer4_clip_triggers']}")
    print()

    # =========================================================================
    # Example 7: Defense Layer Presets
    # =========================================================================
    print("Example 7: Defense Layer Presets")
    print("-" * 70)

    presets = {
        "Default": HybridStreamingConfig(),
        "defense_strict()": HybridStreamingConfig.defense_strict(),
        "defense_relaxed()": HybridStreamingConfig.defense_relaxed(),
        "defense_disabled()": HybridStreamingConfig.defense_disabled(),
        "scientific_default()": HybridStreamingConfig.scientific_default(),
    }

    print(f"{'Preset':<25} {'L1':<5} {'L2':<5} {'L3':<5} {'L4':<5}")
    print("-" * 50)

    for name, config in presets.items():
        print(
            f"{name:<25} {'ON' if config.enable_warm_start_detection else 'OFF':<5} "
            f"{'ON' if config.enable_adaptive_warmup_lr else 'OFF':<5} "
            f"{'ON' if config.enable_cost_guard else 'OFF':<5} "
            f"{'ON' if config.enable_step_clipping else 'OFF':<5}"
        )

    # Use strict preset for warm start refinement
    config = HybridStreamingConfig.defense_strict()
    popt, pcov = curve_fit(
        exponential_decay,
        x,
        y,
        p0=near_optimal_p0,
        method="hybrid_streaming",
        config=config,
        verbose=0,
    )
    print(f"\nFitted with defense_strict(): {popt}")
    print()

    # =========================================================================
    # Example 8: Production Monitoring with Telemetry
    # =========================================================================
    print("Example 8: Production Monitoring with Telemetry")
    print("-" * 70)

    reset_defense_telemetry()

    # Simulate batch of fits with varying quality
    for i in range(10):
        noise = 0.01 if i < 3 else (0.3 if i < 7 else 1.0)
        p0 = true_params * (1 + np.random.uniform(-noise, noise, 3))

        curve_fit(
            exponential_decay,
            x,
            y,
            p0=p0,
            method="hybrid_streaming",
            verbose=0,
        )

    telemetry = get_defense_telemetry()
    rates = telemetry.get_trigger_rates()

    print("After 10 fits with varying starting points:")
    print(f"  Layer 1 (warm start):     {rates.get('layer1_warm_start_rate', 0):.1f}%")
    print(f"  Layer 2 (refinement LR):  {rates.get('layer2_refinement_rate', 0):.1f}%")
    print(f"  Layer 2 (careful LR):     {rates.get('layer2_careful_rate', 0):.1f}%")
    print(f"  Layer 2 (exploration LR): {rates.get('layer2_exploration_rate', 0):.1f}%")
    print(f"  Layer 3 (cost guard):     {rates.get('layer3_cost_guard_rate', 0):.1f}%")
    print(f"  Layer 4 (step clipping):  {rates.get('layer4_clip_rate', 0):.1f}%")

    print("\nPrometheus-Compatible Metrics:")
    for name, value in telemetry.export_metrics().items():
        print(f"  {name}: {value}")

    print()
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nSee also:")
    print("  - examples/notebooks/06_streaming/05_hybrid_streaming_api.ipynb")
    print("  - examples/notebooks/05_feature_demos/defense_layers_demo.ipynb")
    print("  - docs/migration/v0.3.6_defense_layers.rst")


if __name__ == "__main__":
    main()

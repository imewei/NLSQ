#!/usr/bin/env python
"""
4-Layer Defense Strategy for Adam Warmup Divergence Prevention Demo

This script demonstrates NLSQ's 4-layer defense strategy that prevents Adam optimizer
divergence during the warmup phase when initial parameters are already near optimal.

The 4 Defense Layers:
1. Warm Start Detection - Skip warmup if initial loss < 1% of data variance
2. Adaptive Learning Rate - Scale LR based on initial fit quality
3. Cost-Increase Guard - Abort if loss increases > 5% from initial
4. Step Clipping - Limit parameter update magnitude (max norm 0.1)

Version: 0.3.6+

Usage:
    python defense_layers_demo.py
"""

import os

import jax.numpy as jnp
import numpy as np

from nlsq import (
    HybridStreamingConfig,
    curve_fit,
    get_defense_telemetry,
    reset_defense_telemetry,
)


def exponential_decay(x, a, b, c):
    """Three-parameter exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def main():
    """Run the defense layers demonstration."""
    if os.environ.get("NLSQ_EXAMPLES_QUICK"):
        print("Quick mode: skipping defense layers demo (uses hybrid_streaming).")
        return

    np.random.seed(42)

    # Generate synthetic data
    true_params = np.array([5.0, 0.5, 1.0])
    x = np.linspace(0, 10, 500)  # Reduced for faster demo
    y_true = exponential_decay(x, *true_params)
    y = y_true + np.random.normal(0, 0.1, len(x))

    print("=" * 70)
    print("4-Layer Defense Strategy Demo")
    print("=" * 70)
    print(f"\nDataset: {len(x)} samples")
    print(
        f"True parameters: a={true_params[0]}, b={true_params[1]}, c={true_params[2]}"
    )

    # =========================================================================
    # Demo 1: Near-optimal starting point (Layer 1 should trigger)
    # =========================================================================
    print("\n" + "-" * 70)
    print("Demo 1: Near-optimal starting point (Layer 1 warm start detection)")
    print("-" * 70)

    reset_defense_telemetry()

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
    print("\nTelemetry:")
    print(f"  Layer 1 (warm start) triggers: {summary['layer1']['triggers']}")
    print(f"  Layer 1 rate: {rates.get('layer1_warm_start_rate', 0):.1f}%")

    if summary["layer1"]["triggers"] > 0:
        print("  -> Layer 1 detected near-optimal start and skipped warmup")

    # =========================================================================
    # Demo 2: Poor starting point (exploration LR should be used)
    # =========================================================================
    print("\n" + "-" * 70)
    print("Demo 2: Poor starting point (Layer 2 adaptive LR)")
    print("-" * 70)

    reset_defense_telemetry()

    poor_p0 = np.array([1.0, 0.1, 5.0])
    print(f"Poor p0: {poor_p0}")

    popt, pcov = curve_fit(
        exponential_decay,
        x,
        y,
        p0=poor_p0,
        method="hybrid_streaming",
        verbose=0,
    )

    telemetry = get_defense_telemetry()
    summary = telemetry.get_summary()

    print(f"Fitted params: {popt}")
    print("\nTelemetry:")
    print(f"  Layer 2 LR modes: {summary['layer2']['mode_counts']}")

    # =========================================================================
    # Demo 3: Using defense layer presets
    # =========================================================================
    print("\n" + "-" * 70)
    print("Demo 3: Defense Layer Presets")
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

    # =========================================================================
    # Demo 4: Batch monitoring with telemetry
    # =========================================================================
    print("\n" + "-" * 70)
    print("Demo 4: Production Monitoring with Telemetry")
    print("-" * 70)

    reset_defense_telemetry()

    # Simulate batch of fits with varying starting point quality
    n_fits = 5  # Reduced for faster demo
    for i in range(n_fits):
        noise = 0.01 if i < 2 else (0.3 if i < 4 else 1.0)
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

    print(f"After {n_fits} fits:")
    print(f"  Layer 1 (warm start):     {rates.get('layer1_warm_start_rate', 0):.1f}%")
    print(f"  Layer 2 (refinement LR):  {rates.get('layer2_refinement_rate', 0):.1f}%")
    print(f"  Layer 2 (careful LR):     {rates.get('layer2_careful_rate', 0):.1f}%")
    print(f"  Layer 2 (exploration LR): {rates.get('layer2_exploration_rate', 0):.1f}%")
    print(f"  Layer 3 (cost guard):     {rates.get('layer3_cost_guard_rate', 0):.1f}%")
    print(f"  Layer 4 (step clipping):  {rates.get('layer4_clip_rate', 0):.1f}%")

    # Export Prometheus-compatible metrics
    print("\nPrometheus-Compatible Metrics:")
    for name, value in telemetry.export_metrics().items():
        print(f"  {name}: {value}")

    # =========================================================================
    # Demo 5: Custom configuration
    # =========================================================================
    print("\n" + "-" * 70)
    print("Demo 5: Custom Defense Configuration")
    print("-" * 70)

    custom_config = HybridStreamingConfig(
        # Layer 1: Stricter warm start threshold
        enable_warm_start_detection=True,
        warm_start_threshold=0.005,  # 0.5% instead of 1%
        # Layer 2: More conservative learning rates
        enable_adaptive_warmup_lr=True,
        warmup_lr_refinement=1e-7,
        warmup_lr_careful=1e-6,
        # Layer 3: Tighter cost tolerance
        enable_cost_guard=True,
        cost_increase_tolerance=0.02,  # 2% instead of 5%
        # Layer 4: Smaller step limit
        enable_step_clipping=True,
        max_warmup_step_size=0.05,
    )

    print("Custom config:")
    print(f"  warm_start_threshold: {custom_config.warm_start_threshold}")
    print(f"  warmup_lr_refinement: {custom_config.warmup_lr_refinement}")
    print(f"  cost_increase_tolerance: {custom_config.cost_increase_tolerance}")
    print(f"  max_warmup_step_size: {custom_config.max_warmup_step_size}")

    reset_defense_telemetry()

    popt, pcov = curve_fit(
        exponential_decay,
        x,
        y,
        p0=np.array([4.5, 0.45, 1.1]),
        method="hybrid_streaming",
        config=custom_config,
        verbose=0,
    )

    print(f"\nFitted params with custom config: {popt}")
    print(f"Std errors: {np.sqrt(np.diag(pcov))}")

    print("\n" + "=" * 70)
    print("Defense Layers Demo Complete")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Defense layers are enabled by default (no code changes needed)")
    print("  2. Use telemetry to monitor defense behavior in production")
    print("  3. Presets available: defense_strict(), defense_relaxed(), etc.")
    print("  4. Customize individual layers for specific needs")
    print("\nSee also:")
    print("  - examples/notebooks/05_feature_demos/defense_layers_demo.ipynb")
    print("  - docs/migration/v0.3.6_defense_layers.rst")


if __name__ == "__main__":
    main()

"""
Converted from 02_checkpoint_resume.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # Example 2: Checkpoint Save and Resume
#
# This example demonstrates checkpoint save/resume functionality for recovering
# from interruptions during long-running optimizations.
#
# Features demonstrated:
# - Automatic checkpoint saving at intervals
# - Auto-detection of latest checkpoint
# - Resume from specific checkpoint path
# - Full optimizer state preservation
#
# Run this example:
#     python examples/streaming/02_checkpoint_resume.py
#
# ======================================================================

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nlsq import StreamingConfig, StreamingOptimizer


def gaussian_model(x, amp, center, width):
    """Gaussian model: y = amp * exp(-0.5 * ((x - center) / width)^2)"""
    return amp * jnp.exp(-0.5 * ((x - center) / width) ** 2)


def simulate_interruption(iteration, params, loss):
    """Callback to simulate interruption after 5 iterations"""
    if iteration == 5:
        print(f"\n  [SIMULATED INTERRUPTION at iteration {iteration}]")
        return False  # Stop optimization
    return True


def main():
    print("=" * 70)
    print("Streaming Optimizer: Checkpoint Save/Resume Example")
    print("=" * 70)
    print()

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 5000
    x_data = np.linspace(-5, 5, n_samples)
    true_amp, true_center, true_width = 2.0, 0.5, 1.5
    y_true = gaussian_model(x_data, true_amp, true_center, true_width)
    y_data = y_true + 0.05 * np.random.randn(n_samples)

    print(f"Dataset: {n_samples} samples")
    print(f"True parameters: amp={true_amp}, center={true_center}, width={true_width}")
    print()

    # Clean up old checkpoints
    checkpoint_dir = Path("checkpoints_example")
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("checkpoint_*.h5"):
            f.unlink()
        print(f"Cleaned up old checkpoints in {checkpoint_dir}")
        print()

    # Part 1: Initial training with interruption
    print("PART 1: Initial Training (will be interrupted)")
    print("=" * 70)

    config = StreamingConfig(
        batch_size=100,
        max_epochs=10,
        learning_rate=0.001,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_frequency=2,  # Save every 2 iterations (frequent for demo)
        enable_checkpoints=True,
        resume_from_checkpoint=None,  # Don't resume (start fresh)
    )

    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print(f"Checkpoint frequency: every {config.checkpoint_frequency} iterations")
    print()

    optimizer = StreamingOptimizer(config)
    p0 = np.array([1.0, 0.0, 1.0])
    print(f"Initial guess: amp={p0[0]}, center={p0[1]}, width={p0[2]}")
    print()

    print("Starting training (will interrupt after 5 iterations)...")
    result1 = optimizer.fit(
        (x_data, y_data),
        gaussian_model,
        p0,
        callback=simulate_interruption,  # Simulate interruption
        verbose=1,
    )

    print()
    print("Training interrupted!")
    print(f"Iterations completed: {optimizer.iteration}")
    print(f"Best loss so far: {result1['best_loss']:.6e}")
    print(f"Best params so far: {result1['x']}")
    print()

    # Check saved checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_iter_*.h5"))
    print(f"Checkpoints saved: {len(checkpoints)}")
    for cp in sorted(checkpoints):
        print(f"  - {cp.name}")
    print()

    # Part 2: Resume from checkpoint (auto-detect latest)
    print("PART 2: Resume from Checkpoint (auto-detect)")
    print("=" * 70)

    config_resume = StreamingConfig(
        batch_size=100,
        max_epochs=10,
        learning_rate=0.001,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_frequency=2,
        enable_checkpoints=True,
        resume_from_checkpoint=True,  # Auto-detect latest checkpoint
    )

    print("Resuming with auto-detection of latest checkpoint...")
    print()

    optimizer2 = StreamingOptimizer(config_resume)
    result2 = optimizer2.fit(
        (x_data, y_data),
        gaussian_model,
        p0,  # Still provide p0 (used if checkpoint load fails)
        verbose=1,
    )

    print()
    print("Training resumed and completed!")
    print()

    # Part 3: Resume from specific checkpoint path
    print("PART 3: Resume from Specific Checkpoint")
    print("=" * 70)

    # Find a specific checkpoint (e.g., iteration 4)
    specific_checkpoint = checkpoint_dir / "checkpoint_iter_4.h5"
    optimizer3 = None
    if specific_checkpoint.exists():
        print(f"Resuming from specific checkpoint: {specific_checkpoint.name}")
        print()

        config_specific = StreamingConfig(
            batch_size=100,
            max_epochs=10,
            learning_rate=0.001,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_frequency=2,
            enable_checkpoints=True,
            resume_from_checkpoint=str(specific_checkpoint),  # Specific path
        )

        optimizer3 = StreamingOptimizer(config_specific)
        result3 = optimizer3.fit(
            (x_data, y_data),
            gaussian_model,
            p0,
            verbose=1,
        )

        print()
        print(
            f"Resumed from iteration 4, completed at iteration {optimizer3.iteration}"
        )
        print()

    # Display final results
    print("FINAL RESULTS")
    print("=" * 70)
    best_params = result2["x"]
    print("Best parameters:")
    print(f"  amp    = {best_params[0]:.6f} (true: {true_amp})")
    print(f"  center = {best_params[1]:.6f} (true: {true_center})")
    print(f"  width  = {best_params[2]:.6f} (true: {true_width})")
    print(f"  Best loss = {result2['best_loss']:.6e}")
    print()

    # Checkpoint diagnostics
    diag = result2["streaming_diagnostics"]
    if diag["checkpoint_info"]:
        cp_info = diag["checkpoint_info"]
        print("Final Checkpoint:")
        print(f"  Path: {cp_info['path']}")
        print(f"  Saved at: {cp_info['saved_at']}")
        print(f"  Batch index: {cp_info['batch_idx']}")
        print()

    print("=" * 70)
    print("Example complete!")
    print()
    print("Key takeaways:")
    print("  - Checkpoints save full optimizer state (params, momentum, etc.)")
    print("  - resume_from_checkpoint=True auto-detects latest checkpoint")
    print("  - resume_from_checkpoint='path' loads specific checkpoint")
    print("  - Seamless resume from any interruption point")
    print("  - No duplicate batch processing on resume")
    print(f"\nCheckpoints saved in: {checkpoint_dir.absolute()}")

    # Cleanup checkpoint worker threads to prevent memory corruption on exit
    for opt in [optimizer, optimizer2, optimizer3]:
        if opt is not None and hasattr(opt, "_shutdown_checkpoint_worker"):
            opt._shutdown_checkpoint_worker()


if __name__ == "__main__":
    main()

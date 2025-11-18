"""
Converted from callbacks_demo.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # Demo: Progress Callbacks for Optimization Monitoring
#
# This example demonstrates how to use NLSQ's callback system to monitor
# optimization progress with progress bars, logging, and early stopping.
#
# ======================================================================

import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit
from nlsq.callbacks import (
    CallbackBase,
    CallbackChain,
    EarlyStopping,
    IterationLogger,
    ProgressBar,
)


def exponential_decay(x, amplitude, rate, offset):
    """Exponential decay model: amplitude * exp(-rate * x) + offset."""
    return amplitude * jnp.exp(-rate * x) + offset


def generate_sample_data():
    """Generate sample exponential decay data with noise."""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    # True parameters: amplitude=100, rate=0.5, offset=10
    y_true = 100 * np.exp(-0.5 * x) + 10
    y = y_true + np.random.normal(0, 3, size=len(x))
    return x, y, y_true


# ======================================================================
# ## Example 1: Simple Progress Bar
#
# ======================================================================


def example1_progress_bar():
    """Show optimization progress with tqdm progress bar."""
    print("\n" + "=" * 70)
    print("Example 1: Progress Bar")
    print("=" * 70)
    print("\nMonitoring optimization with a progress bar...\n")

    x, y, y_true = generate_sample_data()

    # Create progress bar callback
    callback = ProgressBar(max_nfev=50, desc="Fitting exponential")

    # Fit with progress bar
    popt, pcov = curve_fit(
        exponential_decay, x, y, p0=[80, 0.4, 5], callback=callback, max_nfev=50
    )

    callback.close()

    print("\n✓ Progress bar completed!")
    print(
        f"Fitted parameters: amplitude={popt[0]:.2f}, rate={popt[1]:.3f}, offset={popt[2]:.2f}"
    )


# ======================================================================
# ## Example 2: Iteration Logging
#
# ======================================================================


def example2_iteration_logging():
    """Log optimization progress to file."""
    print("\n" + "=" * 70)
    print("Example 2: Iteration Logging")
    print("=" * 70)
    print("\nLogging optimization details to file...\n")

    x, y, y_true = generate_sample_data()

    # Create logging callback
    callback = IterationLogger(
        filename="optimization.log",
        mode="w",
        log_params=True,  # Include parameter values
    )

    # Fit with logging
    popt, pcov = curve_fit(
        exponential_decay, x, y, p0=[80, 0.4, 5], callback=callback, max_nfev=50
    )

    callback.close()

    print("✓ Log written to 'optimization.log'")
    print(
        f"Fitted parameters: amplitude={popt[0]:.2f}, rate={popt[1]:.3f}, offset={popt[2]:.2f}"
    )
    print("\nFirst few lines of log:\n")
    with open("optimization.log") as f:
        lines = f.readlines()
        print("".join(lines[:10]))  # Show first 10 lines


# ======================================================================
# ## Example 3: Early Stopping
#
# ======================================================================


def example3_early_stopping():
    """Stop optimization early if no improvement."""
    print("\n" + "=" * 70)
    print("Example 3: Early Stopping")
    print("=" * 70)
    print("\nUsing early stopping to prevent wasted iterations...\n")

    x, y, y_true = generate_sample_data()

    # Create early stopping callback
    callback = EarlyStopping(
        patience=10,  # Stop after 10 iterations without improvement
        min_delta=1e-6,  # Minimum improvement threshold
        verbose=True,
    )

    # Fit with early stopping
    popt, pcov = curve_fit(
        exponential_decay,
        x,
        y,
        p0=[80, 0.4, 5],
        callback=callback,
        max_nfev=1000,  # Set high, early stopping will prevent wasted iterations
    )

    print("\n✓ Early stopping completed!")
    print(
        f"Fitted parameters: amplitude={popt[0]:.2f}, rate={popt[1]:.3f}, offset={popt[2]:.2f}"
    )


# ======================================================================
# ## Example 4: Combining Multiple Callbacks
#
# ======================================================================


def example4_callback_chain():
    """Combine progress bar, logging, and early stopping."""
    print("\n" + "=" * 70)
    print("Example 4: Callback Chain")
    print("=" * 70)
    print("\nCombining multiple callbacks together...\n")

    x, y, y_true = generate_sample_data()

    # Combine multiple callbacks
    callback = CallbackChain(
        ProgressBar(max_nfev=50, desc="Optimizing"),
        IterationLogger("combined.log", log_params=False),
        EarlyStopping(patience=10, verbose=False),  # Silent for cleaner demo
    )

    # Fit with callback chain
    popt, pcov = curve_fit(
        exponential_decay, x, y, p0=[80, 0.4, 5], callback=callback, max_nfev=50
    )

    callback.close()

    print("\n✓ All callbacks completed!")
    print(
        f"Fitted parameters: amplitude={popt[0]:.2f}, rate={popt[1]:.3f}, offset={popt[2]:.2f}"
    )
    print("Check 'combined.log' for detailed iteration history.")


# ======================================================================
# ## Example 5: Custom Callback
#
# ======================================================================


class BestParameterTracker(CallbackBase):
    """Custom callback to track best parameters seen so far."""

    def __init__(self):
        self.best_cost = np.inf
        self.best_params = None
        self.history = []

    def __call__(self, iteration, cost, params, info):
        """Track best parameters."""
        self.history.append({"iter": iteration, "cost": cost})

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params.copy()
            print(f"  → New best at iter {iteration}: cost={cost:.6f}")

    def get_best(self):
        """Return best parameters found."""
        return self.best_params, self.best_cost


def example5_custom_callback():
    """Create and use a custom callback."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Callback")
    print("=" * 70)
    print("\nTracking best parameters with custom callback...\n")

    x, y, y_true = generate_sample_data()

    # Create custom callback
    tracker = BestParameterTracker()

    # Fit with custom callback
    popt, pcov = curve_fit(
        exponential_decay, x, y, p0=[80, 0.4, 5], callback=tracker, max_nfev=50
    )

    best_params, best_cost = tracker.get_best()
    print(
        f"\n✓ Best parameters: amplitude={best_params[0]:.2f}, rate={best_params[1]:.3f}, offset={best_params[2]:.2f}"
    )
    print(f"✓ Best cost: {best_cost:.6f}")
    print(f"✓ Final fit parameters match best: {np.allclose(popt, best_params)}")


# ======================================================================
# ## Main Demo
#
# ======================================================================


def main():
    """Run all callback examples."""
    print("\n" + "=" * 70)
    print("NLSQ Callbacks Demo")
    print("=" * 70)
    print("\nDemonstrating callback system for optimization monitoring.\n")

    # Run examples
    example1_progress_bar()
    example2_iteration_logging()
    example3_early_stopping()
    example4_callback_chain()
    example5_custom_callback()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • ProgressBar: Real-time optimization monitoring with tqdm")
    print("  • IterationLogger: Detailed logs for analysis and debugging")
    print("  • EarlyStopping: Avoid wasted iterations when optimization stalls")
    print("  • CallbackChain: Combine multiple callbacks seamlessly")
    print("  • Custom Callbacks: Easy to extend by subclassing CallbackBase")
    print("\nUsage:")
    print("  from nlsq import curve_fit")
    print("  from nlsq.callbacks import ProgressBar")
    print("  popt, pcov = curve_fit(f, x, y, callback=ProgressBar())")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

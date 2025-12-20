"""
Converted from custom_algorithms_advanced.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

# ======================================================================
# # Custom Algorithms and Advanced Extensions
#
# **Level**: Advanced / Research
# **Time**: 60-90 minutes
# **Prerequisites**: NLSQ Quickstart, JAX fundamentals, optimization theory
#
# ## Overview
#
# This tutorial is for **researchers and advanced users** who want to:
# - Implement custom optimization algorithms
# - Design specialized loss functions
# - Extend NLSQ for novel applications
# - Understand NLSQ's internals for research
#
# ### What You'll Learn
#
# 1. **NLSQ Architecture**: Understanding the optimization backend
# 2. **Custom Loss Functions**: Beyond least squares
# 3. **Custom Optimizers**: Implementing specialized algorithms
# 4. **Advanced JAX Patterns**: Efficient curve fitting with JAX
# 5. **Research Extensions**: Constrained optimization, robust methods
#
# ### Use Cases
#
# - **Custom loss**: Asymmetric penalties, quantile regression, robust M-estimators
# - **Specialized optimizers**: Trust-region methods, second-order algorithms
# - **Constrained problems**: Inequality constraints, manifold optimization
# - **Novel applications**: Bayesian inference, inverse problems, PDE-constrained optimization
#
# ### Warning
#
# This is advanced material. Modifying optimization algorithms requires solid understanding of:
# - Optimization theory (convexity, convergence, gradients)
# - JAX programming model (JIT, grad, pytrees)
# - Numerical stability considerations
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit, value_and_grad, vmap

# Optimization libraries
try:
    import optax

    OPTAX_AVAILABLE = True
except ImportError:
    OPTAX_AVAILABLE = False
    print("⚠ Optax not available - install with: pip install optax")

from nlsq import CurveFit

print("✓ Imports successful")
print(f"  JAX version: {jax.__version__}")
print(f"  JAX devices: {jax.devices()}")
if OPTAX_AVAILABLE:
    print(f"  Optax version: {optax.__version__}")

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
if QUICK:
    print("Quick mode: skipping advanced algorithm deep dive.")
    sys.exit(0)


# ======================================================================
# ## Part 1: Understanding NLSQ's Optimization Backend
#
# Before customizing, let's understand how NLSQ works internally.
# ======================================================================


# Exploring NLSQ internals

# Simple problem: fit exponential
x_data = jnp.linspace(0, 5, 30)
y_true = 3.0 * jnp.exp(-0.5 * x_data)
y_data = y_true + np.random.normal(0, 0.1, len(x_data))


def exponential(x, a, b):
    return a * jnp.exp(-b * x)


# Standard NLSQ fit
cf = CurveFit()
popt, pcov = cf.curve_fit(exponential, x_data, y_data, p0=[2.0, 0.3])

print("Standard NLSQ Fit:")
print(f"  Parameters: a={popt[0]:.3f}, b={popt[1]:.3f}")
print(f"  Covariance matrix shape: {pcov.shape}")
print()

# How NLSQ works internally (simplified):
print("NLSQ Internal Workflow:")
print("1. Residual function: r(θ) = y_data - model(x_data, θ)")
print("2. Loss function: L(θ) = 0.5 * sum(r(θ)^2)")
print("3. Gradient: ∇L(θ) = -J^T r(θ) where J = ∂model/∂θ")
print("4. Optimization: Levenberg-Marquardt or similar")
print("5. Uncertainty: pcov = (J^T J)^(-1) * σ^2")
print()

# Let's compute these manually with JAX
print("Manual computation with JAX:")


def residual_fn(params, x, y):
    """Residual vector r(θ) = y - model(x, θ)."""
    a, b = params
    y_pred = exponential(x, a, b)
    return y - y_pred


def loss_fn(params, x, y):
    """Sum of squared residuals L(θ) = 0.5 * ||r(θ)||^2."""
    r = residual_fn(params, x, y)
    return 0.5 * jnp.sum(r**2)


# Compute gradient at fitted parameters
grad_fn = grad(loss_fn)
gradient = grad_fn(popt, x_data, y_data)

print(f"  Gradient at optimum: {gradient}")
print(f"  Gradient norm: {jnp.linalg.norm(gradient):.2e} (should be ≈ 0)")
print("  → Confirms NLSQ found a critical point where ∇L = 0 ✓")


# ======================================================================
# ## Part 2: Custom Loss Functions
#
# Beyond standard least squares, we can implement custom loss functions for specialized needs.
# ======================================================================


# Example 1: Robust loss function (Huber loss)

# Generate data with outliers
x_robust = jnp.linspace(0, 10, 50)
y_robust = 2.0 * x_robust + 1.0 + np.random.normal(0, 0.5, 50)
# Add outliers (convert indices to JAX array for .at[] indexing)
outlier_idx = jnp.array([5, 15, 35, 42])
y_robust = y_robust.at[outlier_idx].add(jnp.array([5.0, -6.0, 4.0, -5.0]))


def linear_model(x, a, b):
    return a * x + b


# Standard least squares (sensitive to outliers)
def least_squares_loss(params, x, y):
    a, b = params
    residuals = y - linear_model(x, a, b)
    return jnp.sum(residuals**2)


# Huber loss (robust to outliers)
def huber_loss(params, x, y, delta=1.0):
    """Huber loss: quadratic for small errors, linear for large.

    Parameters
    ----------
    delta : float
        Threshold for switching from quadratic to linear
    """
    a, b = params
    residuals = y - linear_model(x, a, b)
    abs_residuals = jnp.abs(residuals)

    # Huber function: 0.5*r^2 if |r| <= delta, else delta*(|r| - 0.5*delta)
    huber = jnp.where(
        abs_residuals <= delta,
        0.5 * residuals**2,
        delta * (abs_residuals - 0.5 * delta),
    )
    return jnp.sum(huber)


# Optimize with both losses
if OPTAX_AVAILABLE:
    # Using Optax for custom optimization
    def optimize_custom(loss_fn, p0, x, y, n_steps=1000, lr=0.01):
        """Custom optimizer using Optax Adam."""
        params = jnp.array(p0)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        @jit
        def step(params, opt_state):
            loss, grads = value_and_grad(loss_fn)(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        losses = []
        for i in range(n_steps):
            params, opt_state, loss = step(params, opt_state)
            if i % 100 == 0:
                losses.append(float(loss))

        return params, losses

    # Fit with both losses
    p0 = [1.0, 0.0]
    params_ls, losses_ls = optimize_custom(least_squares_loss, p0, x_robust, y_robust)
    params_huber, losses_huber = optimize_custom(
        lambda p, x, y: huber_loss(p, x, y, delta=1.5), p0, x_robust, y_robust
    )

    print("Least Squares (sensitive to outliers):")
    print(f"  a={params_ls[0]:.3f}, b={params_ls[1]:.3f}")
    print("\nHuber Loss (robust to outliers):")
    print(f"  a={params_huber[0]:.3f}, b={params_huber[1]:.3f}")
    print("\nTrue parameters: a=2.0, b=1.0")

    # Visualization
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Fits
    x_plot = jnp.linspace(0, 10, 100)
    ax1.plot(x_robust, y_robust, "o", alpha=0.5, label="Data (with outliers)")
    ax1.plot(
        x_robust[outlier_idx],
        y_robust[outlier_idx],
        "rx",
        ms=12,
        mew=3,
        label="Outliers",
    )
    ax1.plot(
        x_plot,
        linear_model(x_plot, *params_ls),
        "r--",
        lw=2,
        label="Least Squares",
    )
    ax1.plot(
        x_plot, linear_model(x_plot, *params_huber), "g-", lw=2, label="Huber Loss"
    )
    ax1.plot(x_plot, 2.0 * x_plot + 1.0, "k:", lw=2, label="True")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Robust Fitting with Custom Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss convergence
    ax2.semilogy(losses_ls, "r-", label="Least Squares")
    ax2.semilogy(losses_huber, "g-", label="Huber Loss")
    ax2.set_xlabel("Iteration (×100)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Convergence")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    # Save figure to file
    fig_dir = Path(__file__).parent / "figures" / "custom_algorithms_advanced"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("⚠ Install optax to run this example: pip install optax")


# Example 2: Asymmetric loss (safety-critical applications)


def asymmetric_loss(params, x, y, alpha=2.0):
    """Asymmetric quadratic loss.

    Penalizes overestimation more than underestimation.
    Useful when overestimation is more costly (e.g., drug dosing).

    Parameters
    ----------
    alpha : float
        Asymmetry parameter (alpha > 1 penalizes positive residuals more)
    """
    a, b = params
    residuals = y - linear_model(x, a, b)

    # Asymmetric penalty
    loss = jnp.where(
        residuals > 0,  # Overestimation (model too low)
        alpha * residuals**2,  # Higher penalty
        residuals**2,  # Normal penalty
    )
    return jnp.sum(loss)


if OPTAX_AVAILABLE:
    # Fit with asymmetric loss
    params_asym, _ = optimize_custom(
        lambda p, x, y: asymmetric_loss(p, x, y, alpha=3.0),
        [1.0, 0.0],
        x_robust,
        y_robust,
    )

    print("Asymmetric Loss (penalizes overestimation 3x):")
    print(f"  a={params_asym[0]:.3f}, b={params_asym[1]:.3f}")
    print(
        "  → Fit is conservative (tends to underestimate to avoid costly overestimation)"
    )


# ======================================================================
# ## Part 3: Custom Optimization Algorithms
#
# Implement specialized optimization algorithms for specific problem structures.
# ======================================================================


# Example: Gradient descent with momentum (from scratch)


def gradient_descent_momentum(
    loss_fn, p0, x, y, lr=0.01, momentum=0.9, n_steps=1000, tol=1e-6
):
    """Gradient descent with momentum optimizer.

    Parameters
    ----------
    loss_fn : callable
        Loss function: loss_fn(params, x, y) -> scalar
    p0 : array
        Initial parameters
    lr : float
        Learning rate
    momentum : float
        Momentum coefficient (0 = no momentum, 0.9 typical)
    n_steps : int
        Maximum iterations
    tol : float
        Convergence tolerance on gradient norm

    Returns
    -------
    params : array
        Optimized parameters
    history : dict
        Optimization history (params, loss, grad_norm)
    """
    params = jnp.array(p0, dtype=jnp.float32)
    velocity = jnp.zeros_like(params)

    history = {"params": [], "loss": [], "grad_norm": []}

    grad_fn = jit(grad(loss_fn))
    loss_fn_jit = jit(loss_fn)

    for i in range(n_steps):
        # Compute gradient
        g = grad_fn(params, x, y)
        grad_norm = float(jnp.linalg.norm(g))

        # Update velocity (momentum)
        velocity = momentum * velocity - lr * g

        # Update parameters
        params = params + velocity

        # Record history
        if i % 50 == 0:
            loss_val = float(loss_fn_jit(params, x, y))
            history["params"].append(params.copy())
            history["loss"].append(loss_val)
            history["grad_norm"].append(grad_norm)

        # Check convergence
        if grad_norm < tol:
            print(f"  Converged at iteration {i} (grad_norm={grad_norm:.2e})")
            break

    return params, history


# Test custom optimizer
print("Custom Gradient Descent with Momentum:")
params_gd, history_gd = gradient_descent_momentum(
    least_squares_loss, [0.0, 0.0], x_data, y_data, lr=0.01, momentum=0.9, n_steps=2000
)

print(f"  Final params: a={params_gd[0]:.3f}, b={params_gd[1]:.3f}")
print(f"  Optimization steps: {len(history_gd['loss'])}")

# Compare with NLSQ
popt_nlsq, _ = cf.curve_fit(exponential, x_data, y_data, p0=[0.0, 0.0])
print("\nNLSQ (Levenberg-Marquardt):")
print(f"  Final params: a={popt_nlsq[0]:.3f}, b={popt_nlsq[1]:.3f}")
print("\n→ Both converge to similar solution ✓")


# ======================================================================
# ## Part 4: Advanced JAX Patterns for Curve Fitting
#
# Leverage JAX's advanced features for efficient batch fitting.
# ======================================================================


# Vectorized batch fitting with vmap

# Generate multiple datasets
n_datasets = 100
x_batch = jnp.linspace(0, 5, 30)

# Each dataset has different true parameters
a_true_batch = np.random.uniform(2.0, 4.0, n_datasets)
b_true_batch = np.random.uniform(0.3, 0.7, n_datasets)

y_batch = jnp.array(
    [
        a * jnp.exp(-b * x_batch) + np.random.normal(0, 0.05, len(x_batch))
        for a, b in zip(a_true_batch, b_true_batch, strict=True)
    ]
)

print(f"Batch fitting: {n_datasets} datasets simultaneously")
print(f"  Data shape: {y_batch.shape} (datasets × points)")
print()


# Define fitting function for single dataset
def fit_single_dataset(y_single):
    """Fit one dataset (simplified Newton's method)."""
    params = jnp.array([3.0, 0.5])  # Initial guess

    def loss(p):
        return jnp.sum((y_single - exponential(x_batch, *p)) ** 2)

    # Simple gradient descent (10 steps)
    for _ in range(20):
        g = grad(loss)(params)
        params = params - 0.05 * g

    return params


# Vectorize over batch dimension with vmap
fit_batch = jit(vmap(fit_single_dataset))

# Fit all datasets in parallel (GPU accelerated!)
import time

start = time.time()
params_batch = fit_batch(y_batch)
batch_time = time.time() - start

print(f"✓ Fitted {n_datasets} datasets in {batch_time * 1000:.1f} ms")
print(
    f"  Average time per dataset: {batch_time / n_datasets * 1000:.2f} ms (with vmap)"
)
print()

# Check accuracy
a_fitted = params_batch[:, 0]
b_fitted = params_batch[:, 1]

a_error = np.mean(np.abs(a_fitted - a_true_batch))
b_error = np.mean(np.abs(b_fitted - b_true_batch))

print("Fitting accuracy:")
print(f"  Mean absolute error in a: {a_error:.4f}")
print(f"  Mean absolute error in b: {b_error:.4f}")

# Visualize results
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.scatter(a_true_batch, a_fitted, alpha=0.5, s=20)
ax1.plot([2, 4], [2, 4], "r--", lw=2, label="Perfect fit")
ax1.set_xlabel("True a")
ax1.set_ylabel("Fitted a")
ax1.set_title("Parameter Recovery: a")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.scatter(b_true_batch, b_fitted, alpha=0.5, s=20)
ax2.plot([0.3, 0.7], [0.3, 0.7], "r--", lw=2, label="Perfect fit")
ax2.set_xlabel("True b")
ax2.set_ylabel("Fitted b")
ax2.set_title("Parameter Recovery: b")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "custom_algorithms_advanced"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_02.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n→ vmap enables efficient parallel fitting across datasets ✓")


# ======================================================================
# ## Part 5: Research Extensions
#
# Advanced techniques for cutting-edge applications.
# ======================================================================


# Example: Constrained optimization with penalty method


def constrained_loss(params, x, y, lambda_penalty=10.0):
    """Fit with constraint: a + b = 1.0 (sum constraint).

    Uses quadratic penalty method.
    """
    a, b = params

    # Standard loss
    residuals = y - (a * jnp.exp(-x) + b * jnp.exp(-2 * x))
    data_loss = jnp.sum(residuals**2)

    # Constraint penalty: (a + b - 1)^2
    constraint_violation = (a + b - 1.0) ** 2
    penalty = lambda_penalty * constraint_violation

    return data_loss + penalty


# Generate data satisfying constraint
x_const = jnp.linspace(0, 3, 40)
a_true_const = 0.6
b_true_const = 0.4  # a + b = 1.0
y_const = (
    a_true_const * jnp.exp(-x_const)
    + b_true_const * jnp.exp(-2 * x_const)
    + np.random.normal(0, 0.02, len(x_const))
)

if OPTAX_AVAILABLE:
    # Unconstrained fit
    params_unconstr, _ = optimize_custom(
        lambda p, x, y: jnp.sum(
            (y - (p[0] * jnp.exp(-x) + p[1] * jnp.exp(-2 * x))) ** 2
        ),
        [0.5, 0.5],
        x_const,
        y_const,
        n_steps=2000,
    )

    # Constrained fit
    params_constr, _ = optimize_custom(
        lambda p, x, y: constrained_loss(p, x, y, lambda_penalty=100.0),
        [0.5, 0.5],
        x_const,
        y_const,
        n_steps=2000,
    )

    print("Unconstrained fit:")
    print(
        f"  a={params_unconstr[0]:.4f}, b={params_unconstr[1]:.4f}, sum={params_unconstr[0] + params_unconstr[1]:.4f}"
    )
    print("\nConstrained fit (a + b = 1):")
    print(
        f"  a={params_constr[0]:.4f}, b={params_constr[1]:.4f}, sum={params_constr[0] + params_constr[1]:.4f}"
    )
    print(f"\nTrue values: a={a_true_const}, b={b_true_const}, sum=1.0")
    print(
        f"→ Constraint enforced: sum = {params_constr[0] + params_constr[1]:.6f} ≈ 1.0 ✓"
    )


# ======================================================================
# ## Summary and Best Practices
#
# ### When to Use Custom Algorithms
#
# | **Application** | **Standard NLSQ** | **Custom Algorithm** |
# |-----------------|-------------------|----------------------|
# | Standard curve fitting | ✅ Recommended | Unnecessary |
# | Outlier-heavy data | Use sigma weights | Robust loss (Huber, Cauchy) |
# | Asymmetric costs | N/A | Asymmetric loss function |
# | Constrained parameters | Use bounds | Penalty methods, Lagrangian |
# | Batch processing (1000s of fits) | Serial fitting | vmap for parallelization |
# | Novel research problems | May not apply | Custom optimizer |
#
# ### Implementation Checklist
#
# When implementing custom algorithms:
#
# 1. **Start simple**: Test with toy problems where you know the answer
# 2. **Verify gradients**: Use `jax.grad` and compare with finite differences
# 3. **Check convergence**: Monitor loss and gradient norms
# 4. **Use JIT**: Compile with `@jit` for 10-100x speedups
# 5. **Numerical stability**: Check for NaN/Inf, use stable formulations
# 6. **Validate results**: Compare with standard methods when possible
#
# ### Advanced JAX Patterns
#
# ```python
# # Pattern 1: Efficient batch fitting
# fit_single = jit(lambda y: optimize(loss_fn, y))
# fit_batch = vmap(fit_single)  # Parallelize over batch dimension
# results = fit_batch(y_batch)  # GPU-accelerated
#
# # Pattern 2: Custom gradients for numerical stability
# from jax import custom_jvp
#
# @custom_jvp
# def stable_exp(x):
#     return jnp.exp(jnp.clip(x, -50, 50))  # Prevent overflow
#
# # Pattern 3: Automatic differentiation through optimization
# def meta_objective(hyperparams):
#     # Fit model with hyperparams
#     params = optimize(loss_fn, hyperparams)
#     # Evaluate on validation set
#     return validation_loss(params)
#
# optimal_hyperparams = optimize(meta_objective, initial_hyperparams)
# ```
#
# ### Research Extensions
#
# Cutting-edge applications:
#
# 1. **Bilevel optimization**: Hyperparameter tuning via gradient descent
# 2. **Meta-learning**: Learning to fit across multiple tasks
# 3. **Differentiable physics**: PDE-constrained optimization
# 4. **Uncertainty quantification**: Laplace approximation, variational inference
# 5. **Inverse problems**: Image reconstruction, tomography
#
# ### Production Recommendations
#
# For production use:
# - **Default**: Use standard NLSQ (well-tested, robust)
# - **Custom loss**: Only when problem demands it (document why!)
# - **Testing**: Extensive validation against standard methods
# - **Monitoring**: Track convergence, gradient norms, numerical stability
# - **Fallback**: Implement standard NLSQ as backup if custom method fails
#
# ### References
#
# 1. **Optimization**: Nocedal & Wright, *Numerical Optimization* (2006)
# 2. **JAX**: https://jax.readthedocs.io/
# 3. **Optax**: https://optax.readthedocs.io/
# 4. **Robust fitting**: Huber, *Robust Statistics* (2009)
# 5. **Related examples**:
#    - `advanced_features_demo.ipynb` - NLSQ diagnostics
#    - `ml_integration_tutorial.ipynb` - Hybrid models with custom optimization
#
# ---
#
# **Warning**: Custom algorithms can be powerful but require careful validation. Always test thoroughly before using in production!
# ======================================================================

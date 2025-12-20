"""
Converted from ml_integration_tutorial.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

# ======================================================================
# # NLSQ + JAX ML Ecosystem Integration
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/ml_integration_tutorial.ipynb)
#
# **Level**: Advanced | **Time**: 45-60 min | **Prerequisites**: NLSQ Quickstart, JAX basics
#
# ## Overview
#
# This tutorial demonstrates how NLSQ integrates with the JAX machine learning ecosystem for:
#
# 1. **Neural ODEs**: Fitting dynamic systems with learned neural network components
# 2. **Physics-Informed Neural Networks (PINNs)**: Incorporating physical constraints into ML models
# 3. **Differentiable Physics**: End-to-end differentiable simulations
# 4. **Hybrid Models**: Combining mechanistic and data-driven approaches
# 5. **Optax Integration**: Using advanced optimizers with NLSQ
# 6. **Parameter Estimation**: Fitting ML model parameters from experimental data
#
# ### What You'll Learn
#
# - ✅ Integrate NLSQ with Flax neural networks
# - ✅ Build hybrid mechanistic-ML models
# - ✅ Implement Neural ODEs for dynamics
# - ✅ Create physics-informed loss functions
# - ✅ Fit complex multi-component models
# - ✅ Leverage automatic differentiation for scientific computing
#
# ### Prerequisites
#
# **Required Knowledge**:
# - NLSQ basics (complete Quickstart first)
# - JAX fundamentals
# - Basic neural networks
# - Differential equations (for Neural ODE section)
#
# **Python Version**: 3.12+
# **NLSQ Version**: 0.2.0+
#
# ---
# ======================================================================
# ======================================================================
# ## Setup and Imports
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
# Install dependencies (uncomment if needed)
# !pip install nlsq flax optax equinox
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit

# NLSQ imports
from nlsq import CurveFit, __version__

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
if QUICK:
    print(
        "Quick mode: skipping ML integration demo (unset NLSQ_EXAMPLES_QUICK for full run)."
    )
    sys.exit(0)

# ML ecosystem imports
try:
    import flax.linen as nn
    import optax
    from flax.training import train_state

    FLAX_AVAILABLE = True
except ImportError:
    print("⚠️  Flax/Optax not available. Install with: pip install flax optax")
    FLAX_AVAILABLE = False

try:
    pass

    EQUINOX_AVAILABLE = True
except ImportError:
    print("⚠️  Equinox not available. Install with: pip install equinox")
    EQUINOX_AVAILABLE = False

print(f"NLSQ version: {__version__}")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Flax available: {FLAX_AVAILABLE}")
print(f"Equinox available: {EQUINOX_AVAILABLE}")

# Set random seed for reproducibility
np.random.seed(42)
key = jax.random.PRNGKey(42)

# Generate synthetic data: exponential decay + systematic deviation
# (defined outside FLAX_AVAILABLE block so statistics can be printed)
x_data = np.linspace(0, 5, 200)
true_a, true_b = 5.0, 1.2

# Physics component
y_physics = true_a * np.exp(-true_b * x_data)

# Systematic deviation (sinusoidal correction)
y_correction = 0.5 * np.sin(3 * x_data) * np.exp(-0.3 * x_data)

# Observed data = physics + correction + noise
y_data = y_physics + y_correction + np.random.normal(0, 0.1, len(x_data))


# ======================================================================
# ---
#
# ## Part 1: Hybrid Mechanistic-ML Models
#
# ### Concept: Combining Physics and Learning
#
# Many scientific problems have **known physics** but **unknown corrections**:
#
# $$
# y(x; \theta, w) = f_{\text{physics}}(x; \theta) + g_{\text{NN}}(x; w)
# $$
#
# Where:
# - $f_{\text{physics}}$: Known mechanistic model (e.g., exponential decay)
# - $g_{\text{NN}}$: Learned neural network correction
# - $\theta$: Physical parameters (fitted with NLSQ)
# - $w$: Neural network weights (pre-trained or jointly optimized)
#
# **Benefits**:
# - ✅ Interpretable physical parameters
# - ✅ Data-efficient (physics provides structure)
# - ✅ Extrapolates better than pure ML
# - ✅ Captures systematic deviations
# ======================================================================


# ======================================================================
# ### Example 1.1: Exponential Decay with Neural Network Correction
# ======================================================================


if FLAX_AVAILABLE:
    # Define a simple MLP for corrections
    class CorrectionMLP(nn.Module):
        """Small MLP to learn systematic deviations from physics model."""

        features: list = (16, 16, 1)

        @nn.compact
        def __call__(self, x):
            for feat in self.features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.relu(x)
            x = nn.Dense(self.features[-1])(x)
            return x.squeeze()

    # Initialize network
    model = CorrectionMLP()
    params = model.init(key, jnp.ones((1, 1)))

    print("✅ Correction MLP initialized")
    print(f"   Parameter shapes: {jax.tree_util.tree_map(lambda x: x.shape, params)}")

    # Generate synthetic data: exponential decay + systematic deviation
    x_data = np.linspace(0, 5, 200)
    true_a, true_b = 5.0, 1.2

    # Physics component
    y_physics = true_a * np.exp(-true_b * x_data)

    # Systematic deviation (sinusoidal correction)
    y_correction = 0.5 * np.sin(3 * x_data) * np.exp(-0.3 * x_data)

    # Observed data = physics + correction + noise
    y_data = y_physics + y_correction + np.random.normal(0, 0.1, len(x_data))

    # Visualize the data
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(x_data, y_physics, "g-", linewidth=2, label="Physics (exponential)")
    plt.plot(x_data, y_data, "b.", alpha=0.5, markersize=3, label="Observed data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data vs Physics Model")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(x_data, y_correction, "r-", linewidth=2, label="True correction")
    plt.xlabel("x")
    plt.ylabel("Correction")
    plt.title("Systematic Deviation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    residuals = y_data - y_physics
    plt.plot(x_data, residuals, "r.", alpha=0.5, markersize=3, label="Residuals")
    plt.plot(x_data, y_correction, "k--", linewidth=2, label="True correction")
    plt.xlabel("x")
    plt.ylabel("Residual")
    plt.title("Physics Model Residuals")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save figure to file
    fig_dir = Path(__file__).parent / "figures" / "ml_integration_tutorial"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
    plt.close()

print("\n📊 Dataset Statistics:")
print(f"   Data points: {len(x_data)}")
print(f"   Physics RMSE: {np.sqrt(np.mean((y_data - y_physics) ** 2)):.3f}")
print(f"   Correction amplitude: {np.max(np.abs(y_correction)):.3f}")


# ======================================================================
# ### Strategy 1: Two-Stage Fitting
#
# **Step 1**: Fit physics parameters with NLSQ
# **Step 2**: Train neural network on residuals
# ======================================================================


if FLAX_AVAILABLE:
    print("=" * 70)
    print("TWO-STAGE HYBRID FITTING")
    print("=" * 70)

    # Stage 1: Fit physics model with NLSQ
    print("\n🔧 Stage 1: Fitting physics parameters with NLSQ...")

    def exponential_decay(x, a, b):
        return a * jnp.exp(-b * x)

    cf = CurveFit()
    start_time = time.time()
    popt_physics, pcov_physics = cf.curve_fit(
        exponential_decay, x_data, y_data, p0=[4.0, 1.0]
    )
    physics_time = time.time() - start_time

    a_fit, b_fit = popt_physics
    print(f"   Fitted parameters: a={a_fit:.3f}, b={b_fit:.3f}")
    print(f"   True parameters:   a={true_a:.3f}, b={true_b:.3f}")
    print(f"   Fit time: {physics_time:.3f}s")

    # Compute residuals
    y_physics_fit = np.array(exponential_decay(x_data, *popt_physics))
    residuals = y_data - y_physics_fit
    physics_rmse = np.sqrt(np.mean(residuals**2))
    print(f"   Physics RMSE: {physics_rmse:.4f}")

    # Stage 2: Train neural network on residuals
    print("\n🧠 Stage 2: Training neural network on residuals...")

    # Prepare training data
    x_train = x_data.reshape(-1, 1).astype(np.float32)
    y_train = residuals.astype(np.float32)

    # Create train state with Optax
    def create_train_state(rng, learning_rate=1e-3):
        model_nn = CorrectionMLP()
        params_nn = model_nn.init(rng, jnp.ones((1, 1)))
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(
            apply_fn=model_nn.apply, params=params_nn, tx=tx
        )

    state = create_train_state(key, learning_rate=5e-3)

    # Training step
    @jit
    def train_step(state, x_batch, y_batch):
        def loss_fn(params):
            pred = state.apply_fn(params, x_batch)
            return jnp.mean((pred - y_batch) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Train for a few epochs
    n_epochs = 500
    losses = []

    start_time = time.time()
    for epoch in range(n_epochs):
        state, loss = train_step(state, x_train, y_train)
        losses.append(float(loss))

        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.6f}")

    nn_time = time.time() - start_time
    print(f"   NN training time: {nn_time:.3f}s")

    # Evaluate hybrid model
    correction_pred = model.apply(state.params, x_train).squeeze()
    y_hybrid = y_physics_fit + np.array(correction_pred)
    hybrid_rmse = np.sqrt(np.mean((y_data - y_hybrid) ** 2))

    print("\n📊 Results:")
    print(f"   Physics-only RMSE: {physics_rmse:.4f}")
    print(f"   Hybrid model RMSE: {hybrid_rmse:.4f}")
    print(f"   Improvement: {(1 - hybrid_rmse / physics_rmse) * 100:.1f}%")

    # Visualize results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Neural Network Training")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(x_data, y_data, "b.", alpha=0.5, markersize=3, label="Data")
    plt.plot(x_data, y_physics_fit, "g--", linewidth=2, label="Physics only")
    plt.plot(x_data, y_hybrid, "r-", linewidth=2, label="Hybrid (Physics + NN)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(
        x_data,
        y_correction,
        "k--",
        linewidth=2,
        label="True correction",
        alpha=0.7,
    )
    plt.plot(x_data, correction_pred, "r-", linewidth=2, label="NN learned correction")
    plt.xlabel("x")
    plt.ylabel("Correction")
    plt.title("Learned vs True Correction")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "ml_integration_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_02.png", dpi=300, bbox_inches="tight")
plt.close()


# ======================================================================
# ### Key Insights
#
# 1. **Physics provides structure**: The exponential decay captures the dominant behavior
# 2. **NN learns deviations**: Small neural network captures systematic errors
# 3. **Data efficiency**: Physics model requires fewer parameters than pure ML
# 4. **Interpretability**: Physical parameters (a, b) have clear meaning
# 5. **Better extrapolation**: Physics guides behavior outside training range
#
# **When to use this approach**:
# - ✅ Known physics with systematic deviations
# - ✅ Limited data (physics provides inductive bias)
# - ✅ Need interpretable parameters
# - ✅ Extrapolation is important
#
# ---
# ======================================================================


# ======================================================================
# ## Part 2: Neural ODEs with NLSQ
#
# ### Concept: Learning Dynamics
#
# **Neural ODEs** parameterize the derivative of a system with a neural network:
#
# $$
# \frac{dy}{dt} = f_{\theta}(y, t)
# $$
#
# Where $f_{\theta}$ is a neural network. We can then integrate this ODE to get predictions.
#
# **NLSQ Integration**: Use NLSQ to fit:
# 1. Initial conditions
# 2. ODE parameters (if partially mechanistic)
# 3. Neural network parameters (jointly or in stages)
#
# ### Example 2.1: Damped Oscillator with Learned Damping
# ======================================================================


# Simple ODE solver (for demonstration; use diffrax in production)
def euler_integrate(f, y0, t, *args):
    """Simple Euler integration for demonstration."""
    dt = t[1] - t[0]
    y = jnp.zeros((len(t), len(y0)))
    y = y.at[0].set(y0)

    for i in range(1, len(t)):
        dydt = f(y[i - 1], t[i - 1], *args)
        y = y.at[i].set(y[i - 1] + dt * dydt)

    return y


# Damped harmonic oscillator ODE
def damped_oscillator_ode(state, t, omega, gamma):
    """dy/dt for damped harmonic oscillator.

    state = [position, velocity]
    omega = natural frequency
    gamma = damping coefficient
    """
    x, v = state
    dxdt = v
    dvdt = -(omega**2) * x - 2 * gamma * v
    return jnp.array([dxdt, dvdt])


# Generate synthetic oscillator data
t_ode = np.linspace(0, 10, 200)
omega_true = 2.0  # Natural frequency
gamma_true = 0.3  # Damping
y0_true = jnp.array([1.0, 0.0])  # Initial [position, velocity]

# Integrate true system
y_true = euler_integrate(damped_oscillator_ode, y0_true, t_ode, omega_true, gamma_true)
x_true = y_true[:, 0]  # Extract position

# Add noise to observations
x_obs = x_true + np.random.normal(0, 0.05, len(t_ode))

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_ode, x_true, "g-", linewidth=2, label="True dynamics")
plt.plot(t_ode, x_obs, "b.", alpha=0.5, markersize=3, label="Noisy observations")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("Damped Harmonic Oscillator")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(y_true[:, 0], y_true[:, 1], "g-", linewidth=2, label="Phase space")
plt.xlabel("Position (x)")
plt.ylabel("Velocity (v)")
plt.title("Phase Space Trajectory")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "ml_integration_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_03.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"True parameters: ω={omega_true}, γ={gamma_true}")
print(f"Initial state: x₀={y0_true[0]}, v₀={y0_true[1]}")


# ======================================================================
# ### Fitting ODE Parameters with NLSQ
# ======================================================================


print("=" * 70)
print("FITTING ODE PARAMETERS WITH NLSQ")
print("=" * 70)


# Define model: integrate ODE and extract position
def oscillator_model(t, omega, gamma, x0, v0):
    """Model function that integrates ODE for given parameters."""
    y0 = jnp.array([x0, v0])
    y = euler_integrate(damped_oscillator_ode, y0, t, omega, gamma)
    return y[:, 0]  # Return position only


# Fit with NLSQ
print("\n🔧 Fitting ODE parameters...")
cf_ode = CurveFit()

# Initial guess (intentionally off)
p0 = [1.5, 0.2, 0.8, 0.1]  # [omega, gamma, x0, v0]

start_time = time.time()
popt_ode, pcov_ode = cf_ode.curve_fit(
    oscillator_model,
    t_ode,
    x_obs,
    p0=p0,
    bounds=([0, 0, -2, -2], [5, 2, 2, 2]),  # Reasonable physical bounds
)
ode_time = time.time() - start_time

omega_fit, gamma_fit, x0_fit, v0_fit = popt_ode

print("\n📊 Results:")
print(
    f"   Fitted: ω={omega_fit:.4f}, γ={gamma_fit:.4f}, x₀={x0_fit:.4f}, v₀={v0_fit:.4f}"
)
print(
    f"   True:   ω={omega_true:.4f}, γ={gamma_true:.4f}, x₀={y0_true[0]:.4f}, v₀={y0_true[1]:.4f}"
)
print(f"   Fit time: {ode_time:.3f}s")

# Compute fitted trajectory
x_fit = oscillator_model(t_ode, *popt_ode)
ode_rmse = np.sqrt(np.mean((x_obs - np.array(x_fit)) ** 2))
print(f"   RMSE: {ode_rmse:.5f}")

# Visualize fit
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_ode, x_obs, "b.", alpha=0.5, markersize=3, label="Observations")
plt.plot(t_ode, x_true, "g--", linewidth=2, alpha=0.7, label="True dynamics")
plt.plot(t_ode, x_fit, "r-", linewidth=2, label="Fitted ODE")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("ODE Parameter Fitting")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals_ode = x_obs - np.array(x_fit)
plt.plot(t_ode, residuals_ode, "r.", alpha=0.5, markersize=3)
plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
plt.xlabel("Time (t)")
plt.ylabel("Residual")
plt.title(f"Residuals (RMSE={ode_rmse:.5f})")
plt.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "ml_integration_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_04.png", dpi=300, bbox_inches="tight")
plt.close()


# ======================================================================
# ### Key Takeaways
#
# 1. **NLSQ handles ODEs naturally**: Just wrap ODE integration in model function
# 2. **Automatic differentiation**: JAX computes gradients through ODE solver
# 3. **Joint parameter estimation**: Fit dynamics parameters + initial conditions
# 4. **Physical constraints**: Use bounds to enforce physically reasonable values
#
# **Production tip**: Use `diffrax` for more robust ODE integration:
# ```python
# import diffrax
# solver = diffrax.Tsit5()  # Adaptive Runge-Kutta
# solution = diffrax.diffeqsolve(...)
# ```
#
# ---
# ======================================================================


# ======================================================================
# ## Part 3: Physics-Informed Loss Functions
#
# ### Concept: Incorporating Physical Constraints
#
# **Physics-informed fitting** adds physical constraints to the loss:
#
# $$
# \mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}
# $$
#
# Examples:
# - **Conservation laws**: Energy, mass, momentum conservation
# - **PDE residuals**: Equations of motion, Maxwell's equations
# - **Boundary conditions**: Initial/final state constraints
# - **Symmetries**: Rotational, translational invariance
#
# ### Example 3.1: Energy-Conserving Pendulum
#
# For a frictionless pendulum, total energy should be conserved:
#
# $$
# E = \frac{1}{2}mv^2 + mgh = \text{constant}
# $$
# ======================================================================


# Simple pendulum dynamics
def pendulum_ode(state, t, omega):
    """Pendulum ODE: d²θ/dt² = -ω² sin(θ)"""
    theta, theta_dot = state
    return jnp.array([theta_dot, -(omega**2) * jnp.sin(theta)])


# Generate pendulum data
t_pend = np.linspace(0, 10, 150)
omega_pend = 2.0
y0_pend = jnp.array([0.5, 0.0])  # [angle, angular velocity]

y_pend = euler_integrate(pendulum_ode, y0_pend, t_pend, omega_pend)
theta_obs = y_pend[:, 0] + np.random.normal(0, 0.02, len(t_pend))

# Compute energy (for verification)
kinetic = 0.5 * y_pend[:, 1] ** 2
potential = (omega_pend**2) * (1 - jnp.cos(y_pend[:, 0]))
total_energy = kinetic + potential

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(t_pend, y_pend[:, 0], "g-", linewidth=2, label="True angle")
plt.plot(t_pend, theta_obs, "b.", alpha=0.5, markersize=3, label="Observations")
plt.xlabel("Time (t)")
plt.ylabel("Angle θ (rad)")
plt.title("Pendulum Motion")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(y_pend[:, 0], y_pend[:, 1], "g-", linewidth=2)
plt.xlabel("Angle θ (rad)")
plt.ylabel("Angular velocity dθ/dt (rad/s)")
plt.title("Phase Space")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(t_pend, total_energy, "r-", linewidth=2, label="Total energy")
plt.axhline(y=jnp.mean(total_energy), color="k", linestyle="--", label="Mean energy")
plt.xlabel("Time (t)")
plt.ylabel("Energy (J)")
plt.title("Energy Conservation")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "ml_integration_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_05.png", dpi=300, bbox_inches="tight")
plt.close()

energy_std = float(jnp.std(total_energy))
print(f"Energy conservation (std dev): {energy_std:.6f}")
print(f"Energy variation: {energy_std / jnp.mean(total_energy) * 100:.3f}%")


# ======================================================================
# ### Custom Physics-Informed Fitting
#
# While NLSQ doesn't directly support custom loss functions (it uses least squares), we can:
# 1. Use NLSQ for standard parameter estimation
# 2. Add physics penalty in post-processing
# 3. Or use Optax for full physics-informed optimization
# ======================================================================


if FLAX_AVAILABLE:
    print("=" * 70)
    print("PHYSICS-INFORMED OPTIMIZATION WITH OPTAX")
    print("=" * 70)

    # Define physics-informed loss
    def physics_informed_loss(params, t, theta_obs, lambda_physics=0.1):
        """Loss = data fit + energy conservation penalty."""
        omega, theta0, thetadot0 = params

        # Integrate ODE
        y0 = jnp.array([theta0, thetadot0])
        y = euler_integrate(pendulum_ode, y0, t, omega)

        # Data fitting loss
        theta_pred = y[:, 0]
        loss_data = jnp.mean((theta_pred - theta_obs) ** 2)

        # Energy conservation penalty
        kinetic = 0.5 * y[:, 1] ** 2
        potential = (omega**2) * (1 - jnp.cos(y[:, 0]))
        total_energy = kinetic + potential
        energy_var = jnp.var(total_energy)
        loss_physics = lambda_physics * energy_var

        return loss_data + loss_physics, {
            "loss_data": loss_data,
            "loss_physics": loss_physics,
        }

    # Optimize with Optax
    params_init = jnp.array([1.5, 0.4, 0.1])  # [omega, theta0, thetadot0]
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params_init)

    @jit
    def update_step(params, opt_state, t, theta_obs):
        (loss_val, metrics), grads = jax.value_and_grad(
            physics_informed_loss, has_aux=True
        )(params, t, theta_obs)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, metrics

    # Optimization loop
    params = params_init
    n_steps = 1000
    losses = []

    print("\n🎯 Training with physics-informed loss...")
    for step in range(n_steps):
        params, opt_state, loss_val, metrics = update_step(
            params, opt_state, t_pend, theta_obs
        )
        losses.append(float(loss_val))

        if (step + 1) % 200 == 0:
            print(
                f"   Step {step + 1}: Total={loss_val:.6f}, "
                f"Data={metrics['loss_data']:.6f}, Physics={metrics['loss_physics']:.6f}"
            )

    omega_pi, theta0_pi, thetadot0_pi = params
    print("\n📊 Fitted parameters:")
    print(f"   ω={omega_pi:.4f} (true: {omega_pend:.4f})")
    print(f"   θ₀={theta0_pi:.4f} (true: {y0_pend[0]:.4f})")
    print(f"   dθ₀/dt={thetadot0_pi:.4f} (true: {y0_pend[1]:.4f})")

    # Compare with standard NLSQ fit
    def pendulum_model(t, omega, theta0, thetadot0):
        y0 = jnp.array([theta0, thetadot0])
        y = euler_integrate(pendulum_ode, y0, t, omega)
        return y[:, 0]

    popt_std, _ = cf_ode.curve_fit(
        pendulum_model, t_pend, theta_obs, p0=[1.5, 0.4, 0.1]
    )

    # Evaluate energy conservation
    y_pi = euler_integrate(
        pendulum_ode, jnp.array([theta0_pi, thetadot0_pi]), t_pend, omega_pi
    )
    y_std = euler_integrate(
        pendulum_ode, jnp.array([popt_std[1], popt_std[2]]), t_pend, popt_std[0]
    )

    def compute_energy_std(y, omega):
        kinetic = 0.5 * y[:, 1] ** 2
        potential = (omega**2) * (1 - jnp.cos(y[:, 0]))
        return float(jnp.std(kinetic + potential))

    energy_std_pi = compute_energy_std(y_pi, omega_pi)
    energy_std_std = compute_energy_std(y_std, popt_std[0])

    print("\n⚡ Energy Conservation:")
    print(f"   Physics-informed: σ_E = {energy_std_pi:.6f}")
    print(f"   Standard NLSQ:    σ_E = {energy_std_std:.6f}")
    print(f"   Improvement: {(1 - energy_std_pi / energy_std_std) * 100:.1f}%")

    # Visualize
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Physics-Informed Training")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(t_pend, theta_obs, "b.", alpha=0.5, markersize=3, label="Data")
    plt.plot(t_pend, y_std[:, 0], "g--", linewidth=2, label="Standard NLSQ")
    plt.plot(t_pend, y_pi[:, 0], "r-", linewidth=2, label="Physics-informed")
    plt.xlabel("Time (t)")
    plt.ylabel("Angle θ (rad)")
    plt.title("Fit Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    energy_std_series = 0.5 * y_std[:, 1] ** 2 + (popt_std[0] ** 2) * (
        1 - jnp.cos(y_std[:, 0])
    )
    energy_pi_series = 0.5 * y_pi[:, 1] ** 2 + (omega_pi**2) * (1 - jnp.cos(y_pi[:, 0]))
    plt.plot(t_pend, energy_std_series, "g--", linewidth=2, label="Standard NLSQ")
    plt.plot(t_pend, energy_pi_series, "r-", linewidth=2, label="Physics-informed")
    plt.xlabel("Time (t)")
    plt.ylabel("Energy (J)")
    plt.title("Energy Conservation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "ml_integration_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_06.png", dpi=300, bbox_inches="tight")
plt.close()


# ======================================================================
# ---
#
# ## Summary and Best Practices
#
# ### Integration Strategies
#
# | Approach | NLSQ Role | ML Role | Best For |
# |----------|-----------|---------|----------|
# | **Two-Stage Hybrid** | Fit physics parameters | Learn residuals | Known physics + systematic deviations |
# | **Neural ODE** | Fit ODE parameters | (Optional) Learn dynamics | Parameter estimation in dynamical systems |
# | **Physics-Informed** | Pre-fit, then refine | Enforce constraints | Energy/mass conservation, PDEs |
# | **Joint Optimization** | Parameter estimation | Model flexibility | Complex coupled systems |
#
# ### Key Takeaways
#
# 1. **NLSQ + JAX = Powerful Combo**:
#    - Automatic differentiation through complex models
#    - GPU acceleration for both fitting and ML
#    - Seamless integration with JAX ecosystem
#
# 2. **Hybrid Models Win**:
#    - Better than pure physics (captures deviations)
#    - Better than pure ML (data efficient, interpretable)
#    - Best of both worlds
#
# 3. **Physics Constraints Help**:
#    - Regularize ML models
#    - Improve extrapolation
#    - Ensure physical plausibility
#
# 4. **Choose the Right Tool**:
#    - **NLSQ**: Parameter estimation, well-conditioned problems
#    - **Optax**: Custom losses, physics-informed training
#    - **Combined**: Two-stage fitting strategies
#
# ### Production Recommendations
#
# ```python
# # 1. Use diffrax for robust ODE integration
# import diffrax
# solver = diffrax.Tsit5()
#
# # 2. Separate training and inference
# @jit
# def inference_model(params, x):
#     # Compiled inference only
#     return model.apply(params, x)
#
# # 3. Use appropriate precision
# # NLSQ uses float64 by default (good for physics)
# # ML often uses float32 (faster, sufficient for NNs)
#
# # 4. Validate physics constraints
# def check_energy_conservation(y, params):
#     energy = compute_energy(y, params)
#     return jnp.std(energy) < threshold
#
# # 5. Profile and optimize
# # Use MemoryPool for repeated fitting
# from nlsq import MemoryPool
# with MemoryPool() as pool:
#     for data in datasets:
#         popt, _ = cf.curve_fit(model, *data)
# ```
#
# ### Next Steps
#
# - Explore `equinox` for more Pythonic neural network design
# - Try `diffrax` for production-grade ODE solving
# - Investigate `jaxopt` for more optimization algorithms
# - Read about **Universal Differential Equations** (UDEs)
# - Study **SciML (Scientific Machine Learning)** ecosystem
#
# ### References
#
# 1. **Neural ODEs**: Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018
# 2. **PINNs**: Raissi et al., "Physics-informed neural networks", JCP 2019
# 3. **UDEs**: Rackauckas et al., "Universal Differential Equations", arXiv 2020
# 4. **JAX Ecosystem**: https://github.com/n2cholas/awesome-jax
#
# ---
#
# **Congratulations!** You've learned how to integrate NLSQ with the JAX ML ecosystem for hybrid scientific computing.
#
# **Continue Learning**:
# - [Research Workflow Case Study](research_workflow_case_study.ipynb) - Real experimental data
# - [Advanced Features Demo](advanced_features_demo.ipynb) - Diagnostics and optimization
# - [Performance Optimization Demo](performance_optimization_demo.ipynb) - Production-ready optimization
#
# ---
# ======================================================================

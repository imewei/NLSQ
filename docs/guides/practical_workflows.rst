Practical Workflows Guide
=========================

This guide provides ready-to-use workflows for common scientific and engineering
applications. Each workflow is optimized for its specific use case.

.. contents:: Table of Contents
   :local:
   :depth: 2

Workflow Selection Quick Reference
----------------------------------

.. list-table:: Workflow Decision Matrix
   :header-rows: 1
   :widths: 25 20 20 35

   * - Scenario
     - Preset/Method
     - Key Setting
     - When to Use
   * - Quick exploration
     - ``preset="fast"``
     - Loose tolerances
     - Initial parameter estimation, interactive fitting
   * - Production fitting
     - ``preset="robust"``
     - Multi-start enabled
     - Unknown data quality, automated pipelines
   * - Publication quality
     - ``preset="quality"``
     - Tight tolerances + validation
     - Final results for papers, critical measurements
   * - Large datasets (1-10M)
     - ``preset="large_robust"``
     - Chunked processing
     - High-throughput data, imaging
   * - Huge datasets (10-100M)
     - ``preset="streaming"``
     - Hybrid streaming
     - Time series, spectroscopy arrays
   * - HPC clusters
     - ``preset="hpc_distributed"``
     - Checkpointing
     - Multi-node, fault-tolerant jobs
   * - Low memory
     - ``preset="memory_efficient"``
     - Small chunks
     - Laptops, shared servers

---

Workflows by Dataset Size
-------------------------

Small Datasets (< 10,000 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case**: Laboratory measurements, single experiments, teaching

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from nlsq import fit, curve_fit


    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c


    # Option 1: Simple fit (auto-selects STANDARD tier)
    popt, pcov = fit(model, x, y, p0=[1, 0.5, 0])

    # Option 2: Maximum precision for publication
    popt, pcov = fit(model, x, y, p0=[1, 0.5, 0], preset="quality")

    # Option 3: Fast exploration
    popt, pcov = fit(model, x, y, p0=[1, 0.5, 0], preset="fast")

Medium Datasets (10,000 - 1,000,000 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case**: High-throughput screening, imaging, time-resolved spectroscopy

.. code-block:: python

    from nlsq import fit, OptimizationGoal

    # Auto-detection (recommended)
    popt, pcov = fit(model, x, y, p0=p0)  # Auto-selects chunked if needed

    # Explicit large dataset handling with robustness
    popt, pcov = fit(model, x, y, p0=p0, preset="large_robust")

    # Memory-constrained environment
    popt, pcov = fit(model, x, y, p0=p0, preset="memory_efficient", memory_limit_gb=4.0)

Large Datasets (1,000,000 - 100,000,000 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case**: Synchrotron data, detector arrays, continuous monitoring

.. code-block:: python

    from nlsq import fit, curve_fit_large
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    # Option 1: Automatic streaming
    popt, pcov = fit(model, x, y, p0=p0, preset="streaming")

    # Option 2: Explicit streaming with custom config
    # Note: Defense layers are enabled by default (v0.3.6+)
    config = HybridStreamingConfig(
        normalize=True,
        warmup_iterations=300,
        gauss_newton_tol=1e-8,
        chunk_size=50000,
        enable_checkpoints=True,
    )
    popt, pcov = fit(model, x, y, p0=p0, method="hybrid_streaming")

    # Option 3: curve_fit_large for explicit control
    popt, pcov = curve_fit_large(
        model,
        x,
        y,
        p0=p0,
        memory_limit_gb=8.0,
        show_progress=True,
    )

    # Option 4: For warm-start refinement (v0.3.6+)
    config = HybridStreamingConfig.defense_strict()
    popt, pcov = fit(model, x, y, p0=previous_popt, method="hybrid_streaming")

Massive Datasets (> 100,000,000 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case**: Climate data, particle physics, genomics

.. code-block:: python

    from nlsq import fit
    from nlsq import StreamingOptimizer, StreamingConfig
    from nlsq.streaming_optimizer import create_hdf5_dataset

    # Store data in HDF5 for streaming
    # (data never fully loaded into memory)

    # Option 1: Streaming with checkpoints
    popt, pcov = fit(model, x, y, p0=p0, preset="hpc_distributed")

    # Option 2: Direct streaming optimizer
    config = StreamingConfig(
        batch_size=100000,
        max_epochs=50,
        enable_fault_tolerance=True,
        checkpoint_frequency=50,
    )
    optimizer = StreamingOptimizer(config)
    result = optimizer.fit_streaming(model, "data.h5", p0=p0)

---

Workflows by Application Domain
-------------------------------

Spectroscopy
~~~~~~~~~~~~

**Peak Fitting (Gaussian/Lorentzian)**

.. code-block:: python

    import jax.numpy as jnp
    from nlsq import fit


    def gaussian_peak(x, amplitude, center, sigma, baseline):
        """Single Gaussian peak with baseline."""
        return amplitude * jnp.exp(-0.5 * ((x - center) / sigma) ** 2) + baseline


    def lorentzian_peak(x, amplitude, center, gamma, baseline):
        """Single Lorentzian peak with baseline."""
        return amplitude * gamma**2 / ((x - center) ** 2 + gamma**2) + baseline


    def voigt_approx(x, amplitude, center, sigma, gamma, baseline):
        """Pseudo-Voigt approximation (weighted sum)."""
        eta = gamma / (sigma + gamma + 1e-10)  # Mixing parameter
        G = jnp.exp(-0.5 * ((x - center) / sigma) ** 2)
        L = gamma**2 / ((x - center) ** 2 + gamma**2)
        return amplitude * (eta * L + (1 - eta) * G) + baseline


    # For well-resolved peaks
    popt, pcov = fit(
        gaussian_peak,
        wavelength,
        intensity,
        p0=[1000, 500, 10, 100],
        bounds=([0, 400, 1, 0], [10000, 600, 50, 500]),
    )

    # For overlapping peaks (multi-start recommended)
    popt, pcov = fit(
        gaussian_peak,
        wavelength,
        intensity,
        p0=[1000, 500, 10, 100],
        preset="robust",
        multistart=True,
        n_starts=10,
    )

**Multi-Peak Fitting**

.. code-block:: python

    def multi_gaussian(x, *params):
        """N Gaussian peaks: params = [a1, c1, s1, a2, c2, s2, ..., baseline]"""
        n_peaks = (len(params) - 1) // 3
        result = jnp.zeros_like(x)
        for i in range(n_peaks):
            a, c, s = params[3 * i], params[3 * i + 1], params[3 * i + 2]
            result += a * jnp.exp(-0.5 * ((x - c) / s) ** 2)
        return result + params[-1]  # baseline


    # 3 peaks + baseline = 10 parameters
    p0 = [
        100,
        450,
        10,  # Peak 1
        200,
        500,
        15,  # Peak 2
        150,
        550,
        12,  # Peak 3
        50,
    ]  # Baseline

    # Use global optimization for multi-peak (many local minima)
    popt, pcov = fit(
        multi_gaussian, x, y, p0=p0, preset="robust", multistart=True, n_starts=20
    )

**Exponential Decay (Fluorescence Lifetime)**

.. code-block:: python

    def single_exponential(x, amplitude, tau, offset):
        """Single exponential decay."""
        return amplitude * jnp.exp(-x / tau) + offset


    def biexponential(x, a1, tau1, a2, tau2, offset):
        """Bi-exponential decay (two components)."""
        return a1 * jnp.exp(-x / tau1) + a2 * jnp.exp(-x / tau2) + offset


    def stretched_exponential(x, amplitude, tau, beta, offset):
        """Stretched exponential (Kohlrausch-Williams-Watts)."""
        return amplitude * jnp.exp(-((x / tau) ** beta)) + offset


    # Fluorescence lifetime (high precision)
    popt, pcov = fit(
        biexponential,
        time_ns,
        counts,
        p0=[1000, 2.5, 500, 8.0, 10],
        bounds=([0, 0.1, 0, 0.1, 0], [10000, 100, 10000, 100, 1000]),
        preset="quality",
    )

X-ray Scattering (SAXS/WAXS/XPCS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Form Factor Fitting**

.. code-block:: python

    def sphere_form_factor(q, scale, radius, background):
        """Sphere form factor P(q)."""
        qr = q * radius
        # Avoid division by zero
        qr = jnp.where(qr < 1e-10, 1e-10, qr)
        P_q = (3 * (jnp.sin(qr) - qr * jnp.cos(qr)) / qr**3) ** 2
        return scale * P_q + background


    def guinier(q, I0, Rg, background):
        """Guinier approximation for low-q region."""
        return I0 * jnp.exp(-(q**2) * Rg**2 / 3) + background


    def power_law(q, scale, exponent, background):
        """Power law for high-q region."""
        return scale * q ** (-exponent) + background


    # SAXS fitting (often noisy, use robust)
    popt, pcov = fit(
        sphere_form_factor,
        q,
        I_q,
        p0=[1.0, 50.0, 0.01],
        bounds=([0, 1, 0], [1000, 500, 1]),
        preset="robust",
    )

**XPCS Correlation Functions**

.. code-block:: python

    def simple_exponential_g2(t, beta, tau, baseline):
        """g2 correlation function with single relaxation."""
        return baseline + beta * jnp.exp(-2 * t / tau)


    def stretched_g2(t, beta, tau, gamma, baseline):
        """Stretched exponential g2 (heterogeneous dynamics)."""
        return baseline + beta * jnp.exp(-2 * (t / tau) ** gamma)


    def double_exponential_g2(t, beta1, tau1, beta2, tau2, baseline):
        """Two-relaxation g2 function."""
        return baseline + beta1 * jnp.exp(-2 * t / tau1) + beta2 * jnp.exp(-2 * t / tau2)


    # XPCS with multi-scale parameters (use hybrid streaming)
    # tau can be ~1e-6 to 1e3, beta ~0.1 to 1
    # Defense layers (v0.3.6+) protect against warm-start divergence
    from nlsq import HybridStreamingConfig

    config = HybridStreamingConfig.scientific_default()  # Recommended for XPCS
    popt, pcov = fit(
        stretched_g2,
        lag_time,
        g2_data,
        p0=[0.3, 1.0, 1.0, 1.0],
        bounds=([0, 1e-6, 0.1, 0.9], [1, 1e6, 2.0, 1.1]),
        method="hybrid_streaming",
    )  # Handles scale differences + defense layers

Kinetics and Reaction Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Michaelis-Menten Enzyme Kinetics**

.. code-block:: python

    def michaelis_menten(S, Vmax, Km):
        """Michaelis-Menten enzyme kinetics."""
        return Vmax * S / (Km + S)


    def hill_equation(S, Vmax, K, n):
        """Hill equation (cooperative binding)."""
        return Vmax * S**n / (K**n + S**n)


    def substrate_inhibition(S, Vmax, Km, Ki):
        """Substrate inhibition model."""
        return Vmax * S / (Km + S * (1 + S / Ki))


    # Enzyme kinetics (well-behaved, standard fit)
    popt, pcov = fit(
        michaelis_menten,
        substrate_conc,
        velocity,
        p0=[100, 10],
        bounds=([0, 0], [1000, 1000]),
    )

**Chemical Kinetics (Rate Equations)**

.. code-block:: python

    def first_order_decay(t, A0, k, offset):
        """First-order reaction: A → products."""
        return A0 * jnp.exp(-k * t) + offset


    def second_order_equal(t, A0, k):
        """Second-order reaction with equal concentrations."""
        return A0 / (1 + k * A0 * t)


    def consecutive_reactions(t, A0, k1, k2):
        """A → B → C consecutive first-order reactions.
        Returns concentration of B."""
        return A0 * k1 / (k2 - k1) * (jnp.exp(-k1 * t) - jnp.exp(-k2 * t))


    # For fast kinetics (many data points)
    popt, pcov = fit(
        first_order_decay,
        time,
        concentration,
        p0=[1.0, 0.1, 0.0],
        preset="large_robust" if len(time) > 100000 else "robust",
    )

Binding and Adsorption
~~~~~~~~~~~~~~~~~~~~~~

**Binding Isotherms**

.. code-block:: python

    def langmuir(C, Qmax, Kd):
        """Langmuir binding isotherm."""
        return Qmax * C / (Kd + C)


    def freundlich(C, Kf, n):
        """Freundlich isotherm (heterogeneous surfaces)."""
        return Kf * C ** (1 / n)


    def langmuir_freundlich(C, Qmax, K, n):
        """Langmuir-Freundlich (Sips) isotherm."""
        return Qmax * (K * C) ** n / (1 + (K * C) ** n)


    def two_site_langmuir(C, Q1, K1, Q2, K2):
        """Two-site Langmuir (heterogeneous binding)."""
        return Q1 * C / (K1 + C) + Q2 * C / (K2 + C)


    # Binding curve fitting
    popt, pcov = fit(
        langmuir,
        concentration,
        binding,
        p0=[100, 10],
        bounds=([0, 0], [1000, 1000]),
        preset="robust",
    )

**Dose-Response Curves**

.. code-block:: python

    def four_parameter_logistic(x, bottom, top, EC50, hill):
        """4PL dose-response curve."""
        return bottom + (top - bottom) / (1 + (EC50 / x) ** hill)


    def five_parameter_logistic(x, bottom, top, EC50, hill, asymmetry):
        """5PL asymmetric dose-response."""
        return bottom + (top - bottom) / (1 + (EC50 / x) ** hill) ** asymmetry


    # IC50/EC50 determination
    popt, pcov = fit(
        four_parameter_logistic,
        dose,
        response,
        p0=[0, 100, 10, 1],
        bounds=([0, 0, 0.001, 0.1], [50, 200, 1000, 10]),
        preset="quality",
    )  # Important for pharmacology

Image Analysis
~~~~~~~~~~~~~~

**2D Gaussian Fitting (Point Spread Function)**

.. code-block:: python

    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
        """2D Gaussian with rotation."""
        x, y = xy
        a = jnp.cos(theta) ** 2 / (2 * sigma_x**2) + jnp.sin(theta) ** 2 / (2 * sigma_y**2)
        b = -jnp.sin(2 * theta) / (4 * sigma_x**2) + jnp.sin(2 * theta) / (4 * sigma_y**2)
        c = jnp.sin(theta) ** 2 / (2 * sigma_x**2) + jnp.cos(theta) ** 2 / (2 * sigma_y**2)
        return offset + amplitude * jnp.exp(
            -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
        )


    # Prepare data for 2D fitting
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x, y)
    xy_data = (X.ravel(), Y.ravel())
    z_data = image.ravel()

    # PSF fitting (many pixels, use streaming for large images)
    if image.size > 1_000_000:
        preset = "streaming"
    else:
        preset = "robust"

    popt, pcov = fit(
        gaussian_2d,
        xy_data,
        z_data,
        p0=[image.max(), image.shape[1] // 2, image.shape[0] // 2, 5, 5, 0, image.min()],
        preset=preset,
    )

**FRAP Recovery Curves**

.. code-block:: python

    def frap_single_diffusion(t, F0, Finf, tau):
        """Single-component FRAP recovery."""
        return Finf - (Finf - F0) * jnp.exp(-t / tau)


    def frap_double_diffusion(t, F0, Finf, f_fast, tau_fast, tau_slow):
        """Two-component FRAP (fast + slow diffusion)."""
        recovery_fast = f_fast * (1 - jnp.exp(-t / tau_fast))
        recovery_slow = (1 - f_fast) * (1 - jnp.exp(-t / tau_slow))
        return F0 + (Finf - F0) * (recovery_fast + recovery_slow)


    # FRAP fitting
    popt, pcov = fit(
        frap_single_diffusion,
        time,
        fluorescence,
        p0=[0.2, 1.0, 5.0],
        bounds=([0, 0, 0.1], [1, 2, 1000]),
        preset="robust",
    )

Materials Science
~~~~~~~~~~~~~~~~~

**Stress-Strain Curves**

.. code-block:: python

    def ramberg_osgood(strain, E, sigma_y, n):
        """Ramberg-Osgood stress-strain relation."""
        # Inverse form: strain = stress/E + (stress/sigma_y)^n
        # Forward form requires numerical solution, so we fit inverse
        return strain  # placeholder - use appropriate formulation


    def power_law_hardening(strain, K, n):
        """Power law strain hardening: sigma = K * epsilon^n."""
        return K * strain**n


    def voce_hardening(strain, sigma_y, sigma_sat, theta):
        """Voce hardening law."""
        return sigma_sat - (sigma_sat - sigma_y) * jnp.exp(-theta * strain / sigma_sat)


    # Stress-strain fitting
    popt, pcov = fit(
        power_law_hardening,
        strain_data,
        stress_data,
        p0=[500, 0.2],
        bounds=([0, 0], [2000, 1]),
        preset="robust",
    )

**Thermal Analysis (DSC/TGA)**

.. code-block:: python

    def arrhenius(T, A, Ea):
        """Arrhenius rate constant: k = A * exp(-Ea/RT)."""
        R = 8.314  # J/(mol·K)
        return A * jnp.exp(-Ea / (R * T))


    def kissinger_peak(T, Tm, Ea, A):
        """Kissinger peak shape for DSC."""
        R = 8.314
        x = Ea / (R * T)
        xm = Ea / (R * Tm)
        return A * jnp.exp(xm - x - jnp.exp(xm - x))


    # Thermal decomposition fitting
    popt, pcov = fit(
        arrhenius,
        temperature_K,
        rate_constant,
        p0=[1e13, 100000],
        bounds=([1e8, 10000], [1e20, 500000]),
        preset="quality",
    )

Physics and Signal Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Damped Oscillations**

.. code-block:: python

    def damped_oscillation(t, A, gamma, omega, phi, offset):
        """Damped harmonic oscillator."""
        return A * jnp.exp(-gamma * t) * jnp.cos(omega * t + phi) + offset


    def critically_damped(t, A, gamma, offset):
        """Critically damped oscillator."""
        return A * (1 + gamma * t) * jnp.exp(-gamma * t) + offset


    # Oscillation fitting (phase can cause local minima)
    popt, pcov = fit(
        damped_oscillation,
        time,
        amplitude,
        p0=[1.0, 0.1, 10.0, 0.0, 0.0],
        preset="robust",
        multistart=True,
        n_starts=10,
    )

**Power Spectrum Fitting**

.. code-block:: python

    def lorentzian_spectrum(f, A, f0, gamma, background):
        """Lorentzian power spectrum."""
        return A * gamma**2 / ((f - f0) ** 2 + gamma**2) + background


    def pink_noise(f, A, alpha, background):
        """1/f^alpha noise spectrum."""
        return A / (f**alpha + 1e-10) + background


    # Power spectrum (often many points)
    popt, pcov = fit(
        lorentzian_spectrum,
        frequency,
        power,
        p0=[1.0, 100.0, 10.0, 0.001],
        preset="large_robust" if len(frequency) > 100000 else "robust",
    )

---

Workflows by Hardware Configuration
-----------------------------------

Single CPU (Laptop/Workstation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import fit

    # Default: Uses all CPU cores via JAX
    popt, pcov = fit(model, x, y, p0=p0)

    # Memory-limited laptop
    popt, pcov = fit(model, x, y, p0=p0, preset="memory_efficient", memory_limit_gb=4.0)

Single GPU
~~~~~~~~~~

.. code-block:: python

    import os

    # Ensure JAX sees the GPU
    # (automatic if JAX GPU is installed)

    from nlsq import fit

    # GPU acceleration is automatic
    popt, pcov = fit(model, x, y, p0=p0)

    # For large datasets on GPU (limited VRAM)
    popt, pcov = fit(
        model, x, y, p0=p0, preset="streaming", memory_limit_gb=8.0
    )  # Match GPU VRAM

Multi-GPU Workstation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import fit
    from nlsq import HybridStreamingConfig

    config = HybridStreamingConfig(
        enable_multi_device=True,  # Use all GPUs
        chunk_size=100000,
        normalize=True,
    )

    popt, pcov = fit(model, x, y, p0=p0, method="hybrid_streaming")

HPC Cluster (PBS/SLURM)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import fit
    from nlsq.workflow import ClusterDetector, create_distributed_config

    # Auto-detect cluster configuration
    detector = ClusterDetector()
    cluster_info = detector.detect()

    if cluster_info:
        print(
            f"Running on {cluster_info.node_count} nodes, "
            f"{cluster_info.total_gpus} GPUs"
        )
        config = create_distributed_config(cluster_info)
    else:
        # Fallback for local runs
        config = None

    # Use HPC preset with checkpointing
    popt, pcov = fit(model, x, y, p0=p0, preset="hpc_distributed", workflow=config)

**PBS Job Script Example:**

.. code-block:: bash

    #!/bin/bash
    #PBS -l select=4:ncpus=40:ngpus=4
    #PBS -l walltime=24:00:00
    #PBS -N nlsq_fit

    cd $PBS_O_WORKDIR
    module load python cuda

    # NLSQ auto-detects PBS environment
    python fit_script.py

---

Workflows by Problem Characteristics
------------------------------------

Well-Behaved Problems (Convex, Unique Minimum)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Characteristics:
- Single global minimum
- Smooth objective function
- Good initial guess available

.. code-block:: python

    # Simple fit, no multi-start needed
    popt, pcov = fit(model, x, y, p0=p0, preset="standard")

    # Or for speed
    popt, pcov = fit(model, x, y, p0=p0, preset="fast")

Multi-Modal Problems (Multiple Local Minima)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Characteristics:
- Multiple peaks/components
- Phase parameters
- Unknown number of features

.. code-block:: python

    # Multi-start is essential
    popt, pcov = fit(model, x, y, p0=p0, preset="robust", multistart=True, n_starts=20)

    # Or use global preset
    popt, pcov = fit(model, x, y, p0=p0, preset="global")

Multi-Scale Parameters
~~~~~~~~~~~~~~~~~~~~~~

Characteristics:
- Parameters differ by orders of magnitude
- E.g., amplitude ~1000, decay rate ~0.001

.. code-block:: python

    # Use hybrid streaming with normalization
    from nlsq import HybridStreamingConfig

    config = HybridStreamingConfig(
        normalize=True,
        normalization_strategy="auto",  # or 'bounds' if bounds provided
    )

    popt, pcov = fit(model, x, y, p0=p0, method="hybrid_streaming")

Ill-Conditioned Problems
~~~~~~~~~~~~~~~~~~~~~~~~

Characteristics:
- High correlation between parameters
- Near-singular Jacobian
- Wide parameter uncertainties

.. code-block:: python

    from nlsq import fit

    # Use tight tolerances and quality validation
    popt, pcov = fit(model, x, y, p0=p0, preset="quality", multistart=True, n_starts=20)

    # Check condition number
    import numpy as np

    cond = np.linalg.cond(pcov)
    if cond > 1e10:
        print(f"Warning: Ill-conditioned covariance (cond={cond:.2e})")
        print("Consider reparameterizing the model")

Constrained Problems
~~~~~~~~~~~~~~~~~~~~

Characteristics:
- Physical constraints (positive values, bounded ranges)
- Parameter relationships

.. code-block:: python

    # Use bounds to enforce constraints
    popt, pcov = fit(
        model, x, y, p0=p0, bounds=([0, 0, 0], [np.inf, 100, 10]), preset="robust"
    )


    # For complex constraints, reparameterize:
    def constrained_model(x, log_a, b_fraction):
        """Reparameterized model with a > 0 and 0 < b < 1."""
        a = jnp.exp(log_a)  # Always positive
        b = 1 / (1 + jnp.exp(-b_fraction))  # Sigmoid: (0, 1)
        return a * jnp.exp(-b * x)

---

Workflow Configuration Reference
--------------------------------

Built-in Presets Summary
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 40

   * - Preset
     - Tier
     - Tolerances
     - Multi-start
     - Use Case
   * - ``standard``
     - STANDARD
     - 1e-8
     - No
     - Default SciPy-compatible behavior
   * - ``quality``
     - STANDARD
     - 1e-10
     - 20 starts
     - Publication, critical measurements
   * - ``fast``
     - STANDARD
     - 1e-6
     - No
     - Exploration, interactive fitting
   * - ``large_robust``
     - CHUNKED
     - 1e-8
     - 10 starts
     - Medium-large datasets with robustness
   * - ``streaming``
     - STREAMING
     - 1e-7
     - No
     - Huge datasets (10M-100M points)
   * - ``hpc_distributed``
     - STREAMING_CHECKPOINT
     - 1e-6
     - 10 starts
     - HPC clusters, fault tolerance
   * - ``memory_efficient``
     - STREAMING
     - 1e-7
     - No
     - Low-memory environments

Custom Workflow Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import WorkflowConfig, WorkflowTier, OptimizationGoal

    # Create custom workflow
    config = WorkflowConfig(
        tier=WorkflowTier.STREAMING,
        goal=OptimizationGoal.QUALITY,
        gtol=1e-10,
        ftol=1e-10,
        xtol=1e-10,
        enable_multistart=True,
        n_starts=30,
        sampler="sobol",  # or "lhs", "halton"
        chunk_size=25000,
        enable_checkpoints=True,
        checkpoint_dir="./checkpoints",
    )

    popt, pcov = fit(model, x, y, p0=p0, workflow=config)

YAML Configuration
~~~~~~~~~~~~~~~~~~

Create ``nlsq.yaml`` in your project directory:

.. code-block:: yaml

    # nlsq.yaml
    default_workflow: "my_custom"
    memory_limit_gb: 16.0

    workflows:
      my_custom:
        tier: "STREAMING"
        goal: "ROBUST"
        gtol: 1e-8
        ftol: 1e-8
        xtol: 1e-8
        enable_multistart: true
        n_starts: 15
        sampler: "lhs"
        chunk_size: 50000
        enable_checkpoints: true

      spectroscopy_peaks:
        tier: "STANDARD"
        goal: "QUALITY"
        gtol: 1e-10
        enable_multistart: true
        n_starts: 25

    runtime:
      device: "gpu"
      precision: "float64"

Load and use:

.. code-block:: python

    from nlsq.workflow import load_yaml_config

    config = load_yaml_config("nlsq.yaml")
    popt, pcov = fit(model, x, y, p0=p0, workflow=config["workflows"]["spectroscopy_peaks"])

---

Common Workflow Patterns
------------------------

Pattern 1: Exploratory → Production → Publication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Step 1: Quick exploration to find reasonable p0
    popt_rough, _ = fit(model, x, y, p0=initial_guess, preset="fast")

    # Step 2: Production fit with robustness
    popt_prod, pcov_prod = fit(model, x, y, p0=popt_rough, preset="robust")

    # Step 3: Publication quality with validation
    popt_final, pcov_final = fit(model, x, y, p0=popt_prod, preset="quality")

Pattern 2: Batch Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import fit
    import numpy as np


    def process_dataset(data_files, model, initial_p0):
        """Process multiple datasets with consistent settings."""
        results = []

        for filepath in data_files:
            x, y = load_data(filepath)

            try:
                popt, pcov = fit(model, x, y, p0=initial_p0, preset="robust")
                perr = np.sqrt(np.diag(pcov))
                results.append(
                    {"file": filepath, "params": popt, "errors": perr, "success": True}
                )
            except Exception as e:
                results.append({"file": filepath, "error": str(e), "success": False})

        return results

Pattern 3: Adaptive Workflow Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import fit, auto_select_workflow, OptimizationGoal


    def smart_fit(model, x, y, p0, goal=OptimizationGoal.ROBUST):
        """Automatically select best workflow based on data characteristics."""
        n_points = len(x)
        n_params = len(p0)

        # Get recommended configuration
        config = auto_select_workflow(n_points=n_points, n_params=n_params, goal=goal)

        print(f"Auto-selected: {config}")

        return fit(model, x, y, p0=p0, workflow=config)

Pattern 4: Checkpoint and Resume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import fit
    from nlsq.workflow import create_checkpoint_directory

    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_directory()
    print(f"Checkpoints: {checkpoint_dir}")

    # Long-running fit with checkpoints
    popt, pcov = fit(
        model, x, y, p0=p0, preset="hpc_distributed", checkpoint_dir=checkpoint_dir
    )

    # If interrupted, resume:
    # popt, pcov = fit(model, x, y, p0=p0,
    #                  preset="hpc_distributed",
    #                  checkpoint_dir=checkpoint_dir,
    #                  resume=True)

Pattern 5: Warm-Start Refinement (v0.3.6+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use when refining parameters from a previous fit or when starting near the optimum.
The 4-Layer Defense Strategy prevents Adam warmup from diverging.

.. code-block:: python

    from nlsq import fit, HybridStreamingConfig
    from nlsq import get_defense_telemetry, reset_defense_telemetry

    # Previous fit gave us a good starting point
    previous_popt = [2.5, 0.5, 1.0]

    # Use strict defense preset for warm-start refinement
    config = HybridStreamingConfig.defense_strict()

    # Reset telemetry to track this specific fit
    reset_defense_telemetry()

    # Refined fit with defense layers active
    popt, pcov = fit(
        model,
        x,
        y,
        p0=previous_popt,
        method="hybrid_streaming",
    )

    # Check defense layer activations
    telemetry = get_defense_telemetry()
    print(telemetry.get_summary())

    # If Layer 1 (Warm Start) activated, the optimizer detected
    # that the initial parameters were already near optimal and
    # skipped the warmup phase to avoid overshooting

Pattern 6: Production Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export defense telemetry for production monitoring systems.

.. code-block:: python

    from nlsq import fit, get_defense_telemetry, reset_defense_telemetry


    def fit_with_monitoring(model, x, y, p0, run_id):
        """Production fit with defense telemetry export."""
        reset_defense_telemetry()

        popt, pcov = fit(model, x, y, p0=p0, method="hybrid_streaming")

        # Export telemetry for monitoring
        telemetry = get_defense_telemetry()
        metrics = telemetry.export_metrics()

        # Log to your monitoring system (Prometheus, DataDog, etc.)
        log_metrics(
            {
                "run_id": run_id,
                "warm_start_triggered": metrics["layer1_count"],
                "adaptive_lr_triggered": metrics["layer2_count"],
                "cost_guard_triggered": metrics["layer3_count"],
                "step_clipping_triggered": metrics["layer4_count"],
                "trigger_rates": telemetry.get_trigger_rates(),
            }
        )

        return popt, pcov

---

See Also
--------

- :doc:`streaming_optimizer_comparison` - StreamingOptimizer vs AdaptiveHybridStreamingOptimizer
- :doc:`defense_layers` - 4-Layer Defense Strategy guide
- :doc:`large_datasets` - Large dataset handling tutorial
- :doc:`/api/nlsq.workflow` - Workflow API reference
- :doc:`/user_guide/yaml_configuration` - YAML configuration guide

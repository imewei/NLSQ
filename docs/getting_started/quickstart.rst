Quickstart Tutorial
===================

Get up and running with NLSQ in 5 minutes! This tutorial demonstrates the essential concepts of GPU-accelerated curve fitting.

Learning Objectives
-------------------

By the end of this tutorial, you'll be able to:

- Perform basic curve fitting with NLSQ
- Understand when to use JAX numpy vs regular numpy
- Interpret fitting results and uncertainty estimates
- Handle common fitting scenarios

Your First Fit
---------------

Let's start with the simplest possible example - fitting a line to data.

.. code-block:: python

    import numpy as np
    from nlsq import CurveFit


    # Define the model function
    def linear(x, m, b):
        return m * x + b


    # Generate synthetic data
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([1, 3, 5, 7, 9, 11])  # y = 2x + 1

    # Create fitter and fit
    cf = CurveFit()
    popt, pcov = cf.curve_fit(linear, x, y)

    print(f"Fitted parameters: m = {popt[0]:.3f}, b = {popt[1]:.3f}")
    print(
        f"Parameter uncertainties: σ_m = {np.sqrt(pcov[0,0]):.3f}, σ_b = {np.sqrt(pcov[1,1]):.3f}"
    )

**Expected Output:**

.. code-block::

    Fitted parameters: m = 2.000, b = 1.000
    Parameter uncertainties: σ_m = 0.000, σ_b = 0.000

Nonlinear Fitting with JAX
---------------------------

For functions that use mathematical operations like exponentials, use JAX numpy for GPU compatibility:

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from nlsq import CurveFit


    # Define exponential decay function using JAX numpy
    def exponential_decay(x, A, tau):
        return A * jnp.exp(-x / tau)


    # Generate noisy synthetic data
    x_data = np.linspace(0, 5, 50)
    true_params = [2.5, 1.2]  # A=2.5, tau=1.2
    y_true = true_params[0] * np.exp(-x_data / true_params[1])
    y_data = y_true + 0.1 * np.random.normal(size=len(x_data))

    # Fit with initial guess
    cf = CurveFit()
    popt, pcov = cf.curve_fit(exponential_decay, x_data, y_data, p0=[2.0, 1.0])

    # Extract results
    A_fit, tau_fit = popt
    A_err, tau_err = np.sqrt(np.diag(pcov))

    print(f"True parameters: A = {true_params[0]}, tau = {true_params[1]}")
    print(
        f"Fitted parameters: A = {A_fit:.3f} ± {A_err:.3f}, tau = {tau_fit:.3f} ± {tau_err:.3f}"
    )

**Expected Output:**

.. code-block::

    True parameters: A = 2.5, tau = 1.2
    Fitted parameters: A = 2.486 ± 0.023, tau = 1.201 ± 0.042

Multi-Parameter Functions
-------------------------

NLSQ excels at fitting complex functions with many parameters:

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from nlsq import CurveFit


    # Define a damped oscillation function
    def damped_oscillation(t, A, freq, decay, phase, offset):
        return A * jnp.exp(-t / decay) * jnp.cos(2 * jnp.pi * freq * t + phase) + offset


    # Generate data
    t = np.linspace(0, 4, 200)
    true_params = [3.0, 1.5, 2.0, 0.5, 1.0]  # A, freq, decay, phase, offset
    y_true = (
        true_params[0]
        * np.exp(-t / true_params[2])
        * np.cos(2 * np.pi * true_params[1] * t + true_params[3])
        + true_params[4]
    )
    y_data = y_true + 0.2 * np.random.normal(size=len(t))

    # Fit with reasonable initial guess
    p0 = [2.5, 1.2, 1.8, 0.3, 0.8]
    cf = CurveFit()
    popt, pcov = cf.curve_fit(damped_oscillation, t, y_data, p0=p0)

    # Display results
    param_names = ["Amplitude", "Frequency", "Decay time", "Phase", "Offset"]
    param_errors = np.sqrt(np.diag(pcov))

    print("Fitting Results:")
    print("-" * 50)
    for i, (name, true_val, fit_val, error) in enumerate(
        zip(param_names, true_params, popt, param_errors)
    ):
        print(f"{name:12}: {fit_val:7.3f} ± {error:6.3f} (true: {true_val:6.3f})")

Understanding the Results
-------------------------

The ``curve_fit`` function returns two important objects:

**popt (Optimal Parameters)**
    The best-fit parameter values that minimize the sum of squared residuals.

**pcov (Covariance Matrix)**
    A matrix containing information about parameter uncertainties and correlations.

.. code-block:: python

    # Parameter uncertainties (standard errors)
    param_errors = np.sqrt(np.diag(pcov))

    # Parameter correlation matrix
    param_correlations = pcov / np.outer(param_errors, param_errors)

    print("Correlation matrix:")
    print(param_correlations)

Large Dataset Fitting
----------------------

NLSQ provides automatic handling for large datasets. The ``curve_fit_large`` function detects dataset size and uses appropriate algorithms:

.. code-block:: python

    from nlsq import curve_fit_large
    import jax.numpy as jnp

    # Generate large dataset (1 million points)
    n_points = 1_000_000
    x_large = np.linspace(0, 10, n_points)
    y_large = 2.0 * jnp.exp(-0.5 * x_large) + 0.3 + np.random.normal(0, 0.05, n_points)


    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c


    # Automatic handling - chunking if needed
    popt, pcov = curve_fit_large(
        exponential,
        x_large,
        y_large,
        p0=[2.0, 0.5, 0.3],
        show_progress=True,  # Show progress for large fits
    )

    print(f"Fitted {n_points:,} points")
    print(f"Parameters: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}")

Common Patterns and Best Practices
-----------------------------------

**1. Always Use JAX Numpy for Mathematical Functions**

.. code-block:: python

    # Good: JAX-compatible
    def gaussian(x, A, mu, sigma):
        return A * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)


    # Avoid: Not JAX-compatible
    def gaussian_bad(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)  # Will cause errors

**2. Provide Good Initial Guesses**

.. code-block:: python

    # Estimate parameters from data
    A_guess = np.max(y_data)  # Peak amplitude
    mu_guess = x_data[np.argmax(y_data)]  # Peak position
    sigma_guess = (np.max(x_data) - np.min(x_data)) / 6  # Width estimate

    p0 = [A_guess, mu_guess, sigma_guess]

**3. Handle Fitting Failures Gracefully**

.. code-block:: python

    try:
        popt, pcov = cf.curve_fit(func, x_data, y_data, p0=p0)
        success = True
    except RuntimeError as e:
        print(f"Fitting failed: {e}")
        success = False

    if success:
        # Check fit quality
        residuals = y_data - func(x_data, *popt)
        rms_error = np.sqrt(np.mean(residuals**2))
        print(f"RMS error: {rms_error:.4f}")

Next Steps
----------

Now that you've mastered the basics, explore:

1. :doc:`../guides/advanced_features` - Learn about different algorithms and options
2. :doc:`../guides/large_datasets` - Handle massive datasets efficiently
3. :doc:`../api/large_datasets_api` - Advanced fitting and parameter constraints
4. :doc:`../api/performance_benchmarks` - Performance analysis and benchmarks

Interactive Examples
---------------------

Try these examples in Google Colab:

- `Basic Fitting Examples <https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/NLSQ%20Quickstart.ipynb>`_
- `Large Dataset Demo <https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/large_dataset_demo.ipynb>`_

Troubleshooting
---------------

**Common Issues:**

1. **Import errors**: Ensure JAX is installed with ``pip install "jax[cpu]>=0.4.20"``
2. **Convergence failures**: Try different initial guesses or check data quality
3. **Performance issues**: Use ``curve_fit_large`` for datasets > 100k points

Need help? Check the `GitHub Issues <https://github.com/imewei/NLSQ/issues>`_ for support and bug reports.

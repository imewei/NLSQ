CurveFit Class
==============

The ``CurveFit`` class provides a reusable, stateful interface for curve fitting.

Why Use CurveFit?
-----------------

1. **JIT Reuse**: Compilations cached between calls
2. **Batch Processing**: Fit multiple datasets efficiently
3. **State Access**: Access diagnostics and internals
4. **Customization**: Configure once, use many times

Basic Usage
-----------

.. code-block:: python

   from nlsq import CurveFit
   import jax.numpy as jnp

   # Create instance
   fitter = CurveFit()


   # Define model
   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Fit multiple datasets
   for x, y in datasets:
       popt, pcov = fitter.curve_fit(model, x, y, p0=[2, 0.5, 0])
       process_results(popt, pcov)

Constructor Options
-------------------

.. code-block:: python

   fitter = CurveFit(
       # Diagnostics
       enable_diagnostics=True,  # Collect convergence metrics
       enable_stability=True,  # Enable stability checks
       # Caching
       enable_caching=True,  # JIT compilation caching
       # Recovery
       enable_recovery=True,  # Auto-recovery from issues
   )

curve_fit Method
----------------

Full method signature:

.. code-block:: python

   popt, pcov = fitter.curve_fit(
       f,  # Model function
       xdata,  # Independent variable
       ydata,  # Dependent variable
       p0=None,  # Initial guess
       sigma=None,  # Uncertainties
       absolute_sigma=False,  # Absolute sigma interpretation
       check_finite=True,  # Check for inf/NaN
       bounds=(-np.inf, np.inf),  # Parameter bounds
       method="trf",  # Optimization method
       jac="2-point",  # Jacobian computation
       full_output=False,  # Return full result
       nan_policy="raise",  # NaN handling
       **kwargs  # Additional options
   )

Advanced Options
----------------

**Multi-start optimization:**

.. code-block:: python

   popt, pcov = fitter.curve_fit(
       model,
       x,
       y,
       p0=[...],
       multistart=True,
       n_starts=20,
       bounds=bounds,  # Required for multi-start
   )

**Stability checks:**

.. code-block:: python

   popt, pcov = fitter.curve_fit(
       model,
       x,
       y,
       p0=[...],
       stability="auto",  # 'auto', 'strict', 'none'
       fallback=True,  # Enable fallback strategies
       rescale_data=True,  # Automatic data rescaling
   )

**Full output:**

.. code-block:: python

   popt, pcov, info = fitter.curve_fit(model, x, y, p0=[...], full_output=True)

   print(f"Iterations: {info.nfev}")
   print(f"Final cost: {info.cost}")
   print(f"Status: {info.message}")

Batch Processing Pattern
------------------------

.. code-block:: python

   from nlsq import CurveFit
   import jax.numpy as jnp


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   # Create once
   fitter = CurveFit(enable_diagnostics=True)

   # Process many datasets
   results = []
   for data_file in data_files:
       x, y = load_data(data_file)
       popt, pcov = fitter.curve_fit(model, x, y, p0=[2, 0.5])
       results.append(
           {"file": data_file, "params": popt, "errors": np.sqrt(np.diag(pcov))}
       )

   # First fit is slow (JIT), rest are fast

Accessing Diagnostics
---------------------

.. code-block:: python

   fitter = CurveFit(enable_diagnostics=True)
   popt, pcov = fitter.curve_fit(model, x, y, p0=[...])

   # Access internal state (implementation-dependent)
   # Note: API may change between versions

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import CurveFit


   # Model
   def gaussian(x, amplitude, center, width, offset):
       return amplitude * jnp.exp(-0.5 * ((x - center) / width) ** 2) + offset


   # Create fitter
   fitter = CurveFit(enable_diagnostics=True)

   # Generate test datasets
   np.random.seed(42)
   datasets = []
   for i in range(10):
       x = np.linspace(0, 10, 100)
       # Varying parameters
       amp = 2 + i * 0.5
       ctr = 3 + i * 0.3
       y = amp * np.exp(-0.5 * ((x - ctr) / 1.0) ** 2) + 0.5
       y += 0.2 * np.random.randn(len(x))
       datasets.append((x, y))

   # Fit all datasets
   print("Fitting datasets...")
   for i, (x, y) in enumerate(datasets):
       popt, pcov = fitter.curve_fit(
           gaussian, x, y, p0=[2, 5, 1, 0.5], bounds=([0, 0, 0.1, 0], [10, 10, 5, 2])
       )
       perr = np.sqrt(np.diag(pcov))
       print(
           f"Dataset {i}: amp={popt[0]:.2f}+/-{perr[0]:.2f}, "
           f"ctr={popt[1]:.2f}+/-{perr[1]:.2f}"
       )

When to Use CurveFit vs fit()
-----------------------------

**Use fit() when:**

- Single fit
- Using workflow system
- Simple use case

**Use CurveFit when:**

- Multiple fits with same model
- Need JIT reuse for speed
- Accessing diagnostics
- Custom configuration

Next Steps
----------

- :doc:`least_squares` - Lower-level optimizer control
- :doc:`../factories_di/index` - Factory patterns

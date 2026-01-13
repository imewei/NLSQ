Common Issues
=============

This page covers the most frequent issues and their solutions.

Fit Doesn't Converge
--------------------

**Symptoms:**

- Warning about maximum iterations reached
- Parameters don't change
- Cost function stays high

**Solutions:**

1. **Better initial guess:**

   .. code-block:: python

      # Inspect data to estimate parameters
      import matplotlib.pyplot as plt

      plt.scatter(x, y)
      plt.show()

      # Set p0 based on visual inspection
      p0 = [y.max(), 0.5, y.min()]  # For exponential decay

2. **Add bounds:**

   .. code-block:: python

      bounds = ([0, 0, -10], [100, 10, 10])
      popt, pcov = fit(model, x, y, p0=p0, bounds=bounds)

3. **Use global optimization:**

   .. code-block:: python

      popt, pcov = fit(model, x, y, p0=p0, workflow="auto_global", bounds=bounds)

4. **Increase iterations:**

   .. code-block:: python

      popt, pcov = fit(model, x, y, p0=p0, max_nfev=5000)

Wrong Results
-------------

**Symptoms:**

- Fit looks wrong when plotted
- Parameters are physically unreasonable
- Multiple fits give different answers

**Solutions:**

1. **Check model function:**

   .. code-block:: python

      # Test model with known parameters
      y_test = model(x, 2.0, 0.5, 0.3)
      plt.plot(x, y_test)  # Should look like expected curve

2. **Verify JAX usage:**

   .. code-block:: python

      # Wrong
      import numpy as np


      def model(x, a, b):
          return a * np.exp(-b * x)  # np won't work


      # Correct
      import jax.numpy as jnp


      def model(x, a, b):
          return a * jnp.exp(-b * x)  # jnp works

3. **Check data:**

   .. code-block:: python

      # Look for outliers, missing data, wrong units
      plt.scatter(x, y)
      print(f"x range: {x.min()} to {x.max()}")
      print(f"y range: {y.min()} to {y.max()}")
      print(f"NaN values: {np.isnan(y).sum()}")

4. **Try global optimization:**

   .. code-block:: python

      # Local fit may find wrong minimum
      popt, pcov = fit(model, x, y, p0=p0, workflow="auto_global", bounds=bounds)

Covariance Cannot Be Estimated
------------------------------

**Symptoms:**

- ``pcov`` contains ``inf`` values
- Warning about covariance estimation

**Causes and solutions:**

1. **Poor fit:**

   The model doesn't describe the data. Try a different model.

2. **Parameters at bounds:**

   .. code-block:: python

      # Check if any parameters are at bounds
      print(f"popt: {popt}")
      print(f"bounds: {bounds}")
      # Widen bounds if parameters are constrained

3. **Parameter correlation:**

   Parameters may be unidentifiable. Try:

   - Reducing model complexity
   - Adding more data
   - Fixing some parameters

4. **Numerical issues:**

   .. code-block:: python

      # Enable stability checks
      popt, pcov = fit(model, x, y, p0=p0, stability="auto", rescale_data=True)

Memory Errors
-------------

**Symptoms:**

- "Out of memory" error
- System becomes unresponsive
- Process killed

**Solutions:**

1. **Limit memory usage:**

   .. code-block:: python

      popt, pcov = fit(model, x, y, p0=p0, memory_limit_gb=4.0)

2. **Use float32:**

   .. code-block:: python

      x = x.astype(np.float32)
      y = y.astype(np.float32)

3. **Test on subset first:**

   .. code-block:: python

      # Fit subset to check model
      n_sample = 10000
      idx = np.random.choice(len(x), n_sample)
      popt_test, _ = fit(model, x[idx], y[idx], p0=p0)

      # Full fit with good initial guess
      popt, pcov = fit(model, x, y, p0=popt_test)

4. **GPU memory:**

   .. code-block:: python

      import os

      os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
      os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

Slow Performance
----------------

**Symptoms:**

- Fit takes minutes for small datasets
- Progress seems stuck

**Solutions:**

1. **Use GPU:**

   .. code-block:: bash

      pip install "jax[cuda12_pip]"

2. **Loosen tolerances:**

   .. code-block:: python

      popt, pcov = fit(model, x, y, p0=p0, ftol=1e-6, xtol=1e-6, gtol=1e-6)

3. **Reduce n_starts:**

   .. code-block:: python

      popt, pcov = fit(
          model, x, y, p0=p0, workflow="auto_global", bounds=bounds, n_starts=5
      )  # Instead of 20

4. **JIT compilation overhead:**

   First fit is slow due to JIT compilation. Subsequent fits are faster.

Import Errors
-------------

**"No module named nlsq":**

.. code-block:: bash

   pip install nlsq

**"No module named jax":**

.. code-block:: bash

   pip install jax jaxlib

**"PySide6 not found" (for GUI):**

.. code-block:: bash

   pip install "nlsq[gui_qt]"

Model Function Errors
---------------------

**"TracerArrayConversionError":**

Using Python conditionals with JAX arrays:

.. code-block:: python

   # Wrong - Python if with JAX array
   def model(x, a, b):
       if b < 0:  # Python if doesn't work
           return a * x
       return a * jnp.exp(-b * x)


   # Correct - Use jnp.where
   def model(x, a, b):
       return jnp.where(b < 0, a * x, a * jnp.exp(-b * x))

**"ConcretizationTypeError":**

Shape depends on value:

.. code-block:: python

   # Wrong - shape depends on value
   def model(x, a, n):
       return a * x[: int(n)]  # Can't slice with traced value


   # Correct - fixed shapes
   def model(x, a, b):
       return a * x + b

Next Steps
----------

- :doc:`getting_help` - Get additional support
- :doc:`/reference/index` - API documentation

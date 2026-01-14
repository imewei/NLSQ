Getting Help
============

If you can't solve an issue, here's how to get help.

Documentation Resources
-----------------------

1. **Tutorials** (this guide): Step-by-step learning
2. **How-To Guides**: :doc:`/howto/index` - Task-oriented solutions
3. **Reference**: :doc:`/reference/index` - API documentation
4. **Explanation**: :doc:`/explanation/index` - Concept deep-dives

GitHub Issues
-------------

For bugs or feature requests:

1. Go to https://github.com/imewei/NLSQ/issues
2. Search existing issues first
3. Create a new issue with:

   - NLSQ version (``nlsq.__version__``)
   - Python version
   - Operating system
   - Minimal reproducible example
   - Error message (full traceback)

**Good issue template:**

.. code-block:: text

   **Environment:**
   - NLSQ version: 0.6.4
   - Python: 3.12.0
   - OS: Ubuntu 22.04
   - JAX backend: gpu

   **Description:**
   Fit fails with large dataset.

   **Minimal example:**
   ```python
   from nlsq import fit
   import jax.numpy as jnp
   import numpy as np


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   x = np.linspace(0, 10, 1000000)
   y = 2 * np.exp(-0.5 * x) + np.random.randn(len(x)) * 0.1

   popt, pcov = fit(model, x, y, p0=[2, 0.5])
   ```

   **Error:**
   ```
   [Full traceback here]
   ```

   **Expected behavior:**
   Fit should complete successfully.

Creating Minimal Examples
-------------------------

A good minimal example:

1. **Runs standalone**: No external data files
2. **Reproduces the issue**: Shows the exact problem
3. **Is minimal**: Removes unnecessary complexity

.. code-block:: python

   # Good minimal example
   from nlsq import fit
   import jax.numpy as jnp
   import numpy as np


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y = 2 * np.exp(-0.5 * x) + 0.1 * np.random.randn(len(x))

   # This fails with [error message]
   popt, pcov = fit(model, x, y, p0=[1, 1])

Debugging Tips
--------------

**Print intermediate values:**

.. code-block:: python

   # Check data
   print(f"x: min={x.min()}, max={x.max()}, shape={x.shape}")
   print(f"y: min={y.min()}, max={y.max()}, shape={y.shape}")
   print(f"NaN in y: {np.isnan(y).sum()}")

   # Check model
   y_test = model(x, 2.0, 0.5)
   print(f"Model output: {y_test[:5]}")

**Enable debug logging:**

.. code-block:: python

   import os

   os.environ["NLSQ_DEBUG"] = "1"

   from nlsq import fit

   # ... will print debug information

**Check JAX configuration:**

.. code-block:: python

   import jax

   print(f"JAX version: {jax.__version__}")
   print(f"Devices: {jax.devices()}")
   print(f"Backend: {jax.default_backend()}")

Version Information
-------------------

Include this in bug reports:

.. code-block:: python

   import nlsq
   import jax
   import numpy as np
   import scipy
   import sys

   print(f"NLSQ: {nlsq.__version__}")
   print(f"JAX: {jax.__version__}")
   print(f"NumPy: {np.__version__}")
   print(f"SciPy: {scipy.__version__}")
   print(f"Python: {sys.version}")
   print(f"Platform: {sys.platform}")

Common Solutions Summary
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Problem
     - Quick Fix
   * - Fit doesn't converge
     - Better p0, add bounds, use auto_global
   * - Wrong results
     - Check model uses jnp, plot data
   * - Memory error
     - Set memory_limit_gb, use float32
   * - Slow
     - Use GPU, loosen tolerances
   * - Covariance inf
     - Check fit quality, widen bounds

Next Steps
----------

- :doc:`common_issues` - Detailed solutions
- :doc:`/howto/troubleshooting` - More troubleshooting guides

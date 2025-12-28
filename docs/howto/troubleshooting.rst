Troubleshooting Guide
=====================

This guide helps you diagnose and fix common issues when using NLSQ.

Table of Contents
-----------------

1. `Installation Issues <#installation-issues>`__
2. `GPU/TPU Issues <#gputpu-issues>`__
3. `Convergence Problems <#convergence-problems>`__
4. `Performance Issues <#performance-issues>`__
5. `Memory Issues <#memory-issues>`__
6. `Numerical Stability Issues <#numerical-stability-issues>`__
7. `API and Usage Errors <#api-and-usage-errors>`__
8. `JAX-Specific Issues <#jax-specific-issues>`__

--------------

Installation Issues
-------------------

Issue: ``ModuleNotFoundError: No module named 'nlsq'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** NLSQ not installed

**Solution:**

.. code:: bash

   pip install nlsq

   # Or for development:
   git clone https://github.com/imewei/NLSQ.git
   cd NLSQ
   pip install -e .

**Verify installation:**

.. code:: python

   import nlsq

   print(nlsq.__version__)

Issue: ``ImportError: JAX requires NumPy >= 1.21``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Incompatible NumPy version

**Solution:**

.. code:: bash

   pip install --upgrade numpy>=1.21
   pip install --upgrade jax jaxlib

Issue: CUDA version mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:**

::

   RuntimeError: jaxlib version 0.4.1 is newer than and incompatible with jax version 0.3.25

**Solution:**

.. code:: bash

   # Uninstall and reinstall with matching versions
   pip uninstall jax jaxlib
   pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Or for CPU only:
   pip install "jax[cpu]"

**Check CUDA version:**

.. code:: bash

   nvidia-smi  # Look for CUDA Version
   nvcc --version

--------------

GPU/TPU Issues
--------------

Issue: No GPU detected (using CPU instead)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

.. code:: python

   import jax

   print(jax.devices())
   # Output: [CpuDevice(id=0)]  # Should be GpuDevice

**Solutions:**

1. **Check GPU availability:**

.. code:: bash

   nvidia-smi  # Should show GPU info

2. **Reinstall JAX with CUDA support:**

.. code:: bash

   # For CUDA 12.x
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # For CUDA 11.x
   pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

3. **Check CUDA environment variables:**

.. code:: bash

   echo $CUDA_HOME
   echo $LD_LIBRARY_PATH

4. **Verify JAX can see CUDA:**

.. code:: python

   import jax

   print(jax.local_devices())  # Should include GPU

Issue: GPU out of memory (OOM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:**

::

   RuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1234567890 bytes

**Solutions:**

1. **Limit memory preallocation:**

.. code:: python

   import os

   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"  # Use 75% of GPU

   from nlsq import curve_fit

2. **Use chunking for large datasets:**

.. code:: python

   from nlsq.streaming.large_dataset import fit_large_dataset

   popt, pcov, info = fit_large_dataset(
       model,
       x,
       y,
       p0=[2, 1],
       memory_limit_gb=4.0,  # Limit GPU memory usage
       chunk_size=100_000,
   )

3. **Use memory-efficient solver:**

.. code:: python

   popt, pcov = curve_fit(
       model, x, y, p0=[2, 1], solver="cg"  # More memory efficient than 'svd'
   )

4. **Clear GPU cache:**

.. code:: python

   import jax

   jax.clear_backends()  # Release all GPU memory

Issue: GPU slower than CPU
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Dataset too small (JIT overhead dominates)

**Solution:**

.. code:: python

   import os

   # For datasets < 10K points, use CPU
   if len(x) < 10000:
       os.environ["JAX_PLATFORM_NAME"] = "cpu"

   from nlsq import curve_fit

**Or benchmark both:**

.. code:: python

   import time

   # CPU timing
   os.environ["JAX_PLATFORM_NAME"] = "cpu"
   start = time.time()
   popt_cpu, _ = curve_fit(model, x, y, p0=[2, 1])
   cpu_time = time.time() - start

   # GPU timing
   os.environ["JAX_PLATFORM_NAME"] = "gpu"
   start = time.time()
   popt_gpu, _ = curve_fit(model, x, y, p0=[2, 1])
   gpu_time = time.time() - start

   print(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s")

--------------

Convergence Problems
--------------------

Issue: ``RuntimeError: Optimal parameters not found``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Optimization failed to converge

**Diagnosis:**

.. code:: python

   try:
       popt, pcov = curve_fit(model, x, y, p0=[2, 1])
   except RuntimeError as e:
       print(f"Error: {e}")
       # Get more details
       result = curve_fit(model, x, y, p0=[2, 1], full_output=True)
       print(f"Status: {result.status}")
       print(f"Message: {result.message}")

**Solutions:**

1. **Improve initial guess (``p0``):**

.. code:: python

   # Bad: p0 far from solution
   popt, pcov = curve_fit(model, x, y, p0=[100, 0.001])  # May fail

   # Good: p0 closer to expected values
   popt, pcov = curve_fit(model, x, y, p0=[2, 1])  # More likely to succeed

2. **Set realistic bounds:**

.. code:: python

   popt, pcov = curve_fit(
       model, x, y, p0=[2, 1], bounds=([0, 0], [10, 5])  # Constrain search space
   )

3. **Increase tolerance:**

.. code:: python

   popt, pcov = curve_fit(
       model,
       x,
       y,
       p0=[2, 1],
       ftol=1e-6,  # Default: 1e-8 (looser tolerance)
       xtol=1e-6,
       gtol=1e-6,
   )

4. **Increase max iterations:**

.. code:: python

   popt, pcov = curve_fit(
       model, x, y, p0=[2, 1], max_nfev=10000  # Default: 100 * (n_params + 1)
   )

5. **Scale your data:**

.. code:: python

   # Bad: x in [0, 1e6], y in [1e-10, 1e-8]
   popt, pcov = curve_fit(model, x, y, p0=[2, 1])  # May fail

   # Good: Scale to reasonable ranges
   x_scaled = x / 1e6  # Now in [0, 1]
   y_scaled = y * 1e10  # Now in [1, 100]
   popt, pcov = curve_fit(model, x_scaled, y_scaled, p0=[2, 1])

   # Unscale results
   popt[0] = popt[0] / 1e10  # Unscale amplitude parameter

Issue: Fit converges but results are wrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Local minimum or poor initial guess

**Solutions:**

1. **Try multiple initial guesses:**

.. code:: python

   import numpy as np

   p0_guesses = [[1, 0.5], [2, 1.0], [5, 2.0], [10, 0.1]]

   best_cost = np.inf
   best_popt = None

   for p0 in p0_guesses:
       try:
           popt, pcov = curve_fit(model, x, y, p0=p0)
           cost = np.sum((y - model(x, *popt)) ** 2)
           if cost < best_cost:
               best_cost = cost
               best_popt = popt
       except RuntimeError:
           continue

   print(f"Best fit: {best_popt}, cost: {best_cost}")

2. **Visualize the fit:**

.. code:: python

   import matplotlib.pyplot as plt

   popt, pcov = curve_fit(model, x, y, p0=[2, 1])

   plt.figure(figsize=(10, 4))

   # Plot 1: Data and fit
   plt.subplot(1, 2, 1)
   plt.plot(x, y, "o", label="Data")
   plt.plot(x, model(x, *popt), "-", label="Fit")
   plt.legend()

   # Plot 2: Residuals
   plt.subplot(1, 2, 2)
   residuals = y - model(x, *popt)
   plt.plot(x, residuals, "o")
   plt.axhline(0, color="r", linestyle="--")
   plt.ylabel("Residuals")

   plt.tight_layout()
   plt.show()

Issue: Covariance matrix has ``inf`` or ``nan``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Jacobian is singular or near-singular

**Solutions:**

1. **Check parameter identifiability:**

.. code:: python

   # Some parameters may not be identifiable from data
   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # If data doesn't cover x=0, 'a+c' not separately identifiable

   # Solution: Fix one parameter or add constraints
   popt, pcov = curve_fit(lambda x, a, b: model(x, a, b, c=0.5), x, y, p0=[2, 1])  # Fix c

2. **Add regularization via bounds:**

.. code:: python

   popt, pcov = curve_fit(
       model,
       x,
       y,
       p0=[2, 1, 0.5],
       bounds=([0, 0, 0], [10, 5, 2]),  # Prevent singular solutions
   )

3. **Check for redundant parameters:**

.. code:: python

   # Bad: Parameters are correlated
   def model(x, a, b, c, d):
       return a * jnp.exp(-b * x) + c * jnp.exp(-d * x)


   # If b ≈ d, parameters are redundant


   # Good: Use fewer parameters
   def model(x, a, b):
       return a * jnp.exp(-b * x)

--------------

Performance Issues
------------------

Issue: First fit is very slow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** JIT compilation overhead

**Solution:** This is expected. Subsequent fits will be much faster.

.. code:: python

   from nlsq import CurveFit

   # Create reusable fitter
   fitter = CurveFit()

   # First call: slow (compilation + execution)
   popt1, pcov1 = fitter.curve_fit(model, x1, y1, p0=[2, 1])  # ~500ms

   # Subsequent calls: fast (execution only)
   popt2, pcov2 = fitter.curve_fit(model, x2, y2, p0=[2, 1])  # ~30ms
   popt3, pcov3 = fitter.curve_fit(model, x3, y3, p0=[2, 1])  # ~30ms

Issue: Fit is slower than SciPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Diagnosis:**

1. **Check dataset size:**

.. code:: python

   print(f"Dataset size: {len(x)}")

If ``< 1000`` points: NLSQ overhead may not be worth it. Use SciPy.

2. **Check if GPU is being used:**

.. code:: python

   import jax

   print(f"Devices: {jax.devices()}")

3. **Benchmark with timing:**

.. code:: python

   popt, pcov, res, post_time, compile_time = curve_fit(
       model, x, y, p0=[2, 1], timeit=True
   )

   print(f"Compile time: {compile_time:.3f}s")
   print(f"Execution time: {post_time:.3f}s")

**Solutions:**

-  Use GPU for datasets > 10K points
-  Use ``CurveFit`` class for multiple fits
-  See :doc:`optimize_performance`

Issue: Memory usage keeps growing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** JIT cache growing or memory not being released

**Solutions:**

1. **Clear JIT cache periodically:**

.. code:: python

   import jax

   # After many fits
   jax.clear_caches()

2. **Disable JIT caching (not recommended):**

.. code:: python

   from jax import config

   config.update("jax_compilation_cache_dir", "")

3. **Use chunking for large datasets:**

.. code:: python

   from nlsq.streaming.large_dataset import fit_large_dataset

   popt, pcov, info = fit_large_dataset(model, x_large, y_large, memory_limit_gb=4.0)

--------------

Memory Issues
-------------

Issue: ``MemoryError`` during fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions:**

1. **Use chunking:**

.. code:: python

   from nlsq.streaming.large_dataset import fit_large_dataset

   popt, pcov, info = fit_large_dataset(
       model,
       x,
       y,
       p0=[2, 1],
       chunk_size=100_000,  # Process 100K points at a time
       memory_limit_gb=4.0,
   )

2. **Use minibatch solver:**

.. code:: python

   popt, pcov = curve_fit(model, x, y, p0=[2, 1], solver="minibatch", batch_size=50_000)

3. **Reduce precision (not recommended for production):**

.. code:: python

   from jax import config

   config.update("jax_enable_x64", False)  # Use float32 instead of float64

--------------

Numerical Stability Issues
--------------------------

Issue: ``RuntimeError: NaN or Inf encountered``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Causes:** - Overflow/underflow in model function - Division by zero -
Log of negative number

**Solutions:**

1. **Add numerical safeguards:**

.. code:: python

   import jax.numpy as jnp


   # Bad: Can overflow or divide by zero
   def model(x, a, b):
       return a / (1 + jnp.exp(-b * x))


   # Good: Add safeguards
   def model(x, a, b):
       # Clip to prevent overflow
       z = jnp.clip(-b * x, -100, 100)
       return a / (1 + jnp.exp(z))

2. **Use stable numerical functions:**

.. code:: python

   # Bad: log(exp(x)) can overflow
   result = jnp.log(jnp.exp(x))

   # Good: Use logsumexp
   result = x  # Equivalent but stable

3. **Check input data:**

.. code:: python

   import numpy as np

   # Check for inf/nan
   assert np.all(np.isfinite(x))
   assert np.all(np.isfinite(y))

   # Check for very large/small values
   print(f"x range: [{x.min()}, {x.max()}]")
   print(f"y range: [{y.min()}, {y.max()}]")

Issue: Ill-conditioned Jacobian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** - Large uncertainty estimates - Covariance matrix has very
large or very small values - Warning: “Covariance cannot be estimated”

**Solutions:**

1. **Scale parameters:**

.. code:: python

   # Bad: Parameters have very different scales
   def model(x, a, b):
       return a * jnp.exp(-b * x)  # a ~ 1e6, b ~ 1e-6


   # Good: Rescale inside model
   def model(x, a_scaled, b_scaled):
       a = a_scaled * 1e6
       b = b_scaled * 1e-6
       return a * jnp.exp(-b * x)


   # Fit with scaled parameters
   popt_scaled, pcov = curve_fit(model, x, y, p0=[1, 1])

   # Unscale results
   a_fit = popt_scaled[0] * 1e6
   b_fit = popt_scaled[1] * 1e-6

2. **Use parameter scaling:**

.. code:: python

   popt, pcov = curve_fit(
       model, x, y, p0=[2, 1], x_scale="jac"  # Automatic parameter scaling
   )

3. **Check condition number:**

.. code:: python

   from nlsq.diagnostics import check_condition_number

   result = curve_fit(model, x, y, p0=[2, 1])
   cond = check_condition_number(result.jac)

   if cond > 1e10:
       print(f"Warning: Ill-conditioned (κ = {cond:.2e})")

--------------

API and Usage Errors
--------------------

Issue: ``TypeError: curve_fit() got an unexpected keyword argument``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Using SciPy-specific arguments in NLSQ

**Solution:**

.. code:: python

   # SciPy-only arguments (not supported in NLSQ):
   # - full_output (use return_eval=True instead)
   # - epsfcn, factor, diag (LM-specific)

   # NLSQ equivalent:
   popt, pcov = curve_fit(model, x, y, return_eval=False)  # Instead of full_output=False

Issue: ``ValueError: p0 must be a 1-D array``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Incorrect p0 format

**Solutions:**

.. code:: python

   # Bad
   popt, pcov = curve_fit(model, x, y, p0=[[2, 1]])  # 2D array

   # Good
   popt, pcov = curve_fit(model, x, y, p0=[2, 1])  # 1D array or list

Issue: ``ValueError: Residuals are not finite``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Model returns inf/nan

**Debug:**

.. code:: python

   # Test model manually
   p_test = [2, 1]
   y_model = model(x, *p_test)
   print(f"Model output finite: {np.all(np.isfinite(y_model))}")

   # Check for specific issues
   print(f"Contains NaN: {np.any(np.isnan(y_model))}")
   print(f"Contains Inf: {np.any(np.isinf(y_model))}")

--------------

JAX-Specific Issues
-------------------

Issue: ``TypeError: jax.numpy function called with non-jax array``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Mixing NumPy and JAX arrays incorrectly

**Solution:**

.. code:: python

   import numpy as np
   import jax.numpy as jnp


   # Model function: use jnp
   def model(x, a, b):
       return a * jnp.exp(-b * x)  # jnp


   # Data generation: use np
   x = np.linspace(0, 5, 100)  # np is fine for data
   y = model(x, 2.5, 1.3)  # JAX auto-converts

   # Fitting: works with both
   popt, pcov = curve_fit(model, x, y, p0=[2, 1])

Issue: ``ConcretizationTypeError: Abstract tracer value``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Using Python control flow in JIT-compiled function

**Problem:**

.. code:: python

   # Bad: Python if statement (not JIT-compatible)
   def model(x, a, b, c):
       if a > 0:  # Error!
           return a * jnp.exp(-b * x)
       else:
           return c

**Solution:**

.. code:: python

   # Good: Use JAX control flow
   def model(x, a, b, c):
       return jnp.where(a > 0, a * jnp.exp(-b * x), c)

Issue: ``TracerBoolConversionError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Using array values in boolean context

**Problem:**

.. code:: python

   # Bad
   def model(x, a, b):
       if x[0] > 5:  # Error! Can't convert traced array to bool
           return a * x
       return b * x

**Solution:**

.. code:: python

   # Good
   def model(x, a, b):
       return jnp.where(x > 5, a * x, b * x)

--------------

Diagnostic Flowchart
--------------------

::

   Fit fails?
     │
     ├─> ImportError/ModuleNotFoundError ──> Installation Issues
     │
     ├─> RuntimeError: "not found" ─────────> Convergence Problems
     │
     ├─> MemoryError/OOM ───────────────────> Memory Issues
     │
     ├─> Slow performance ──────────────────> Performance Issues
     │
     ├─> NaN/Inf in results ────────────────> Numerical Stability
     │
     ├─> TypeError (JAX/tracing) ───────────> JAX-Specific Issues
     │
     └─> Other ─────────────────────────────> API and Usage Errors

--------------

Getting Help
------------

If this guide doesn’t resolve your issue:

1. **Check documentation:**

   -  :doc:`../api/index`
   -  :doc:`../reference/configuration`
   -  :doc:`advanced_api`
   -  :doc:`optimize_performance`

2. **Search GitHub issues:**

   -  https://github.com/imewei/NLSQ/issues

3. **Create minimal reproducible example:**

.. code:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit

   # Minimal data
   x = np.linspace(0, 5, 50)
   y = 2.5 * np.exp(-1.3 * x) + 0.1 * np.random.randn(50)


   # Minimal model
   def model(x, a, b):
       return a * jnp.exp(-b * x)


   # Your issue
   popt, pcov = curve_fit(model, x, y, p0=[2, 1])  # Describe problem here

4. **Report bug with:**

   -  Python version
   -  JAX version (``jax.__version__``)
   -  NLSQ version (``nlsq.__version__``)
   -  GPU info (if applicable)
   -  Minimal reproducible code
   -  Full error traceback

--------------

Interactive Notebooks
---------------------

Hands-on tutorials for debugging and troubleshooting:

- `Troubleshooting Guide <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/03_advanced/troubleshooting_guide.ipynb>`_ (25 min) - Debug convergence issues and common problems
- `NLSQ Challenges <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/03_advanced/nlsq_challenges.ipynb>`_ (45 min) - Difficult optimization problems and solutions
- `Interpreting Diagnostics <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/06_streaming/04_interpreting_diagnostics.ipynb>`_ - Understanding streaming results and diagnostics

--------------

Quick Reference: Common Solutions
---------------------------------

==================== =================================
Problem              Quick Fix
==================== =================================
Import error         ``pip install nlsq``
No GPU               ``pip install "jax[cuda12_pip]"``
Out of memory        Use ``fit_large_dataset()``
Slow first fit       Use ``CurveFit()`` class
Convergence failure  Better ``p0``, set ``bounds``
NaN/Inf              Add numerical safeguards
Tracing error        Use ``jnp.where()`` not ``if``
Wrong results        Check ``p0``, visualize fit
Large covariance     Scale parameters
Slow with small data Force CPU mode
==================== =================================

--------------

**Last Updated:** 2025-10-07

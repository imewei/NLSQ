Core API Reference
==================

This page documents the main functions and classes for curve fitting.

Primary Functions
-----------------

fit()
~~~~~

.. autofunction:: nlsq.fit
   :no-index:

The ``fit()`` function is the recommended entry point. It provides a unified
interface with preset-based configuration.

**Presets:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Preset
     - Description
   * - ``"default"``
     - Standard configuration for most use cases
   * - ``"fast"``
     - Optimized for speed, lower precision
   * - ``"precise"``
     - Higher precision, more iterations
   * - ``"global"``
     - Global optimization to avoid local minima
   * - ``"streaming"``
     - For datasets that exceed memory

**Example:**

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   popt, pcov = fit(model, x_data, y_data, p0=[1.0, 0.5, 0.0])

curve_fit()
~~~~~~~~~~~

.. autofunction:: nlsq.curve_fit
   :no-index:

Drop-in replacement for ``scipy.optimize.curve_fit``. Use this when migrating
from SciPy or when you need precise control over optimization parameters.

**Key Parameters:**

- ``f``: Model function with signature ``f(x, *params)``
- ``xdata``: Independent variable data
- ``ydata``: Dependent variable data
- ``p0``: Initial parameter guesses
- ``sigma``: Measurement uncertainties
- ``bounds``: Parameter bounds as ``([lowers], [uppers])``
- ``max_nfev``: Maximum function evaluations
- ``gtol``, ``ftol``, ``xtol``: Convergence tolerances

**Example:**

.. code-block:: python

   from nlsq import curve_fit
   import jax.numpy as jnp


   def exponential(x, a, tau, c):
       return a * jnp.exp(-x / tau) + c


   popt, pcov = curve_fit(
       exponential, x_data, y_data, p0=[1.0, 10.0, 0.0], bounds=([0, 0, -1], [10, 100, 1])
   )

curve_fit_large()
~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.curve_fit_large
   :no-index:

Handles datasets that exceed GPU memory through automatic chunking.

**Key Parameters:**

- ``memory_limit_gb``: Maximum GPU memory to use (default: auto-detect)
- ``chunk_size``: Manual chunk size (overrides auto)
- All parameters from ``curve_fit()``

**Example:**

.. code-block:: python

   from nlsq import curve_fit_large

   # Fit 10 million points with automatic chunking
   popt, pcov = curve_fit_large(model, x_large, y_large, p0=p0, memory_limit_gb=8.0)

LeastSquares
~~~~~~~~~~~~

.. autoclass:: nlsq.LeastSquares
   :members: solve
   :no-index:

Low-level least squares solver class. Use this when you need full control over
the optimization process or are working with non-standard objective functions.

Classes
-------

CurveFit
~~~~~~~~

.. autoclass:: nlsq.CurveFit
   :members:
   :special-members: __init__
   :no-index:

Reusable curve fitting class that caches JIT compilation for repeated fits.

**Example:**

.. code-block:: python

   from nlsq import CurveFit

   fitter = CurveFit()

   # First call compiles (slower)
   result1 = fitter.curve_fit(model, x1, y1, p0=p0)

   # Subsequent calls use cached compilation (fast)
   result2 = fitter.curve_fit(model, x2, y2, p0=p0)
   result3 = fitter.curve_fit(model, x3, y3, p0=p0)

OptimizeResult
~~~~~~~~~~~~~~

.. autoclass:: nlsq.result.OptimizeResult
   :members:
   :no-index:

The result object returned by optimization functions.

**Key Attributes:**

- ``x``: Optimal parameters (same as ``popt`` in curve_fit)
- ``cost``: Final value of the cost function
- ``fun``: Final residuals
- ``jac``: Final Jacobian matrix
- ``nfev``: Number of function evaluations
- ``nit``: Number of iterations
- ``message``: Termination message
- ``success``: Whether optimization succeeded

Return Values
-------------

curve_fit Return Format
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   popt, pcov = curve_fit(model, x, y, p0=p0)

   # popt: Optimal parameter values (JAX array)
   # pcov: Covariance matrix of parameters (JAX array)

   # Standard errors
   perr = jnp.sqrt(jnp.diag(pcov))

fit Return Format
~~~~~~~~~~~~~~~~~

.. code-block:: python

   popt, pcov = fit(model, x, y, p0=p0)

   # Same as curve_fit

   # With full_output=True:
   popt, pcov, info = fit(model, x, y, p0=p0, full_output=True)

   # info contains:
   # - residuals: Final residuals
   # - jacobian: Final Jacobian
   # - nfev: Function evaluations
   # - nit: Iterations

Error Handling
--------------

NLSQ raises informative exceptions:

.. code-block:: python

   from nlsq import curve_fit
   from nlsq.exceptions import ConvergenceError, ValidationError

   try:
       popt, pcov = curve_fit(model, x, y, p0=p0)
   except ConvergenceError as e:
       print(f"Did not converge: {e}")
   except ValidationError as e:
       print(f"Invalid input: {e}")

See Also
--------

- :doc:`/tutorials/01_first_fit` - Getting started tutorial
- :doc:`/howto/migrate_from_scipy` - Migration from SciPy
- :doc:`large_data` - Large dataset APIs

Numerical Stability Guide
=========================

This guide explains NLSQ's numerical stability features and how to use them
effectively for challenging optimization problems.

Overview
--------

NLSQ provides numerical stability monitoring to flag problems that can cause
optimization divergence. The stability system handles:

- **NaN/Inf detection** in the data and Jacobian matrices
- **Condition number monitoring** for ill-conditioned problems
- **Parameter and data scale warnings** (no data or ``p0`` is ever rescaled)
- **SVD skip** for large Jacobians to avoid performance degradation

Stability Modes
---------------

The ``stability`` parameter in ``curve_fit()`` controls behavior:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Behavior
     - Use Case
   * - ``stability=False``
     - No stability checks (default)
     - Simple problems, maximum speed
   * - ``stability='check'``
     - Warn about issues
     - Debugging, identify problems
   * - ``stability='auto'``
     - Detect issues and warn; data and ``p0`` are never rescaled or modified
     - Production use, challenging problems

Basic Usage
-----------

Enable stability mode with a single parameter:

.. code-block:: python

   from nlsq import curve_fit
   import jax.numpy as jnp
   import numpy as np


   def exponential(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Data with challenging characteristics
   x = np.linspace(0, 1e6, 1000)  # Large x-range
   y = 2.5 * np.exp(-0.5 * x) + 1.0

   # Enable stability checks (issues are reported, data is not modified)
   popt, pcov = curve_fit(exponential, x, y, p0=[2.5, 0.5, 1.0], stability="auto")

Physics Applications
--------------------

For physics applications (XPCS, scattering, spectroscopy) the data keep their
physical units: ``stability='auto'`` never rescales ``xdata``, ``ydata`` or
``p0``. ``rescale_data=False`` only silences the warning that the data would
have qualified for rescaling:

.. code-block:: python

   from nlsq import curve_fit
   import jax.numpy as jnp


   def g2_model(tau, baseline, contrast, gamma):
       """XPCS intensity autocorrelation function."""
       return baseline + contrast * jnp.exp(-2 * gamma * tau) ** 2


   # Time delays in physical units (seconds)
   tau = np.logspace(-6, 1, 200)  # 1µs to 10s
   y = 1.0 + 0.3 * np.exp(-2 * 100 * tau) ** 2

   # Physical units are always preserved; rescale_data=False only
   # suppresses the "Data rescaling was NOT applied" warning
   popt, pcov = curve_fit(
       g2_model,
       tau,
       y,
       p0=[1.0, 0.3, 100.0],
       stability="auto",
       rescale_data=False,
   )

The fitted parameters (for example the decay rate ``gamma`` in s^-1) are
therefore always in the units of your model. If conditioning is a problem, pass
``x_scale='jac'`` or rescale the data yourself and interpret the fitted
parameters accordingly.

Large Jacobian Optimization
---------------------------

For large datasets (>10M Jacobian elements), SVD computation becomes expensive.
NLSQ automatically skips SVD for large Jacobians:

.. code-block:: python

   from nlsq import curve_fit

   # Large dataset: 10M points × 3 params = 30M Jacobian elements
   x_large = np.linspace(0, 100, 10_000_000)
   y_large = model(x_large, *true_params) + noise

   # SVD automatically skipped (>10M elements)
   popt, pcov = curve_fit(model, x_large, y_large, p0=p0, stability="auto")

   # Custom threshold
   popt, pcov = curve_fit(
       model,
       x_large,
       y_large,
       p0=p0,
       stability="auto",
       max_jacobian_elements_for_svd=5_000_000,  # Skip above 5M
   )

**What happens when SVD is skipped?**

- NaN/Inf checking is still performed (O(n) complexity)
- Condition number monitoring is disabled
- No regularization applied
- Optimization proceeds without stability overhead

Performance Impact
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Overhead
     - Notes
   * - ``stability=False``
     - 0
     - No stability checks
   * - ``stability='check'``
     - ~1ms for 1M points
     - Only monitoring, warnings only
   * - ``stability='auto'``
     - ~1-5ms
     - Full detection and warnings; no data is modified

**Per-iteration vs initialization-only:**

Prior to v0.3.0, stability checks ran per-iteration, causing optimization
divergence due to accumulated perturbations. Now stability checks run only
at initialization, reducing overhead and preventing divergence.

Configuration Options
---------------------

All stability-related parameters:

.. code-block:: python

   from nlsq import curve_fit

   popt, pcov = curve_fit(
       model,
       x,
       y,
       p0=p0,
       # Stability mode
       stability="auto",  # 'auto', 'check', or False
       # Rescale warning (no data is ever rescaled)
       rescale_data=True,  # Warn when data would have qualified (default)
       # SVD threshold
       max_jacobian_elements_for_svd=10_000_000,  # Skip SVD above this
   )

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Configure stability defaults via environment:

.. code-block:: bash

   # Disable persistent JAX cache
   export NLSQ_DISABLE_PERSISTENT_CACHE=1

   # Custom JAX cache directory. The standard JAX_COMPILATION_CACHE_DIR, if
   # set, takes precedence over this.
   export NLSQ_JAX_CACHE_DIR=/tmp/nlsq_cache

   # Minimum compilation time to cache
   export NLSQ_CACHE_MIN_COMPILE_TIME_SECS=2

Troubleshooting
---------------

Optimization Diverges
~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Cost increases, parameters explode, NaN in results

**Solutions**:

1. Enable stability mode:

   .. code-block:: python

      popt, pcov = curve_fit(model, x, y, p0=p0, stability="auto")

2. Check initial parameters:

   .. code-block:: python

      from nlsq.stability import check_problem_stability

      report = check_problem_stability(model, x, y, p0=p0)
      print(f"Condition number: {report['condition_number']:.2e}")

3. Use bounds to constrain parameters:

   .. code-block:: python

      popt, pcov = curve_fit(model, x, y, p0=p0, bounds=([0, 0, 0], [10, 10, 10]))

Slow Optimization
~~~~~~~~~~~~~~~~~

**Symptoms**: Optimization takes much longer than expected

**Solutions**:

1. Disable stability checks:

   .. code-block:: python

      popt, pcov = curve_fit(model, x, y, p0=p0, stability=False)

2. Lower SVD threshold for large datasets:

   .. code-block:: python

      popt, pcov = curve_fit(
          model, x, y, p0=p0, stability="auto", max_jacobian_elements_for_svd=1_000_000
      )

3. Use check mode instead of auto:

   .. code-block:: python

      popt, pcov = curve_fit(model, x, y, p0=p0, stability="check")

Ill-Conditioned Problems
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Large uncertainties, unstable parameter estimates

**Solutions**:

1. Rescale data:

   .. code-block:: python

      x_scaled = (x - x.mean()) / x.std()
      y_scaled = (y - y.mean()) / y.std()
      popt, pcov = curve_fit(model, x_scaled, y_scaled, p0=p0)

2. Use Jacobian-based parameter scaling (``stability='auto'`` only warns, it
   does not rescale):

   .. code-block:: python

      popt, pcov = curve_fit(model, x, y, p0=p0, x_scale="jac")

3. Add regularization via bounds:

   .. code-block:: python

      # Soft bounds prevent extreme parameters
      popt, pcov = curve_fit(
          model, x, y, p0=p0, bounds=([-1e10] * n_params, [1e10] * n_params)
      )

See Also
--------

- :doc:`../api/nlsq.stability` - Stability module API reference
- :doc:`../api/nlsq.validators` - Input validation
- :doc:`../howto/troubleshooting` - General troubleshooting guide
- :doc:`../howto/optimize_performance` - Performance optimization

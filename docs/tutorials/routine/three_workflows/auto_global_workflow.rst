workflow="auto_global" - Global Optimization
============================================

The ``auto_global`` workflow provides robust global optimization for problems
with multiple local minima or unknown initial parameters. It requires bounds
and automatically selects between Multi-Start and CMA-ES strategies.

When to Use
-----------

Use ``auto_global`` workflow when:

- You don't know reasonable initial parameter values
- Your model may have multiple local minima
- Previous fits converged to unexpected solutions
- You need robust, reproducible fitting

.. important::

   ``auto_global`` **requires bounds**. Bounds define the search space for
   global optimization.

Basic Usage
-----------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Global optimization with bounds
   popt, pcov = fit(
       model,
       xdata,
       ydata,
       p0=[1.0, 0.5, 0.0],  # Initial guess (optional but helpful)
       workflow="auto_global",
       bounds=([0, 0, -1], [10, 5, 1]),
   )  # Required!

How It Works
------------

``auto_global`` automatically selects the best global search strategy:

**Multi-Start Optimization** (default):

- Generates multiple starting points using Latin Hypercube Sampling
- Runs local optimization from each starting point
- Returns the best result

**CMA-ES** (Covariance Matrix Adaptation Evolution Strategy):

- Evolutionary algorithm for complex landscapes
- Selected automatically when parameter scales vary widely
- Requires optional ``evosax`` dependency

Controlling the Search
----------------------

**Number of starts:**

.. code-block:: python

   # More starts = more thorough search (slower)
   popt, pcov = fit(
       model, x, y, p0=[...], workflow="auto_global", bounds=bounds, n_starts=20
   )  # Default is 10

**Force CMA-ES:**

.. code-block:: python

   from nlsq.global_optimization import CMAESConfig

   config = CMAESConfig(n_generations=200)
   popt, pcov = fit(
       model,
       x,
       y,
       p0=[...],
       workflow="auto_global",
       bounds=bounds,
       method="cmaes",
       cmaes_config=config,
   )

Automatic Method Selection
--------------------------

NLSQ automatically chooses between Multi-Start and CMA-ES based on
parameter scale ratio:

.. code-block:: python

   scale_ratio = max(upper - lower) / min(upper - lower)

   if scale_ratio > 1000:
       method = "cmaes"  # Wide parameter ranges
   else:
       method = "multi-start"

This means parameters with very different scales (e.g., ``a`` ranges 0-1000
while ``b`` ranges 0-0.001) trigger CMA-ES automatically.

Memory Strategy
---------------

Like ``auto``, the ``auto_global`` workflow automatically handles memory:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Memory Strategy
     - Multi-Start
     - CMA-ES
   * - **STANDARD**
     - n_starts × parallel TRF
     - CMA-ES + TRF refinement
   * - **CHUNKED**
     - LargeDatasetFitter + multi-start
     - CMA-ES + chunked evaluation
   * - **STREAMING**
     - Streaming + multi-start
     - CMA-ES + data streaming

Complete Example
----------------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp
   import numpy as np


   # Model with multiple local minima
   def double_gaussian(x, a1, c1, w1, a2, c2, w2, offset):
       g1 = a1 * jnp.exp(-0.5 * ((x - c1) / w1) ** 2)
       g2 = a2 * jnp.exp(-0.5 * ((x - c2) / w2) ** 2)
       return g1 + g2 + offset


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 200)
   y_true = (
       3.0 * np.exp(-0.5 * ((x - 3) / 0.8) ** 2)
       + 2.0 * np.exp(-0.5 * ((x - 7) / 1.0) ** 2)
       + 0.2
   )
   y = y_true + 0.15 * np.random.normal(size=len(x))

   # Define bounds for 7 parameters
   lower = [0, 0, 0.1, 0, 0, 0.1, -1]
   upper = [10, 10, 3, 10, 10, 3, 1]

   # Global optimization
   popt, pcov = fit(
       double_gaussian,
       x,
       y,
       p0=[2, 3, 1, 2, 7, 1, 0],
       workflow="auto_global",
       bounds=(lower, upper),
       n_starts=15,
   )

   # Print results
   print("Fitted parameters:")
   names = ["a1", "c1", "w1", "a2", "c2", "w2", "offset"]
   true_vals = [3.0, 3.0, 0.8, 2.0, 7.0, 1.0, 0.2]
   for name, fitted, true in zip(names, popt, true_vals):
       print(f"  {name}: {fitted:.3f} (true: {true})")

Performance Considerations
--------------------------

**Speed**: ``auto_global`` is slower than ``auto`` because it:

- Evaluates the model at many starting points
- Runs multiple local optimizations
- May use evolutionary strategies (CMA-ES)

**Recommendations**:

- Start with ``n_starts=10`` (default)
- Increase if results are inconsistent
- Use tight bounds to reduce search space
- For production, consider caching results

Comparison: auto vs auto_global
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - ``auto``
     - ``auto_global``
   * - Initial guess dependency
     - High
     - Low
   * - Speed
     - Fast
     - Slower
   * - Bounds
     - Optional
     - Required
   * - Multiple minima
     - May miss global
     - Explores globally
   * - When to use
     - Good p0 known
     - Unknown landscape

Troubleshooting
---------------

**All starts converge to same (wrong) solution:**

- Widen the bounds
- Increase ``n_starts``
- Check if model is appropriate for data

**CMA-ES not available:**

.. code-block:: bash

   pip install evosax

**Slow performance:**

- Reduce ``n_starts``
- Narrow the bounds
- Use ``auto`` if you have a good initial guess

Next Steps
----------

- :doc:`hpc_workflow` - For long-running HPC jobs
- :doc:`../data_handling/bounds` - Setting appropriate bounds
- :doc:`../troubleshooting/common_issues` - Debugging global optimization

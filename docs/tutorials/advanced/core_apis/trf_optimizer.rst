TRF Optimizer
=============

The Trust Region Reflective (TRF) algorithm is NLSQ's core optimizer.

Algorithm Overview
------------------

TRF solves bounded nonlinear least squares problems:

.. code-block:: text

   minimize:  0.5 * ||f(x)||²
   subject to: lb ≤ x ≤ ub

The algorithm:

1. **Trust Region**: Approximate objective in a local region
2. **Reflective**: Handle bounds via reflection at boundaries
3. **Adaptive**: Adjust trust radius based on progress

Key Components
--------------

Located in ``nlsq/core/trf.py`` (2544 lines):

.. code-block:: python

   from nlsq.core.trf import TrustRegionReflective

   trf = TrustRegionReflective(
       fun=residual_func,
       x0=initial_params,
       lb=lower_bounds,
       ub=upper_bounds,
       f_scale=1.0,
       ftol=1e-8,
       xtol=1e-8,
       gtol=1e-8,
       max_nfev=100,
       tr_solver="exact",
       tr_options={},
   )

   result = trf.solve()

Iteration Steps
---------------

Each iteration performs:

1. **Gradient computation**: ``g = J^T @ f``
2. **Scaling**: Apply parameter scaling ``D``
3. **Subproblem**: Solve trust region subproblem
4. **Step computation**: Find step direction ``p``
5. **Reflection**: Handle bound violations
6. **Ratio evaluation**: ``ratio = actual / predicted``
7. **Trust update**: Adjust trust radius

.. code-block:: text

   ratio > 0.75  →  expand trust region (×2)
   ratio > 0.25  →  keep trust region
   ratio < 0.25  →  contract trust region (×0.25)

Trust Region Solvers
--------------------

**Exact (SVD-based):**

.. code-block:: python

   # For small/medium problems
   # Solves: min ||J*p + f||² s.t. ||D*p|| ≤ Δ

   tr_solver = "exact"  # Uses SVD decomposition

**LSMR (iterative):**

.. code-block:: python

   # For large problems where SVD is expensive
   tr_solver = "lsmr"
   tr_options = {"maxiter": 100, "atol": 1e-10, "btol": 1e-10}

JIT-Compiled Helpers
--------------------

Located in ``nlsq/core/trf_jit.py``:

.. code-block:: python

   from nlsq.core.trf_jit import (
       compute_gradient_jit,
       solve_lsq_trust_region_jit,
       minimize_quadratic_1d_jit,
   )

These functions are JIT-compiled for GPU acceleration.

Profiling
---------

Use ``TRFProfiler`` for timing:

.. code-block:: python

   from nlsq.core.profiler import TRFProfiler

   profiler = TRFProfiler()

   # Pass to optimizer
   result = optimizer.least_squares(
       fun=residuals, x0=x0, _profiler=profiler  # Internal option
   )

   # Get timing breakdown
   profiler.print_summary()

Convergence Criteria
--------------------

Optimization stops when any criterion is met:

.. code-block:: python

   # Function tolerance (relative cost reduction)
   # |Δcost| / cost < ftol
   ftol = 1e-8

   # Parameter tolerance (relative step size)
   # ||Δx|| / ||x|| < xtol
   xtol = 1e-8

   # Gradient tolerance (gradient norm)
   # ||g||_inf < gtol
   gtol = 1e-8

   # Maximum evaluations
   max_nfev = 100 * n_params

Status Codes
------------

.. list-table::
   :header-rows: 1
   :widths: 10 50 40

   * - Status
     - Meaning
     - Action
   * - 1
     - ftol satisfied
     - Success
   * - 2
     - xtol satisfied
     - Success
   * - 3
     - gtol satisfied
     - Success
   * - 0
     - max_nfev reached
     - Increase max_nfev
   * - -1
     - Improper input
     - Check parameters

Algorithm Tuning
----------------

**For fast convergence:**

.. code-block:: python

   # Start near solution
   p0 = good_initial_guess

   # Looser tolerances
   ftol = 1e-6
   xtol = 1e-6
   gtol = 1e-6

**For high precision:**

.. code-block:: python

   # Tight tolerances
   ftol = 1e-12
   xtol = 1e-12
   gtol = 1e-12

   # More iterations
   max_nfev = 1000

**For ill-conditioned problems:**

.. code-block:: python

   # Parameter scaling
   x_scale = "jac"  # or provide manual scaling

   # LSMR for stability
   tr_solver = "lsmr"

Next Steps
----------

- :doc:`result_types` - Understanding results
- :doc:`../performance/index` - Performance tuning

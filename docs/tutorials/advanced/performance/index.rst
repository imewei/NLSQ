Performance Optimization
========================

This chapter covers techniques for maximizing NLSQ performance.

.. toctree::
   :maxdepth: 1

   jit_caching
   memory_management
   sparse_jacobians
   profiling

Chapter Overview
----------------

**JIT Caching** (10 min)
   Understanding and optimizing JIT compilation caching.

**Memory Management** (10 min)
   Memory pools, budgets, and efficient data handling.

**Sparse Jacobians** (10 min)
   Leveraging sparsity for large problems.

**Profiling** (10 min)
   Using TRFProfiler and identifying bottlenecks.

Performance Quick Tips
----------------------

1. **Use GPU**: 10-20x speedup for large datasets
2. **Reuse CurveFit**: JIT cached between calls
3. **Use float32**: Half memory for very large data
4. **Loosen tolerances**: Trade precision for speed
5. **Profile first**: Identify actual bottlenecks

.. code-block:: python

   # Fast pattern
   fitter = CurveFit()  # Create once
   for x, y in datasets:
       popt, pcov = fitter.curve_fit(model, x, y)  # JIT reused

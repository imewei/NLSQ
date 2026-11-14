Performance Optimization
========================

This chapter covers techniques for maximizing NLSQ performance.

Performance Quick Tips
----------------------

1. **Use GPU for large datasets**: it was faster than the CPU from 1M points on the four models measured, depended on the model at 100K points, and was slower below that (:doc:`/api/performance_benchmarks`)
2. **Reuse CurveFit**: JIT cached between calls
3. **Use streaming**: Memory-efficient for very large data
4. **Loosen tolerances**: Trade precision for speed
5. **Profile first**: Identify actual bottlenecks

.. code-block:: python

   # Fast pattern
   fitter = CurveFit()  # Create once
   for x, y in datasets:
       popt, pcov = fitter.curve_fit(model, x, y)  # JIT reused

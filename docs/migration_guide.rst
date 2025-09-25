Migration Guide
===============

This guide helps users migrate to NLSQ's enhanced capabilities and take advantage of the new advanced features introduced in recent versions.

Migrating from Basic Usage
---------------------------

If you're currently using basic NLSQ functionality, you can gradually adopt advanced features without changing your existing code.

From Simple curve_fit
~~~~~~~~~~~~~~~~~~~~~

**Before (still works):**

.. code-block:: python

   from nlsq import curve_fit
   import jax.numpy as jnp


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.1])

**After (enhanced with advanced features):**

.. code-block:: python

   from nlsq import curve_fit_large  # Drop-in replacement with auto-optimization
   import jax.numpy as jnp


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   # Automatically handles large datasets and optimizes memory usage
   popt, pcov = curve_fit_large(
       model,
       x,
       y,
       p0=[1.0, 0.1],
       show_progress=True,  # Progress bar for large fits
       memory_limit_gb=8.0,  # Automatic memory management
   )

Migrating from Custom CurveFit Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before:**

.. code-block:: python

   from nlsq import CurveFit

   cf = CurveFit()
   popt, pcov = cf.curve_fit(model, x, y, p0=p0)

**After (with stability features):**

.. code-block:: python

   from nlsq import CurveFit, SmartCache

   # Add stability and recovery features
   cache = SmartCache()

   cf = CurveFit(enable_stability=True, enable_recovery=True)

   popt, pcov = cf.curve_fit(model, x, y, p0=p0)

   # Standard curve_fit returns popt and pcov
   print(f"Parameters: {popt}")
   print(f"Covariance: {pcov}")

Migrating Large Dataset Workflows
----------------------------------

If you were previously handling large datasets manually, NLSQ now provides automatic solutions.

From Manual Chunking
~~~~~~~~~~~~~~~~~~~~

**Before (manual approach):**

.. code-block:: python

   import numpy as np
   from nlsq import curve_fit

   # Manual chunking approach
   chunk_size = 100000
   n_chunks = len(x) // chunk_size + 1
   results = []

   for i in range(n_chunks):
       start = i * chunk_size
       end = min((i + 1) * chunk_size, len(x))

       if end - start < 1000:  # Skip tiny chunks
           continue

       x_chunk = x[start:end]
       y_chunk = y[start:end]

       popt_chunk, _ = curve_fit(model, x_chunk, y_chunk, p0=p0)
       results.append(popt_chunk)

   # Manual averaging of results
   popt = np.mean(results, axis=0)

**After (automatic chunking):**

.. code-block:: python

   from nlsq import curve_fit_large

   # Automatic intelligent chunking with <1% error
   popt, pcov = curve_fit_large(
       model,
       x,
       y,
       p0=p0,
       memory_limit_gb=8.0,
       show_progress=True,
       auto_size_detection=True,  # Automatically decide processing strategy
   )

   # Get detailed information about the chunking process
   print(f"Processing strategy automatically selected")
   print(f"Dataset size: {len(x):,} points")

From Memory Management Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before (manual memory management):**

.. code-block:: python

   import gc
   import psutil


   # Manual memory monitoring
   def check_memory():
       mem = psutil.virtual_memory()
       if mem.percent > 80:
           gc.collect()
           print("Triggered garbage collection")


   # Manual memory-safe fitting
   check_memory()
   popt, pcov = curve_fit(model, x, y, p0=p0)
   check_memory()

**After (automatic memory management):**

.. code-block:: python

   from nlsq import MemoryConfig, memory_context, curve_fit

   # Automatic memory management
   config = MemoryConfig(
       memory_limit_gb=8.0, enable_garbage_collection=True, enable_memory_monitoring=True
   )

   with memory_context(config):
       popt, pcov = curve_fit(model, x, y, p0=p0)
       # Memory automatically managed throughout the process

Adopting New Error Handling
----------------------------

Enhanced error handling and recovery mechanisms provide more robust optimization.

From Basic Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~

**Before:**

.. code-block:: python

   from nlsq import curve_fit
   import numpy as np

   try:
       popt, pcov = curve_fit(model, x, y, p0=p0)
   except Exception as e:
       print(f"Fit failed: {e}")
       # Manual retry with different parameters
       popt, pcov = curve_fit(model, x, y, p0=p0 * 1.1)

**After (automatic recovery):**

.. code-block:: python

   from nlsq import CurveFit, RecoveryManager, OptimizationError

   # Automatic recovery with multiple strategies
   recovery = RecoveryManager(
       enable_parameter_perturbation=True,
       enable_algorithm_switching=True,
       max_recovery_attempts=3,
   )

   cf = CurveFit(recovery_manager=recovery)

   try:
       result = cf.curve_fit_robust(model, x, y, p0=p0)
       print(f"Success after {result.recovery_stats.attempts} attempts")
       print(f"Final algorithm: {result.algorithm_used}")
       popt, pcov = result.popt, result.pcov

   except OptimizationError as e:
       print(f"Optimization failed after all recovery attempts: {e}")

Upgrading Algorithm Selection
-----------------------------

Move from manual algorithm selection to intelligent automatic selection.

From Manual Algorithm Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before:**

.. code-block:: python

   from nlsq import curve_fit

   # Manual algorithm testing
   algorithms = ["trf", "lm", "dogbox"]
   best_result = None
   best_residual = float("inf")

   for method in algorithms:
       try:
           popt, pcov = curve_fit(model, x, y, p0=p0, method=method)
           residual = np.sum((y - model(x, *popt)) ** 2)

           if residual < best_residual:
               best_residual = residual
               best_result = (popt, pcov, method)
       except:
           continue

   popt, pcov, best_method = best_result
   print(f"Best method: {best_method}")

**After (automatic selection):**

.. code-block:: python

   from nlsq.algorithm_selector import auto_select_algorithm
   from nlsq import curve_fit

   # Automatic algorithm selection based on problem characteristics
   recommendations = auto_select_algorithm(f=model, xdata=x, ydata=y, p0=p0)

   best_method = recommendations.get("algorithm", "trf")
   popt, pcov = curve_fit(model, x, y, p0=p0, method=best_method)

   print(f"Auto-selected: {best_method}")
   print(f"Fitted parameters: {popt}")

Adding Input Validation
-----------------------

Enhance reliability by adding comprehensive input validation.

From No Validation
~~~~~~~~~~~~~~~~~~

**Before:**

.. code-block:: python

   from nlsq import curve_fit

   # Direct fitting without validation
   popt, pcov = curve_fit(model, x, y, p0=p0)

**After (with validation):**

.. code-block:: python

   from nlsq import curve_fit, InputValidator, ValidationError

   # Add comprehensive input validation
   validator = InputValidator()

   try:
       warnings, errors, clean_x, clean_y = validator.validate_curve_fit_inputs(
           func=model, xdata=x, ydata=y, p0=p0, bounds=bounds
       )

       # Handle warnings (non-blocking)
       for warning in warnings:
           print(f"Warning: {warning}")

       # Check for blocking errors
       if errors:
           raise ValidationError(f"Validation failed: {errors}")

       # Use validated data
       popt, pcov = curve_fit(model, clean_x, clean_y, p0=p0)

   except ValidationError as e:
       print(f"Input validation failed: {e}")

Performance Optimization Migration
----------------------------------

Migrate to performance-optimized workflows for better speed and resource usage.

Adding Caching
~~~~~~~~~~~~~~

**Before (no caching):**

.. code-block:: python

   from nlsq import curve_fit

   # Multiple similar fits without caching
   results = []
   for dataset in datasets:
       x, y = dataset
       popt, pcov = curve_fit(model, x, y, p0=p0)  # Recompiles each time
       results.append(popt)

**After (with smart caching):**

.. code-block:: python

   from nlsq import CurveFit, SmartCache

   # Enable caching for repeated similar fits
   cache = SmartCache(
       enable_function_caching=True, enable_jacobian_caching=True, max_cache_size_gb=2.0
   )

   cf = CurveFit(cache=cache)
   results = []

   for dataset in datasets:
       x, y = dataset
       popt, pcov = cf.curve_fit(model, x, y, p0=p0)  # Uses cached compilation
       results.append(popt)

   # Check cache performance
   stats = cache.get_stats()
   print(f"Cache hit rate: {stats.hit_rate:.1%}")

Complete Migration Example
--------------------------

Here's a complete example showing migration from basic usage to full advanced features:

**Before (basic usage):**

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Basic fitting
   popt, pcov = curve_fit(model, x, y, p0=[1, 0.1, 0])
   print(f"Parameters: {popt}")

**After (with advanced features):**

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import CurveFit, curve_fit_large, MemoryConfig, memory_context, InputValidator
   from nlsq.algorithm_selector import auto_select_algorithm


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Configure memory management
   memory_config = MemoryConfig(memory_limit_gb=8.0)
   validator = InputValidator()

   # Fitting with advanced features
   try:
       # Input validation
       warnings, errors, clean_x, clean_y = validator.validate_curve_fit_inputs(
           f=model, xdata=x, ydata=y, p0=[1, 0.1, 0]
       )

       if not errors:
           # Use memory context
           with memory_context(memory_config):
               if len(x) > 1_000_000:  # Large dataset
                   popt, pcov = curve_fit_large(
                       model,
                       clean_x,
                       clean_y,
                       p0=[1, 0.1, 0],
                       show_progress=True,
                       memory_limit_gb=8.0,
                   )
               else:  # Regular dataset
                   # Auto-select algorithm
                   recommendations = auto_select_algorithm(
                       f=model, xdata=clean_x, ydata=clean_y, p0=[1, 0.1, 0]
                   )
                   method = recommendations.get("algorithm", "trf")

                   cf = CurveFit(enable_stability=True, enable_recovery=True)
                   popt, pcov = cf.curve_fit(
                       model, clean_x, clean_y, p0=[1, 0.1, 0], method=method
                   )

           print(f"Parameters: {popt}")
           print(f"Parameter errors: {np.sqrt(np.diag(pcov))}")

   except Exception as e:
       print(f"Optimization error: {e}")

Best Practices for Migration
----------------------------

1. **Gradual Adoption**: Start with ``curve_fit_large`` as a drop-in replacement
2. **Enable Monitoring**: Add diagnostic monitoring for production workflows
3. **Configure Memory**: Set appropriate memory limits for your system
4. **Use Validation**: Enable input validation for robust pipelines
5. **Add Caching**: Enable caching for repeated similar problems
6. **Configure Recovery**: Set up recovery strategies for critical applications

Migration Checklist
-------------------

- [ ] Replace ``curve_fit`` with ``curve_fit_large`` for large datasets
- [ ] Add memory configuration with ``MemoryConfig`` and ``memory_context``
- [ ] Enable algorithm selection with ``AlgorithmSelector``
- [ ] Add diagnostic monitoring with ``DiagnosticMonitor``
- [ ] Configure caching with ``SmartCache`` for repeated fits
- [ ] Add input validation with ``InputValidator``
- [ ] Set up recovery strategies with ``RecoveryManager``
- [ ] Update error handling to use new exception types
- [ ] Test performance improvements with new features
- [ ] Update documentation and comments to reflect new capabilities

Compatibility Notes
-------------------

- All existing code continues to work without changes
- New features are opt-in and don't affect existing workflows
- Performance improvements are automatic when using new functions
- Advanced features can be adopted incrementally
- No breaking changes to existing API

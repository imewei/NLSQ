Advanced Features Guide
=======================

NLSQ includes features for memory management, algorithm selection, diagnostics, caching, recovery, and input validation.

Memory Management
-----------------

NLSQ provides memory management to control memory usage and prevent out-of-memory errors.

Memory Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq import MemoryConfig, memory_context, get_memory_config
   import numpy as np

   # Configure memory settings
   config = MemoryConfig(
       memory_limit_gb=8.0,  # Maximum memory usage
       enable_mixed_precision=True,  # Use mixed precision to save memory
       enable_memory_monitoring=True,  # Monitor memory usage in real-time
       max_cache_size_gb=2.0,  # Limit cache size
       enable_garbage_collection=True,  # Auto-trigger garbage collection
       chunk_size_factor=0.8,  # Conservative chunking
       min_chunk_size=10000,  # Minimum chunk size
       max_chunk_size=1000000,  # Maximum chunk size
   )

   # Use memory context for temporary settings
   with memory_context(config):
       cf = CurveFit()
       popt, pcov = cf.curve_fit(func, x, y, p0=p0)

   # Check current memory configuration
   current_config = get_memory_config()
   print(f"Memory limit: {current_config.memory_limit_gb} GB")

Memory Monitoring
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq import get_memory_stats, MemoryManager

   # Get detailed memory statistics
   stats = get_memory_stats()
   print(f"Total memory usage: {stats.total_memory_gb:.2f} GB")
   print(f"Array pool size: {stats.pool_size_mb:.1f} MB")
   print(f"Cache memory: {stats.cache_memory_gb:.2f} GB")

   # Advanced memory management
   manager = MemoryManager(gc_threshold=0.8, safety_factor=1.2)

   # Monitor memory during optimization
   with manager.monitor_memory():
       result = cf.curve_fit(func, large_x, large_y)

Algorithm Selection
-------------------

NLSQ can select algorithms based on problem characteristics.

Basic Algorithm Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq.algorithm_selector import AlgorithmSelector, auto_select_algorithm
   from nlsq import curve_fit
   import jax.numpy as jnp

   # Create selector
   selector = AlgorithmSelector()


   # Define your model
   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Auto-select best algorithm
   recommendations = auto_select_algorithm(f=model, xdata=x, ydata=y, p0=[1.0, 0.5, 0.1])

   # Extract method from recommendations
   method = recommendations.get("algorithm", "trf")

   # Use selected algorithm
   popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5, 0.1], method=method)

   print(f"Selected algorithm: {method}")

Advanced Selection Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Quick auto-selection for common cases
   from nlsq.algorithm_selector import auto_select_algorithm

   recommendations = auto_select_algorithm(
       func=model, xdata=x, ydata=y, p0=p0, memory_limit_gb=4.0
   )

   # Use the recommended settings
   optimal_method = recommendations.get("algorithm", "trf")
   popt, pcov = curve_fit(model, x, y, p0=p0, method=optimal_method)

Diagnostics & Monitoring
------------------------

Monitor optimization progress and detect convergence issues.

Convergence Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq import ConvergenceMonitor, CurveFit
   import numpy as np

   # Configure convergence monitoring
   conv_monitor = ConvergenceMonitor(
       window_size=10, sensitivity=1.2  # Moving window size  # Detection sensitivity
   )

   # Use CurveFit with stability features enabled
   cf = CurveFit(enable_stability=True, enable_recovery=True)

   # Perform fitting
   popt, pcov = cf.curve_fit(func, x, y, p0=p0)

Using OptimizationDiagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq.diagnostics import OptimizationDiagnostics
   import numpy as np

   # Create diagnostics object
   diagnostics = OptimizationDiagnostics()

   # Record optimization data during fitting
   # (This would typically be done internally by the optimizer)
   diagnostics.record_iteration(
       iteration=0, residual=1.0, parameters=np.array([1.0, 0.5]), jacobian=None
   )

   # Access diagnostic information
   summary = diagnostics.get_summary()
   print(f"Total iterations: {summary.get('n_iterations', 0)}")

Smart Caching System
--------------------

Cache expensive computations like function evaluations.

Basic Caching Setup
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq import SmartCache, cached_function, curve_fit
   import jax.numpy as jnp

   # Configure caching
   cache = SmartCache(
       max_memory_items=1000, disk_cache_enabled=True, cache_dir=".nlsq_cache"
   )


   # Define cacheable function
   @cached_function
   def expensive_model(x, a, b, c):
       return a * jnp.exp(-b * x**2) + c * jnp.sin(x)


   # First fit - compiles and may cache internally
   popt1, pcov1 = curve_fit(expensive_model, x1, y1, p0=[1.0, 0.1, 0.5])

   # Second fit - may use cached JIT compilation
   popt2, pcov2 = curve_fit(expensive_model, x2, y2, p0=[1.2, 0.15, 0.4])

Cache Management and Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq import get_global_cache, clear_all_caches

   # Check cache performance
   cache_stats = cache.get_stats()
   print(f"Cache hits: {cache_stats.hits}")
   print(f"Cache misses: {cache_stats.misses}")
   print(f"Hit rate: {cache_stats.hit_rate:.1%}")
   print(f"Memory usage: {cache_stats.memory_usage_mb:.1f} MB")
   print(f"Disk usage: {cache_stats.disk_usage_mb:.1f} MB")

   # Global cache management
   global_cache = get_global_cache()
   print(f"Global cache size: {len(global_cache)} items")

   # Clear all caches when needed
   clear_all_caches()

Optimization Recovery & Error Handling
---------------------------------------

Handle optimization failures with recovery strategies.

Using OptimizationRecovery
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq import OptimizationRecovery, CurveFit, curve_fit
   import numpy as np

   # Create recovery handler
   recovery = OptimizationRecovery(max_attempts=3, perturbation_factor=0.1)

   # CurveFit with recovery enabled
   cf = CurveFit(enable_recovery=True)

   # Standard fitting with error handling
   try:
       popt, pcov = cf.curve_fit(func, x, y, p0=p0_initial)
       print(f"Fitted parameters: {popt}")

   except Exception as e:
       print(f"Optimization failed: {e}")
       # Try manual recovery
       perturbed_p0 = recovery.perturb_parameters(p0_initial)
       popt, pcov = curve_fit(func, x, y, p0=perturbed_p0)
       # Handle graceful failure


Input Validation & Error Prevention
------------------------------------

Validate inputs to catch errors early.

Basic Input Validation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq import InputValidator
   import numpy as np

   # Create validator
   validator = InputValidator(fast_mode=True)  # Use fast mode for performance

   # Validate inputs before fitting
   try:
       warnings, errors, clean_x, clean_y = validator.validate_curve_fit_inputs(
           func=model_function, xdata=x, ydata=y, p0=p0, bounds=bounds, sigma=sigma
       )

       # Handle warnings
       for warning in warnings:
           print(f"Warning: {warning}")

       # Check for errors
       if errors:
           for error in errors:
               print(f"Error: {error}")
           raise ValidationError("Input validation failed")

       # Use cleaned data for fitting
       popt, pcov = curve_fit(model_function, clean_x, clean_y, p0=p0)

   except ValidationError as e:
       print(f"Validation failed: {e}")
       # Handle validation errors appropriately

Custom Validation Rules
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create validator with custom rules
   custom_validator = InputValidator(fast_mode=False)  # Enable all checks


   # Add custom validation logic
   def custom_validate_physics_model(func, xdata, ydata, p0):
       """Custom validation for physics models."""
       # Check physical constraints
       if np.any(xdata < 0):
           raise ValidationError("Time values must be non-negative")

       if p0 is not None and len(p0) > 0:
           if p0[0] <= 0:  # Amplitude must be positive
               raise ValidationError("Amplitude parameter must be positive")

       # Check data quality
       signal_to_noise = np.std(ydata) / np.std(np.diff(ydata))
       if signal_to_noise < 2.0:
           warnings.warn("Low signal-to-noise ratio detected", UserWarning)

       return True


   # Use custom validation
   custom_validate_physics_model(physics_model, time_data, signal_data, initial_params)

Performance Optimization Tips
-----------------------------

Best Practices for Advanced Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Memory Management**: Set memory limits based on your system
2. **Algorithm Selection**: Use auto_select_algorithm for optimal method choice
3. **Caching**: Enable caching for repeated computations
4. **Stability**: Enable stability features in CurveFit
5. **Recovery**: Enable recovery for robust operation
6. **Validation**: Use InputValidator for data quality checks

.. code-block:: python

   # Example: Combining multiple features
   from nlsq import (
       CurveFit,
       MemoryConfig,
       memory_context,
       SmartCache,
       InputValidator,
       curve_fit_large,
   )
   from nlsq.algorithm_selector import auto_select_algorithm

   # Configure features
   memory_config = MemoryConfig(memory_limit_gb=8.0)
   cache = SmartCache(disk_cache_enabled=True)
   validator = InputValidator(fast_mode=True)


   # Production-ready fitting function
   def robust_curve_fit(func, x, y, p0, bounds=None):
       # Validate inputs
       warnings, errors, clean_x, clean_y = validator.validate_curve_fit_inputs(
           f=func, xdata=x, ydata=y, p0=p0, bounds=bounds
       )

       if errors:
           raise ValueError(f"Validation errors: {errors}")

       # Use memory context
       with memory_context(memory_config):
           # Auto-select algorithm
           recommendations = auto_select_algorithm(
               f=func, xdata=clean_x, ydata=clean_y, p0=p0, bounds=bounds
           )
           method = recommendations.get("algorithm", "trf")

           # Fit based on dataset size
           if len(clean_x) > 1_000_000:
               return curve_fit_large(
                   func,
                   clean_x,
                   clean_y,
                   p0=p0,
                   bounds=bounds,
                   method=method,
                   show_progress=True,
               )
           else:
               cf = CurveFit(enable_stability=True, enable_recovery=True)
               return cf.curve_fit(
                   func, clean_x, clean_y, p0=p0, bounds=bounds, method=method
               )


   # Use in production
   popt, pcov = robust_curve_fit(model, xdata, ydata, p0=[1, 2, 3])

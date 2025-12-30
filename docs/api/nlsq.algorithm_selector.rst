nlsq.algorithm\_selector module
================================

.. automodule:: nlsq.precision.algorithm_selector
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``algorithm_selector`` module provides automatic selection of the best optimization algorithm based on problem characteristics.

Key Features
------------

- **Automatic algorithm selection** based on problem analysis
- **Performance optimization** with problem-specific tuning
- **Convergence analysis** and parameter adjustment
- **Robustness testing** with multiple initialization strategies

Supported Algorithms
--------------------

- **Trust Region Reflective (TRF)**: Best for bounded problems
- **Levenberg-Marquardt (LM)**: Fast for well-conditioned problems
- **Dogleg**: Efficient for trust-region problems

Classes
-------

.. autoclass:: nlsq.algorithm_selector.AlgorithmSelector
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: nlsq.algorithm_selector.auto_select_algorithm
   :noindex:

Example Usage
-------------

.. code-block:: python

   from nlsq.algorithm_selector import auto_select_algorithm
   from nlsq import curve_fit
   import jax.numpy as jnp


   # Define model
   def model_nonlinear(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Generate data
   x = jnp.linspace(0, 10, 100)
   y = 2.5 * jnp.exp(-0.5 * x) + 1.0 + 0.1 * jnp.random.randn(100)

   # Auto-select best algorithm
   recommendations = auto_select_algorithm(
       f=model_nonlinear, xdata=x, ydata=y, p0=[1.0, 0.5, 0.1]
   )

   # Use recommended algorithm
   method = recommendations.get("algorithm", "trf")
   popt, pcov = curve_fit(model_nonlinear, x, y, p0=[1.0, 0.5, 0.1], method=method)

   print(f"Selected algorithm: {method}")
   print(f"Fitted parameters: {popt}")

Selection Criteria
------------------

The selector analyzes:

- **Problem size**: Number of parameters and data points
- **Bounds**: Presence of parameter bounds
- **Conditioning**: Jacobian condition number
- **Nonlinearity**: Degree of nonlinearity
- **Sparsity**: Jacobian sparsity pattern

Advanced Usage
--------------

.. code-block:: python

   from nlsq.algorithm_selector import AlgorithmSelector

   # Create selector with custom settings
   selector = AlgorithmSelector(enable_diagnostics=True, robustness_test=True)

   # Analyze problem
   analysis = selector.analyze_problem(
       f=model_nonlinear, xdata=x, ydata=y, p0=[1.0, 0.5, 0.1]
   )

   # Get detailed recommendations
   recommendations = selector.recommend_algorithm(analysis)
   print(f"Primary: {recommendations['primary']}")
   print(f"Fallback: {recommendations['fallback']}")
   print(f"Reasoning: {recommendations['reasoning']}")

See Also
--------

- :doc:`nlsq.trf` - Trust Region Reflective algorithm
- :doc:`nlsq.least_squares` - Least squares solver
- :doc:`../reference/configuration` - Configuration reference

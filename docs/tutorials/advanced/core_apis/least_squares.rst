LeastSquares Class
==================

The ``LeastSquares`` class provides direct access to the optimization engine.

Why Use LeastSquares?
---------------------

1. **Custom Residuals**: Define any residual function
2. **Full Control**: All optimizer parameters exposed
3. **Diagnostics**: Detailed convergence information
4. **Integration**: Connect to custom pipelines

Basic Usage
-----------

.. code-block:: python

   from nlsq.core.least_squares import LeastSquares
   import jax.numpy as jnp
   import numpy as np


   # Define residual function
   def residuals(params, xdata, ydata):
       a, b, c = params
       return a * jnp.exp(-b * xdata) + c - ydata


   # Create optimizer
   optimizer = LeastSquares()

   # Run optimization
   result = optimizer.least_squares(
       fun=residuals,
       x0=[1.0, 0.5, 0.0],
       args=(xdata, ydata),
       bounds=(-np.inf, np.inf),
       method="trf",
   )

   # Extract results
   popt = result.x
   print(f"Optimal parameters: {popt}")

Constructor Options
-------------------

.. code-block:: python

   optimizer = LeastSquares(
       enable_stability=True,  # Stability checks
       enable_diagnostics=True,  # Convergence metrics
       max_jacobian_elements_for_svd=10_000_000,  # SVD threshold
   )

least_squares Method
--------------------

Full signature:

.. code-block:: python

   result = optimizer.least_squares(
       fun,  # Residual function f(x, *args) -> residuals
       x0,  # Initial parameter guess
       jac=None,  # Jacobian: callable, '2-point', '3-point', 'cs'
       bounds=(-np.inf, np.inf),  # (lower, upper) bounds
       method="trf",  # 'trf', 'dogbox', 'lm'
       ftol=1e-8,  # Function tolerance
       xtol=1e-8,  # Parameter tolerance
       gtol=1e-8,  # Gradient tolerance
       x_scale="jac",  # Parameter scaling
       loss="linear",  # Loss function
       f_scale=1.0,  # Soft margin for outliers
       diff_step=None,  # Finite difference step
       tr_solver="exact",  # 'exact' or 'lsmr'
       tr_options={},  # Trust region options
       jac_sparsity=None,  # Sparsity structure
       max_nfev=None,  # Max function evaluations
       verbose=0,  # Verbosity level
       args=(),  # Additional args for fun
       kwargs={},  # Additional kwargs for fun
       jacobian_mode="auto",  # 'auto', 'fwd', 'rev'
       xdata=None,  # For diagnostics
       ydata=None,  # For diagnostics
       data_mask=None,  # Point masking
       transform=None,  # Residual transform
   )

Return Value
------------

The ``result`` is an ``OptimizeResult`` object:

.. code-block:: python

   result = optimizer.least_squares(...)

   result.x  # Optimal parameters
   result.cost  # Final cost (0.5 * sum(residuals^2))
   result.fun  # Residual values at solution
   result.jac  # Jacobian at solution
   result.grad  # Gradient at solution
   result.optimality  # Optimality measure
   result.active_mask  # Bounds active at solution
   result.nfev  # Function evaluations
   result.njev  # Jacobian evaluations
   result.status  # Convergence status
   result.message  # Status message
   result.success  # True if converged

Jacobian Options
----------------

**Automatic differentiation (default):**

.. code-block:: python

   result = optimizer.least_squares(
       fun=residuals,
       x0=x0,
       jac=None,  # Uses autodiff
       jacobian_mode="auto",  # Selects fwd/rev based on dimensions
   )

**Analytical Jacobian:**

.. code-block:: python

   def jacobian(params, xdata, ydata):
       a, b, c = params
       exp_term = jnp.exp(-b * xdata)
       da = exp_term
       db = -a * xdata * exp_term
       dc = jnp.ones_like(xdata)
       return jnp.column_stack([da, db, dc])


   result = optimizer.least_squares(
       fun=residuals, x0=x0, jac=jacobian, args=(xdata, ydata)
   )

**Finite differences:**

.. code-block:: python

   result = optimizer.least_squares(
       fun=residuals,
       x0=x0,
       jac="2-point",  # or '3-point', 'cs' (complex step)
       args=(xdata, ydata),
   )

Trust Region Solver
-------------------

**Exact solver (default for small problems):**

.. code-block:: python

   result = optimizer.least_squares(
       fun=residuals, x0=x0, tr_solver="exact"  # Uses SVD decomposition
   )

**LSMR solver (for large problems):**

.. code-block:: python

   result = optimizer.least_squares(
       fun=residuals,
       x0=x0,
       tr_solver="lsmr",  # Iterative solver
       tr_options={"maxiter": 100},
   )

Loss Functions
--------------

Robust loss functions for outliers:

.. code-block:: python

   # Linear (default): sum(rho(f_i^2))
   # where rho(z) = z for linear

   result = optimizer.least_squares(
       fun=residuals,
       x0=x0,
       loss="soft_l1",  # or 'huber', 'cauchy', 'arctan'
       f_scale=0.1,  # Soft margin
   )

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq.core.least_squares import LeastSquares

   # Generate data
   np.random.seed(42)
   xdata = np.linspace(0, 10, 100)
   ydata = 2.5 * np.exp(-0.5 * xdata) + 0.3 + 0.1 * np.random.randn(100)


   # Define residuals
   def residuals(params):
       a, b, c = params
       return a * jnp.exp(-b * xdata) + c - ydata


   # Create optimizer with diagnostics
   optimizer = LeastSquares(enable_stability=True, enable_diagnostics=True)

   # Run optimization
   result = optimizer.least_squares(
       fun=residuals,
       x0=[1.0, 0.3, 0.0],
       bounds=([0, 0, -1], [10, 5, 1]),
       method="trf",
       ftol=1e-10,
       xtol=1e-10,
       gtol=1e-10,
       verbose=2,
   )

   # Results
   print(f"\nOptimization result:")
   print(f"  Parameters: {result.x}")
   print(f"  Cost: {result.cost}")
   print(f"  Iterations: {result.nfev}")
   print(f"  Status: {result.message}")
   print(f"  Success: {result.success}")

When to Use LeastSquares
------------------------

**Use LeastSquares when:**

- Custom residual functions
- Need full result object
- Integrating with other systems
- Research/algorithm development

**Use CurveFit/fit() when:**

- Standard curve fitting
- Don't need low-level control
- Want covariance computed automatically

Next Steps
----------

- :doc:`trf_optimizer` - Algorithm details
- :doc:`../orchestration/index` - Decomposed components

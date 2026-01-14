CovarianceComputer
==================

.. versionadded:: 0.6.4

The ``CovarianceComputer`` estimates parameter uncertainties via SVD-based
covariance computation.

Basic Usage
-----------

.. code-block:: python

   from nlsq.core.orchestration import CovarianceComputer

   computer = CovarianceComputer()
   cov_result = computer.compute(
       result=optimize_result,  # From LeastSquares
       n_data=len(ydata),
       sigma=None,
       absolute_sigma=False,
   )

   # Access results
   pcov = cov_result.pcov
   perr = cov_result.perr
   condition_number = cov_result.condition_number

CovarianceResult
----------------

The ``compute()`` method returns a ``CovarianceResult`` object:

.. code-block:: python

   @dataclass
   class CovarianceResult:
       pcov: np.ndarray  # Covariance matrix
       perr: np.ndarray  # Parameter errors (sqrt of diagonal)
       condition_number: float  # Jacobian condition number
       rank: int  # Jacobian rank
       s: np.ndarray  # Singular values

How It Works
------------

Covariance is computed from the Jacobian at the solution:

.. code-block:: text

   1. Get Jacobian J at optimal parameters
   2. Compute J^T @ J (approximates Hessian)
   3. Use SVD: J = U @ S @ V^T
   4. Covariance = V @ diag(1/s²) @ V^T × s²_reduced
   5. Scale by sigma if provided

.. code-block:: python

   # Internally:
   # pcov ≈ (J^T J)^-1 × residual_variance

SVD-Based Computation
---------------------

SVD provides numerical stability:

.. code-block:: python

   # Standard inverse (numerically unstable)
   # pcov = np.linalg.inv(J.T @ J) * s_sq

   # SVD-based (stable)
   U, s, Vh = np.linalg.svd(J, full_matrices=False)
   pcov = (Vh.T / s**2) @ Vh * residual_variance

Sigma Handling
--------------

**Without sigma (default):**

.. code-block:: python

   cov_result = computer.compute(
       result=result, n_data=100, sigma=None, absolute_sigma=False
   )
   # Covariance scaled by residual variance

**With absolute sigma:**

.. code-block:: python

   cov_result = computer.compute(
       result=result, n_data=100, sigma=measurement_errors, absolute_sigma=True
   )
   # Covariance reflects actual uncertainties

Condition Number
----------------

The condition number indicates numerical stability:

.. code-block:: python

   cov_result = computer.compute(...)
   cond = cov_result.condition_number

   if cond > 1e10:
       print("Warning: Ill-conditioned Jacobian")
   elif cond > 1e6:
       print("Note: Moderately ill-conditioned")
   else:
       print("Good conditioning")

Handling Failures
-----------------

If covariance cannot be computed:

.. code-block:: python

   cov_result = computer.compute(...)

   if np.any(np.isinf(cov_result.pcov)):
       print("Covariance estimation failed")
       # Reasons:
       # - Singular Jacobian
       # - Parameters at bounds
       # - Ill-conditioning

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq.core.least_squares import LeastSquares
   from nlsq.core.orchestration import CovarianceComputer


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y = 2.5 * np.exp(-0.5 * x) + 0.3 + 0.1 * np.random.randn(100)
   sigma = 0.1 * np.ones(100)

   # Run optimization
   optimizer = LeastSquares()
   result = optimizer.least_squares(fun=lambda p: model(x, *p) - y, x0=[2, 0.5, 0])

   # Compute covariance
   computer = CovarianceComputer()
   cov_result = computer.compute(
       result=result, n_data=len(y), sigma=sigma, absolute_sigma=True
   )

   print(f"Optimal parameters: {result.x}")
   print(f"Parameter errors: {cov_result.perr}")
   print(f"Condition number: {cov_result.condition_number:.2e}")
   print(f"Covariance matrix:\n{cov_result.pcov}")

Next Steps
----------

- :doc:`streaming_coordinator` - Memory strategy
- :doc:`../stability/index` - Numerical stability

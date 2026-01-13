Optimization Pipeline
=====================

This page explains how data flows through NLSQ's optimization pipeline.

Pipeline Overview
-----------------

.. code-block:: text

   User Code
       │
       ▼
   fit() / curve_fit()              ◄── Entry point
       │
       ▼
   ┌─────────────────────────────────────────────────┐
   │                  CurveFit                        │
   │  ┌─────────────────────────────────────────────┐│
   │  │ DataPreprocessor                             ││
   │  │ - Validate inputs                           ││
   │  │ - Convert to JAX arrays                     ││
   │  │ - Handle NaN, masking                       ││
   │  └─────────────────────────────────────────────┘│
   │                    │                             │
   │                    ▼                             │
   │  ┌─────────────────────────────────────────────┐│
   │  │ OptimizationSelector                         ││
   │  │ - Detect parameter count                    ││
   │  │ - Process bounds                            ││
   │  │ - Generate initial guess                    ││
   │  │ - Select method/solver                      ││
   │  └─────────────────────────────────────────────┘│
   │                    │                             │
   │                    ▼                             │
   │  ┌─────────────────────────────────────────────┐│
   │  │ StreamingCoordinator                         ││
   │  │ - Analyze memory requirements               ││
   │  │ - Select: STANDARD / CHUNKED / STREAMING    ││
   │  └─────────────────────────────────────────────┘│
   └────────────────────┬────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────┐
   │              LeastSquares                        │
   │  - Create residual function                     │
   │  - Configure Jacobian (auto-diff)               │
   │  - Set up trust region options                  │
   └────────────────────┬────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────┐
   │          TrustRegionReflective                   │
   │  - Iterative optimization loop                  │
   │  - Trust region updates                         │
   │  - Convergence checking                         │
   └────────────────────┬────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────┐
   │           CovarianceComputer                     │
   │  - Compute Jacobian at solution                 │
   │  - SVD for covariance estimation                │
   │  - Apply sigma transformation                   │
   └────────────────────┬────────────────────────────┘
                        │
                        ▼
                   (popt, pcov)

Step-by-Step Walkthrough
------------------------

**1. Entry Point (fit/curve_fit)**

.. code-block:: python

   from nlsq import fit

   popt, pcov = fit(model, x, y, p0=[1, 0.5])
   # fit() creates CurveFit instance and calls curve_fit()

**2. Data Preprocessing**

.. code-block:: python

   # DataPreprocessor handles:
   # - Type conversion to JAX arrays
   # - NaN handling based on nan_policy
   # - Data masking
   # - Shape validation

   preprocessed = DataPreprocessor().preprocess(
       f=model, xdata=x, ydata=y, sigma=None, nan_policy="raise"
   )

**3. Optimization Selection**

.. code-block:: python

   # OptimizationSelector determines:
   # - Number of parameters (from model signature)
   # - Bounds processing
   # - Initial guess validation
   # - Method selection (trf, dogbox)

   config = OptimizationSelector().select(
       f=model, xdata=x, ydata=y, p0=[1, 0.5], bounds=None, method="trf"
   )

**4. Memory Strategy Selection**

.. code-block:: python

   # StreamingCoordinator analyzes memory:
   # - Data size vs available memory
   # - Peak memory estimation
   # - Strategy selection

   decision = StreamingCoordinator().decide(xdata=x, ydata=y, n_params=2, workflow="auto")
   # Returns: strategy='standard', 'chunked', or 'streaming'

**5. LeastSquares Optimization**

.. code-block:: python

   # LeastSquares sets up the optimization:
   # - Wraps model as residual function
   # - Configures automatic differentiation
   # - Calls TrustRegionReflective

   optimizer = LeastSquares()
   result = optimizer.least_squares(
       fun=residuals,
       x0=p0,
       jac="2-point",  # or analytical
       bounds=(-np.inf, np.inf),
       method="trf",
   )

**6. Trust Region Iteration**

.. code-block:: python

   # TRF performs iterative optimization:
   while not converged:
       # 1. Compute Jacobian (auto-diff or finite difference)
       J = compute_jacobian(x_current)

       # 2. Solve trust region subproblem
       step = solve_subproblem(J, residuals, trust_radius)

       # 3. Evaluate new point
       x_new = x_current + step
       cost_new = compute_cost(x_new)

       # 4. Update trust region
       if cost_new < cost_current:
           x_current = x_new
           expand_trust_region()
       else:
           contract_trust_region()

       # 5. Check convergence
       converged = check_convergence(ftol, xtol, gtol)

**7. Covariance Computation**

.. code-block:: python

   # CovarianceComputer estimates uncertainties:
   # - Final Jacobian at solution
   # - SVD decomposition
   # - Covariance from inverse of J^T J

   cov_result = CovarianceComputer().compute(
       result=optimize_result, n_data=len(y), sigma=None, absolute_sigma=False
   )

Jacobian Computation
--------------------

NLSQ uses JAX autodiff for Jacobians:

.. code-block:: python

   import jax

   # Forward-mode: efficient for few parameters
   J = jax.jacfwd(residual_func)(params)

   # Reverse-mode: efficient for many parameters
   J = jax.jacrev(residual_func)(params)

   # Auto-selection based on dimensions
   # n_params < n_residuals → forward-mode
   # n_params > n_residuals → reverse-mode

Global Optimization Path
------------------------

For ``workflow='auto_global'``:

.. code-block:: text

   fit(workflow='auto_global')
           │
           ▼
   MethodSelector
   - scale_ratio > 1000? → CMA-ES
   - otherwise → Multi-Start
           │
           ├──► MultiStartOrchestrator
           │       │
           │       ├── Latin Hypercube Sampling
           │       ├── n parallel local optimizations
           │       └── Select best result
           │
           └──► CMAESOptimizer (if selected)
                   │
                   ├── Evolutionary search
                   ├── BIPOP restarts
                   └── Local refinement

Next Steps
----------

- :doc:`jax_patterns` - JAX-specific patterns
- :doc:`../core_apis/index` - Using core classes directly

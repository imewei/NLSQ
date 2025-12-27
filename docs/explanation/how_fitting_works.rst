How Curve Fitting Works
=======================

This guide explains the mathematical foundation of nonlinear least squares
curve fitting - what it means to "fit" a model to data and how the algorithm
finds optimal parameters.

The Fitting Problem
-------------------

Given:
- Data points: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)
- A model function: f(x; θ) with parameters θ = (θ₁, θ₂, ..., θₘ)

Find the parameters θ that minimize the **sum of squared residuals**:

.. math::

   S(θ) = \sum_{i=1}^{n} [y_i - f(x_i; θ)]^2

This is called the **least squares** objective because we're minimizing
the sum of squared differences (residuals) between data and model.

Why Squared Residuals?
~~~~~~~~~~~~~~~~~~~~~~

1. **Mathematical convenience**: The squared function is smooth and
   differentiable everywhere, making optimization easier.

2. **Statistical interpretation**: If errors are normally distributed,
   minimizing squared residuals gives the maximum likelihood estimate.

3. **Equal treatment**: Positive and negative residuals contribute equally.

Linear vs Nonlinear Least Squares
---------------------------------

**Linear least squares**: The model is linear in parameters:

.. math::

   y = θ_1 x + θ_2

This has a direct algebraic solution (normal equations).

**Nonlinear least squares**: The model is nonlinear in parameters:

.. math::

   y = θ_1 e^{-θ_2 x}

No direct solution exists - we need iterative optimization.

The Optimization Process
------------------------

NLSQ uses the **Trust Region Reflective (TRF)** algorithm:

1. **Initialize**: Start with initial guess θ₀

2. **Evaluate**: Compute residuals r = y - f(x; θ)

3. **Linearize**: Compute Jacobian J = ∂f/∂θ (automatic with JAX)

4. **Solve subproblem**: Find step direction δ within a "trust region"

5. **Update**: θ ← θ + δ

6. **Check convergence**: Stop if change is small enough

7. **Repeat**: Go to step 2

.. code-block:: text

   θ₀ → [compute residuals] → [compute Jacobian] → [solve for δ]
    ↑                                                    ↓
    └──────────────── [update θ ← θ + δ] ←──────────────┘

The Jacobian Matrix
-------------------

The Jacobian J is an n × m matrix of partial derivatives:

.. math::

   J_{ij} = \frac{\partial f(x_i; θ)}{\partial θ_j}

It tells us how sensitive the model output is to each parameter.

**In NLSQ**, the Jacobian is computed automatically using JAX's
automatic differentiation - no manual derivatives needed!

.. code-block:: python

   # You just write the model
   def model(x, a, b):
       return a * jnp.exp(-b * x)


   # NLSQ automatically computes ∂f/∂a and ∂f/∂b

Trust Region Method
-------------------

Instead of taking the full Newton step, TRF restricts the step to a
"trust region" - a ball of radius Δ around the current point.

This ensures stability:

- **Step too large?** Reduce trust region
- **Good step?** Expand trust region
- **Near bounds?** Use reflected steps

See :doc:`trust_region` for more details.

Convergence Criteria
--------------------

Optimization stops when any of these are satisfied:

1. **Gradient tolerance (gtol)**: Gradient is nearly zero

   .. math::

      \|J^T r\|_\infty < \text{gtol}

2. **Function tolerance (ftol)**: Cost isn't decreasing

   .. math::

      \frac{|S_{k+1} - S_k|}{S_k} < \text{ftol}

3. **Step tolerance (xtol)**: Parameters aren't changing

   .. math::

      \frac{\|δ\|}{\|θ\|} < \text{xtol}

4. **Maximum iterations**: Safety limit reached

Parameter Uncertainties
-----------------------

After finding optimal θ*, we estimate uncertainties from the covariance matrix:

.. math::

   \text{cov}(θ) \approx s^2 (J^T J)^{-1}

where s² is the residual variance:

.. math::

   s^2 = \frac{S(θ^*)}{n - m}

The standard error of each parameter is:

.. math::

   σ_{θ_i} = \sqrt{\text{cov}(θ)_{ii}}

Goodness of Fit
---------------

**R-squared (coefficient of determination)**:

.. math::

   R^2 = 1 - \frac{S(θ^*)}{S_{\text{tot}}}

where Sₜₒₜ = Σ(yᵢ - ȳ)² is the total variance.

- R² = 1: Perfect fit
- R² = 0: Model no better than mean
- R² < 0: Model worse than mean (rare, indicates problems)

**Reduced chi-squared**:

.. math::

   χ^2_ν = \frac{S(θ^*)}{n - m}

Should be approximately 1 for a good fit with correctly estimated errors.

When Fitting Fails
------------------

Common issues:

1. **Local minimum**: Found a suboptimal solution

   - Use better initial guesses
   - Try global optimization (preset='global')

2. **Ill-conditioned Jacobian**: Parameters are poorly determined

   - Simplify model
   - Fix some parameters

3. **Divergence**: Cost keeps increasing

   - Check data quality
   - Adjust bounds

4. **Oscillation**: Parameters alternate without converging

   - Check for parameter correlations
   - Reparameterize model

See :doc:`/howto/debug_bad_fits` for troubleshooting.

Summary
-------

.. list-table::
   :header-rows: 1

   * - Concept
     - Meaning
   * - Residuals
     - Difference between data and model
   * - Least squares
     - Minimize sum of squared residuals
   * - Jacobian
     - How model changes with parameters
   * - Trust region
     - Safe step size limit
   * - Covariance
     - Parameter uncertainties
   * - R²
     - Fraction of variance explained

See Also
--------

- :doc:`trust_region` - Trust Region Reflective algorithm details
- :doc:`jax_autodiff` - How automatic differentiation works
- :doc:`/tutorials/02_understanding_results` - Practical result interpretation

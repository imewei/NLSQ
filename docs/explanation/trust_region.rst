Trust Region Reflective Algorithm
==================================

NLSQ uses the **Trust Region Reflective (TRF)** algorithm for optimization.
This guide explains how TRF works and why it's well-suited for curve fitting.

Why Trust Region?
-----------------

Classical Newton methods take steps based on local quadratic approximations.
These steps can be too large, causing:

- Overshooting the minimum
- Divergence in flat regions
- Numerical instability

Trust region methods solve this by:

1. Building a quadratic model of the objective
2. Limiting the step size to a "trust region"
3. Adjusting the region size based on how well the model predicts

The Algorithm
-------------

At each iteration:

1. **Build quadratic model**:

   .. math::

      m(δ) = \|r + Jδ\|^2 \approx S(θ + δ)

   where r = residuals, J = Jacobian

2. **Minimize within trust region**:

   .. math::

      \min_δ m(δ) \quad \text{subject to} \quad \|δ\| ≤ Δ

3. **Compute actual vs predicted improvement**:

   .. math::

      ρ = \frac{S(θ) - S(θ + δ)}{m(0) - m(δ)}

4. **Update trust region**:

   - ρ > 0.75: Expand region (Δ ← 2Δ)
   - ρ < 0.25: Shrink region (Δ ← 0.25Δ)
   - 0.25 ≤ ρ ≤ 0.75: Keep same

5. **Accept or reject step**:

   - ρ > 0: Accept (θ ← θ + δ)
   - ρ ≤ 0: Reject (keep θ)

Handling Bounds
---------------

The "Reflective" part of TRF handles parameter bounds elegantly:

**Problem**: Parameters must stay within [lower, upper].

**Solution**: Transform the problem so that the optimizer can explore
freely, but the actual parameters stay bounded.

.. code-block:: text

   Unconstrained space     Bounded space
         -∞ ─────────────── lower
          │                   │
          0 ═══════════════ middle
          │                   │
         +∞ ─────────────── upper

When a step would exceed a bound:

1. **Reflect**: Bounce back from the boundary
2. **Reduce**: Take a smaller step along the boundary
3. **Project**: Clip to the nearest valid value

This is why bounds in NLSQ "just work" - you specify them, and TRF
handles the constrained optimization automatically.

Comparison with Other Methods
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Pros
     - Cons
   * - **TRF** (NLSQ default)
     - Robust, handles bounds well, converges reliably
     - Slightly slower per iteration
   * - Levenberg-Marquardt
     - Fast for unconstrained problems
     - No bound handling
   * - Dogbox
     - Simple bound handling
     - Can be slow near bounds
   * - Gauss-Newton
     - Fast when near solution
     - Can diverge, no bounds

NLSQ uses TRF because it combines robustness with excellent bound handling.

SVD Subproblem Solver
---------------------

The trust region subproblem is solved using Singular Value Decomposition (SVD):

.. math::

   J = UΣV^T

This provides:

1. **Numerical stability**: Works even with near-singular Jacobians
2. **Condition information**: Singular values reveal ill-conditioning
3. **Null space handling**: Properly manages underdetermined problems

Convergence Properties
----------------------

TRF has strong convergence guarantees:

1. **Global convergence**: From any starting point, converges to a
   stationary point (local minimum or saddle)

2. **Local quadratic convergence**: Near the solution, converges
   quadratically (like Newton's method)

3. **Superlinear convergence**: Even with inexact subproblem solutions

Tuning Parameters
-----------------

NLSQ exposes TRF tuning via tolerances:

.. code-block:: python

   popt, pcov = curve_fit(
       model,
       x,
       y,
       gtol=1e-8,  # Gradient tolerance
       ftol=1e-8,  # Function tolerance
       xtol=1e-8,  # Step tolerance
       max_nfev=500,  # Max function evaluations
   )

**Guidelines**:

- Tighter tolerances (1e-10): More precise, slower
- Looser tolerances (1e-6): Faster, may stop early
- Default (1e-8): Good balance for most problems

Monitoring Convergence
----------------------

NLSQ provides convergence information:

.. code-block:: python

   result = curve_fit(model, x, y)

   print(f"Iterations: {result.nit}")
   print(f"Function evaluations: {result.nfev}")
   print(f"Final cost: {result.cost}")
   print(f"Optimality: {result.optimality}")  # Gradient norm
   print(f"Message: {result.message}")

Summary
-------

The Trust Region Reflective algorithm:

1. Uses a quadratic model of the objective
2. Limits steps to a trust region for stability
3. Adjusts region size based on prediction quality
4. Handles bounds via reflective transformations
5. Solves subproblems with SVD for numerical stability

This makes it robust, reliable, and suitable for a wide range of
curve fitting problems.

See Also
--------

- :doc:`how_fitting_works` - Overall fitting process
- :doc:`numerical_stability` - Stability measures
- :doc:`/howto/debug_bad_fits` - Troubleshooting

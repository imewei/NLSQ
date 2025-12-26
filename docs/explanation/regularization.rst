Group Variance Regularization
=============================

This guide documents the group variance regularization feature in NLSQ's
hybrid streaming optimizer. This feature prevents per-group parameters
(such as per-angle contrast and offset in XPCS laminar flow fitting)
from absorbing physical signals that should be captured by shared parameters.

.. contents:: Table of Contents
   :local:
   :depth: 2


Motivation
----------

In multi-angle or multi-channel fitting problems, a common model structure is:

.. math::

   y_\phi(q, t) = C_\phi \cdot g_2(q, t; \Gamma) + O_\phi

where:

- :math:`C_\phi` is the per-angle contrast
- :math:`O_\phi` is the per-angle offset
- :math:`\Gamma` are the shared physical parameters (e.g., relaxation rates)

Without regularization, the per-angle parameters :math:`C_\phi` and
:math:`O_\phi` can absorb angle-dependent physical signals that should
be captured by the shared parameters :math:`\Gamma`. This leads to:

- Artificially uniform :math:`\Gamma` values across angles
- Loss of physical information in the fitted parameters
- Systematic bias in extracted quantities

Group variance regularization addresses this by penalizing variance
within parameter groups, encouraging :math:`C_\phi` values to remain
similar to each other (and likewise for :math:`O_\phi`), while allowing
:math:`\Gamma` to capture the true angle-dependent physics.


Mathematical Formulation
------------------------

The regularized loss function becomes:

.. math::

   \mathcal{L} = \text{MSE} + \lambda \sum_g \text{Var}(\theta_g)

where:

- :math:`\text{MSE}` is the mean squared error of the fit
- :math:`\lambda` is the regularization strength (``group_variance_lambda``)
- :math:`\theta_g` is the vector of parameters in group :math:`g`
- :math:`\text{Var}(\theta_g) = \frac{1}{n_g}\sum_{i \in g}(\theta_i - \bar{\theta}_g)^2`

The variance penalty is added in two phases:

Phase 1 (Adam Warmup)
~~~~~~~~~~~~~~~~~~~~~

During Adam warmup, the loss function directly includes the variance penalty:

.. code-block:: python

   def loss_fn(params, x_batch, y_batch):
       predictions = model(x_batch, *params)
       mse = jnp.mean((y_batch - predictions) ** 2)

       variance_penalty = 0.0
       for start, end in group_slices:
           group_params = params[start:end]
           variance_penalty += jnp.var(group_params)

       return mse + lambda_ * variance_penalty

Phase 2 (Gauss-Newton)
~~~~~~~~~~~~~~~~~~~~~~

During Gauss-Newton optimization, the regularization modifies the
normal equations by adding gradient and Hessian contributions:

**Gradient contribution** (added to :math:`J^T r`):

.. math::

   \nabla_{\theta_g} \text{Var}(\theta_g) = \frac{2}{n_g}(\theta_i - \bar{\theta}_g)

**Hessian contribution** (added to :math:`J^T J`):

.. math::

   H_{\text{var}} = \frac{2}{n_g}\left(I - \frac{1}{n_g}\mathbf{1}\mathbf{1}^T\right)

where :math:`I` is the identity matrix and :math:`\mathbf{1}` is a
vector of ones. This Hessian has the form of a centering matrix that
penalizes deviations from the group mean.


Configuration
-------------

Group variance regularization is configured through ``HybridStreamingConfig``:

.. code-block:: python

   from nlsq import HybridStreamingConfig

   config = HybridStreamingConfig(
       # Enable the feature
       enable_group_variance_regularization=True,
       # Regularization strength
       group_variance_lambda=0.1,
       # Define parameter groups as (start, end) slices
       # Example: 23 contrast params [0:23], 23 offset params [23:46]
       group_variance_indices=[(0, 23), (23, 46)],
   )

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

``enable_group_variance_regularization`` : bool, default=False
    Enable the variance penalty. When False, standard MSE optimization
    is used without regularization.

``group_variance_lambda`` : float, default=0.01
    Regularization strength. Larger values more strongly penalize
    variance within parameter groups.

    - **0.001-0.01**: Light regularization, allows moderate group variation
    - **0.1-1.0**: Moderate regularization, constrains groups to be similar
    - **10-1000**: Strong regularization, forces groups to be nearly uniform

    A practical formula for XPCS applications:

    .. math::

       \lambda \approx 0.1 \times \frac{n_{\text{data}}}{n_\phi \times \sigma_{\text{exp}}^2}

    where :math:`\sigma_{\text{exp}}` is the expected experimental variation
    (e.g., 0.05 for 5% variation).

``group_variance_indices`` : list of tuple, default=None
    List of (start, end) tuples defining parameter groups. Each tuple
    specifies a slice ``[start:end]`` of the parameter vector.

    If None when regularization is enabled, no groups are regularized
    (effectively disabling the feature).


Example: XPCS Laminar Flow Fitting
----------------------------------

For XPCS laminar flow analysis with 23 angular positions:

.. code-block:: python

   import jax.numpy as jnp
   from nlsq import curve_fit, HybridStreamingConfig

   # Model: g2(q, t) for 23 angles
   # Parameters layout:
   #   [0:23]   - contrast C_phi for each angle
   #   [23:46]  - offset O_phi for each angle
   #   [46:]    - shared physical parameters (Gamma, etc.)


   def laminar_flow_model(x, *params):
       n_phi = 23
       contrast = jnp.array(params[:n_phi])
       offset = jnp.array(params[n_phi : 2 * n_phi])
       gamma = params[2 * n_phi :]  # Shared physics parameters

       # Compute g2 correlation function
       # ... physics implementation ...

       return contrast * g2_theory + offset


   # Configure with group variance regularization
   config = HybridStreamingConfig(
       enable_group_variance_regularization=True,
       group_variance_lambda=0.1,
       group_variance_indices=[
           (0, 23),  # Regularize contrast group
           (23, 46),  # Regularize offset group
       ],
       # Other settings
       precision="float64",
       gauss_newton_tol=1e-10,
   )

   # Initial parameters
   p0 = (
       [0.3] * 23  # Initial contrast (same for all angles)
       + [0.0] * 23  # Initial offset (zero for all angles)
       + [1.0, 0.1]  # Initial shared parameters
   )

   # Fit with regularization
   popt, pcov = curve_fit(
       laminar_flow_model,
       x_data,
       y_data,
       p0=p0,
       method="hybrid_streaming",
       hybrid_config=config,
   )

   # Extract results
   fitted_contrast = popt[:23]
   fitted_offset = popt[23:46]
   fitted_physics = popt[46:]

   # Check group variance
   print(f"Contrast std: {jnp.std(fitted_contrast):.4f}")
   print(f"Offset std: {jnp.std(fitted_offset):.4f}")


Choosing Lambda
---------------

The regularization strength :math:`\lambda` controls the trade-off between:

- **Fit quality**: Lower :math:`\lambda` allows per-group parameters to
  fit the data more closely
- **Physical constraint**: Higher :math:`\lambda` forces per-group
  parameters to be more uniform

L-Curve Method
~~~~~~~~~~~~~~

Perform fits with varying :math:`\lambda` and plot the L-curve:

.. code-block:: python

   import matplotlib.pyplot as plt

   lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
   mse_values = []
   variance_values = []

   for lam in lambdas:
       config = HybridStreamingConfig(
           enable_group_variance_regularization=True,
           group_variance_lambda=lam,
           group_variance_indices=[(0, 23), (23, 46)],
       )

       popt, _ = curve_fit(
           model, x, y, p0=p0, method="hybrid_streaming", hybrid_config=config
       )

       # Compute unregularized MSE
       residuals = y - model(x, *popt)
       mse = float(jnp.mean(residuals**2))
       mse_values.append(mse)

       # Compute total group variance
       var_total = jnp.var(popt[:23]) + jnp.var(popt[23:46])
       variance_values.append(float(var_total))

   # Plot L-curve
   plt.figure()
   plt.loglog(mse_values, variance_values, "o-")
   for i, lam in enumerate(lambdas):
       plt.annotate(f"{lam}", (mse_values[i], variance_values[i]))
   plt.xlabel("MSE (data fidelity)")
   plt.ylabel("Group variance (regularization)")
   plt.title("L-curve for lambda selection")
   plt.show()

Choose :math:`\lambda` at the "corner" of the L-curve where both MSE
and variance are reasonably low.

Physical Constraints
~~~~~~~~~~~~~~~~~~~~

If you have prior knowledge of expected parameter variation:

.. code-block:: python

   # Expected 5% variation in contrast across angles
   expected_sigma = 0.05 * mean_contrast

   # Set lambda to penalize deviations beyond expected variation
   lambda_ = 0.1 * n_data / (n_angles * expected_sigma**2)


Validation
----------

After fitting, validate the regularization effect:

.. code-block:: python

   # Check that group variance is reduced
   contrast_std = jnp.std(popt[:23])
   offset_std = jnp.std(popt[23:46])

   print(f"Contrast coefficient of variation: {contrast_std/jnp.mean(popt[:23]):.1%}")
   print(f"Offset standard deviation: {offset_std:.4f}")

   # Compare with unregularized fit
   config_unreg = HybridStreamingConfig(
       enable_group_variance_regularization=False,
   )
   popt_unreg, _ = curve_fit(
       model, x, y, p0=p0, method="hybrid_streaming", hybrid_config=config_unreg
   )

   contrast_std_unreg = jnp.std(popt_unreg[:23])
   print(f"Contrast std (regularized): {contrast_std:.4f}")
   print(f"Contrast std (unregularized): {contrast_std_unreg:.4f}")


Implementation Details
----------------------

Source Files
~~~~~~~~~~~~

The implementation spans two files:

**nlsq/hybrid_streaming_config.py**
    - Configuration dataclass with three new fields
    - Validation for ``group_variance_lambda > 0``
    - Validation for valid ``(start, end)`` tuples

**nlsq/adaptive_hybrid_streaming.py**
    - ``_create_warmup_loss_fn()`` (line ~817): Adds variance penalty to loss
    - ``_gauss_newton_iteration()`` (line ~1690): Adds gradient and Hessian terms

Numerical Stability
~~~~~~~~~~~~~~~~~~~

The Hessian contribution :math:`H_{\text{var}} = \frac{2}{n}(I - \frac{1}{n}\mathbf{1}\mathbf{1}^T)`
is a rank-deficient matrix (rank = n-1). This is intentional: the null
space corresponds to uniform shifts of all parameters in the group,
which do not change the variance. The existing regularization factor
in the Gauss-Newton solver handles this gracefully.

Computational Cost
~~~~~~~~~~~~~~~~~~

The additional cost per iteration is:

- **Gradient**: O(n_params) for each group
- **Hessian**: O(n_group^2) for each group

For typical XPCS applications with 23 angles and 2 groups, this adds
negligible overhead compared to the Jacobian computation.


See Also
--------

- :doc:`streaming` - Streaming optimizer concepts
- :doc:`../howto/handle_large_data` - Large dataset handling

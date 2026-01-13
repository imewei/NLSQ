Numerical Stability
===================

This chapter covers NLSQ's numerical stability features and how to handle
ill-conditioned problems.

.. toctree::
   :maxdepth: 1

   numerical_guards
   svd_fallback
   condition_monitoring
   recovery

Chapter Overview
----------------

**Numerical Guards** (10 min)
   NumericalStabilityGuard for detecting and fixing issues.

**SVD Fallback** (10 min)
   Fallback strategies when standard methods fail.

**Condition Monitoring** (5 min)
   Tracking condition numbers during optimization.

**Recovery** (10 min)
   Automatic recovery from optimization failures.

Stability Features
------------------

NLSQ includes multiple stability layers:

1. **Input Validation**: Check for NaN, inf, invalid shapes
2. **Condition Monitoring**: Track Jacobian condition number
3. **SVD Fallback**: Switch to stable SVD when needed
4. **Auto-Recovery**: Retry with different strategies

.. code-block:: python

   from nlsq import fit

   # Enable stability features
   popt, pcov = fit(
       model,
       x,
       y,
       p0=[...],
       stability="auto",  # Auto-detect issues
       fallback=True,  # Enable fallbacks
       rescale_data=True,
   )  # Normalize data

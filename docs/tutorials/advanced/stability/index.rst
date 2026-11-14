Numerical Stability
===================

This chapter covers NLSQ's numerical stability features and how to handle
ill-conditioned problems.

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

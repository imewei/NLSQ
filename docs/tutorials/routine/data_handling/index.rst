Data Handling
=============

This chapter covers how to prepare and input data for curve fitting.

.. toctree::
   :maxdepth: 1

   basic_data
   uncertainties
   bounds
   large_datasets

Chapter Overview
----------------

**Basic Data** (5 min)
   Loading and preparing x, y data arrays.

**Uncertainties** (10 min)
   Using measurement errors for weighted fitting.

**Bounds** (5 min)
   Constraining parameters to valid ranges.

**Large Datasets** (10 min)
   Handling datasets with millions of points.

Quick Reference
---------------

.. code-block:: python

   from nlsq import fit
   import numpy as np

   # Basic fit
   popt, pcov = fit(model, x, y, p0=[...])

   # With uncertainties
   popt, pcov = fit(model, x, y, p0=[...], sigma=errors)

   # With bounds
   popt, pcov = fit(model, x, y, p0=[...], bounds=([lower], [upper]))

   # Large dataset (automatic handling)
   popt, pcov = fit(model, x_large, y_large, p0=[...])
   # NLSQ auto-detects size and uses appropriate strategy

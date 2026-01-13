Fitting Presets
===============

The GUI provides three presets that map to NLSQ's 3-workflow system.

The Three Presets
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 25 30

   * - Preset
     - Workflow
     - Speed
     - Best For
   * - **Fast**
     - ``auto``
     - Fastest
     - Quick fits, good initial guess
   * - **Robust**
     - ``auto_global``
     - Moderate
     - Unknown parameters, exploration
   * - **Quality**
     - ``auto_global``
     - Slowest
     - Publication-quality results

Fast Preset
-----------

**Settings:**

- Workflow: ``auto``
- Tolerances: ``1e-6``
- Single local optimization

**When to use:**

- You have a good initial guess
- Data clearly follows the model
- Quick exploratory fitting

**Equivalent code:**

.. code-block:: python

   popt, pcov = fit(
       model, x, y, p0=[...], workflow="auto", ftol=1e-6, xtol=1e-6, gtol=1e-6
   )

Robust Preset
-------------

**Settings:**

- Workflow: ``auto_global``
- n_starts: 10
- Tolerances: ``1e-8``
- Requires bounds (prompted if not set)

**When to use:**

- Initial guess is uncertain
- Multiple local minima possible
- First fit of new data

**Equivalent code:**

.. code-block:: python

   popt, pcov = fit(
       model,
       x,
       y,
       p0=[...],
       workflow="auto_global",
       bounds=bounds,
       n_starts=10,
       ftol=1e-8,
       xtol=1e-8,
       gtol=1e-8,
   )

Quality Preset
--------------

**Settings:**

- Workflow: ``auto_global``
- n_starts: 20
- Tolerances: ``1e-10``
- Thorough search
- Requires bounds

**When to use:**

- Publication-quality results needed
- Final fit for reports
- Maximum precision required

**Equivalent code:**

.. code-block:: python

   popt, pcov = fit(
       model,
       x,
       y,
       p0=[...],
       workflow="auto_global",
       bounds=bounds,
       n_starts=20,
       ftol=1e-10,
       xtol=1e-10,
       gtol=1e-10,
   )

Choosing a Preset
-----------------

.. code-block:: text

   Do you have good initial guess?
           │
      ┌────┴────┐
      │YES      │NO
      ▼         ▼
   ┌──────┐   Need publication quality?
   │ Fast │         │
   └──────┘    ┌────┴────┐
               │YES      │NO
               ▼         ▼
          ┌─────────┐ ┌────────┐
          │ Quality │ │ Robust │
          └─────────┘ └────────┘

Typical Workflow
----------------

1. **Start with Fast**: Quick check if model fits
2. **Use Robust if Fast fails**: Better exploration
3. **Finish with Quality**: Final publication results

Setting Bounds for Global Presets
---------------------------------

Robust and Quality presets require parameter bounds:

1. On Fitting Options page, click "Set Bounds"
2. For each parameter, enter lower and upper limits
3. Use physical constraints when known
4. Wider bounds = more exploration but slower

**Tips for setting bounds:**

- Amplitude: ``[0, max(y) * 2]``
- Rates/decay: ``[0, 10]`` or based on domain knowledge
- Centers: ``[min(x), max(x)]``
- Widths: ``[0.01, (max(x) - min(x))]``

Comparing Results
-----------------

After fitting with different presets, compare:

1. **Parameter values**: Should be similar
2. **Uncertainties**: Quality preset gives smaller errors
3. **Chi-squared**: Should be similar
4. **Fit time**: Fast << Robust < Quality

If results differ significantly between presets, the Robust/Quality
result is likely more reliable.

Next Steps
----------

- :doc:`../three_workflows/index` - Detailed workflow documentation
- :doc:`../troubleshooting/common_issues` - Troubleshooting

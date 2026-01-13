Visualization API
=================

NLSQ provides tools for generating publication-quality figures of fit results.

FitVisualizer
-------------

.. autoclass:: nlsq.cli.visualization.FitVisualizer
   :members: generate, _calculate_confidence_band
   :no-index:

The ``FitVisualizer`` class automates the creation of standardized plots.

**Style Presets:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Preset
     - Description
   * - ``publication``
     - (Default) Standard scientific publication style (serif fonts, 300 DPI)
   * - ``nature``
     - Nature journal specification (single column width, Arial font, no grid)
   * - ``science``
     - Science journal specification (sans-serif, compact)
   * - ``presentation``
     - Large fonts and thick lines suitable for slides/projectors
   * - ``minimal``
     - Clean look with no grid or top/right spines

**Example:**

.. code-block:: python

   from nlsq.cli.visualization import FitVisualizer

   visualizer = FitVisualizer()

   # Using a configuration dictionary
   config = {
       "visualization": {
           "style": "nature",
           "output_dir": "figures",
           "formats": ["pdf", "png"],
       }
   }

   # Generate plots
   paths = visualizer.generate(result, data, model, config)

CurveFitResult Plotting
-----------------------

The :class:`~nlsq.result.CurveFitResult` object returned by :func:`~nlsq.curve_fit` or :func:`~nlsq.fit` includes methods for quick verification plots.

.. method:: CurveFitResult.plot(ax=None, show_residuals=True, show_confidence=True)

   Plot the data, fitted curve, and confidence bands.

   :param ax: Optional matplotlib axes to plot on.
   :param show_residuals: If True, adds a residuals subplot.
   :param show_confidence: If True, plots the 95% confidence band (requires covariance).

.. method:: CurveFitResult.confidence_band(x, alpha=0.95)

   Calculate the confidence interval for the mean response at points ``x``.

   :param x: Input coordinate array.
   :param alpha: Confidence level (0 to 1, default 0.95).
   :return: (lower_bound, upper_bound) arrays.

Uses the Delta Method (error propagation using the Jacobian matrix) to estimate uncertainties.

See Also
--------

- :doc:`cli` - Using visualization from the command line

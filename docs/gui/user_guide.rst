GUI User Guide
==============

The NLSQ GUI provides an interactive graphical interface for performing
nonlinear least squares curve fitting with GPU/TPU acceleration.

.. contents:: On This Page
   :local:
   :depth: 2

Getting Started
---------------

Launching the GUI
~~~~~~~~~~~~~~~~~

There are three ways to launch the NLSQ GUI:

.. code-block:: bash

   # Using the CLI command (recommended)
   nlsq gui

   # Using Streamlit directly
   streamlit run nlsq/gui/app.py

   # As a desktop application (requires pywebview)
   python -m nlsq.gui.run_desktop

Workflow Overview
~~~~~~~~~~~~~~~~~

The GUI follows a sequential workflow:

1. **Data Loading** - Import your experimental data
2. **Model Selection** - Choose a fitting function
3. **Fitting Options** - Configure parameters and run the fit
4. **Results** - View fit statistics and visualizations
5. **Export** - Save results in various formats

Navigate between pages using the sidebar. Pages become accessible as you
complete earlier steps.

Theme Toggle
~~~~~~~~~~~~

Toggle between light and dark themes using the switch at the top of the sidebar.

Data Loading
------------

The Data Loading page is where you import your experimental data for curve fitting.

Supported Formats
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Format
     - Extensions
     - Description
   * - CSV
     - ``.csv``
     - Comma-separated values with optional header
   * - ASCII
     - ``.txt``, ``.dat``, ``.asc``
     - Whitespace or tab-delimited text files
   * - NumPy
     - ``.npz``
     - Compressed NumPy array files
   * - HDF5
     - ``.h5``, ``.hdf5``
     - Hierarchical Data Format 5

Loading from File
~~~~~~~~~~~~~~~~~

1. Click the file uploader or drag and drop your file
2. Select the file format (or leave as "auto" for automatic detection)
3. Preview the data in the table below

Loading from Clipboard
~~~~~~~~~~~~~~~~~~~~~~

1. Copy data from Excel, Google Sheets, or a text file
2. Click "Paste from Clipboard"
3. The GUI auto-detects the delimiter (tab, comma, or whitespace)

Column Selection
~~~~~~~~~~~~~~~~

After loading, assign columns to data roles:

- **X Column**: Independent variable (required)
- **Y Column**: Dependent variable (required)
- **Sigma Column**: Uncertainties/errors (optional)
- **Z Column**: For 2D surface data only

Use the dropdown selectors or click column headers in the preview table.

Data Validation
~~~~~~~~~~~~~~~

The GUI validates your data automatically:

- Checks for NaN (Not a Number) values
- Checks for Inf (Infinity) values
- Verifies minimum number of data points

Model Selection
---------------

The Model Selection page is where you choose the mathematical function to fit your data.

Built-in Models
~~~~~~~~~~~~~~~

NLSQ includes 7 pre-defined models:

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Model
     - Equation
     - Parameters
   * - Linear
     - :math:`y = ax + b`
     - a, b
   * - Exponential Decay
     - :math:`y = a \cdot e^{-bx} + c`
     - a, b, c
   * - Exponential Growth
     - :math:`y = a \cdot e^{bx} + c`
     - a, b, c
   * - Gaussian
     - :math:`y = A \cdot e^{-(x-\mu)^2/(2\sigma^2)}`
     - A, mu, sigma
   * - Lorentzian
     - :math:`y = \frac{A}{1 + ((x-x_0)/\gamma)^2}`
     - A, x0, gamma
   * - Sigmoid
     - :math:`y = \frac{L}{1 + e^{-k(x-x_0)}} + b`
     - L, k, x0, b
   * - Power Law
     - :math:`y = a \cdot x^b`
     - a, b

Polynomial Models
~~~~~~~~~~~~~~~~~

Select "Polynomial" and choose a degree from 0 to 10:

- Degree 0: Constant (y = c0)
- Degree 1: Linear (y = c0*x + c1)
- Degree 2: Quadratic (y = c0*x² + c1*x + c2)
- And so on...

Custom Models
~~~~~~~~~~~~~

Write your own model function in Python:

.. code-block:: python

   import jax.numpy as jnp


   def my_model(x, a, b, c):
       return a * jnp.exp(-b * x) * jnp.cos(c * x)

.. important::

   Use ``jax.numpy`` (or ``jnp``) instead of standard ``numpy`` for GPU
   acceleration and automatic differentiation.

Options for custom models:

- Type code directly in the editor
- Upload a ``.py`` file and select the function

Model Preview
~~~~~~~~~~~~~

After selection, view:

- LaTeX-rendered equation
- Parameter names
- Parameter count

Fitting Options
---------------

The Fitting Options page provides two modes for configuring the optimization.

Guided Mode
~~~~~~~~~~~

Use preset configurations for common use cases:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 45

   * - Preset
     - Tolerance
     - Multi-start
     - Use Case
   * - Fast
     - 1e-6
     - Disabled
     - Quick exploratory fits
   * - Robust
     - 1e-8
     - 10 starts
     - Reliable production fits
   * - Quality
     - 1e-10
     - 20 starts
     - High-precision requirements

Advanced Mode
~~~~~~~~~~~~~

Full control over all fitting parameters organized in tabs:

**Fitting Tab**

- **Method**: Trust Region Reflective (TRF), Levenberg-Marquardt (LM), Dogbox
- **Tolerances**: gtol, ftol, xtol
- **Max Iterations**: Maximum optimization iterations
- **Loss Function**: linear, huber, soft_l1, cauchy

**Multi-start Tab**

- **Enable/Disable**: Toggle global optimization
- **Number of Starts**: How many initial points to try
- **Sampler**: Latin Hypercube (LHS), Sobol, Halton
- **Center on p0**: Whether to include initial guess as one start

**Streaming Tab**

- **Chunk Size**: Points processed per iteration for large datasets
- **Normalize**: Auto-scale data for numerical stability
- **Defense Layers**: Numerical stability configurations

**HPC Tab**

- **Multi-device**: Enable distributed computing
- **Checkpointing**: Save progress for long-running fits

Initial Parameters (p0)
~~~~~~~~~~~~~~~~~~~~~~~

For each model parameter:

- Enter a manual initial guess, or
- Check "Auto" to use the model's ``estimate_p0`` method

Parameter Bounds
~~~~~~~~~~~~~~~~

Optionally constrain parameters:

- Set lower and upper bounds per parameter
- Leave empty for unbounded optimization

Running the Fit
~~~~~~~~~~~~~~~

1. Configure options
2. Click "Run Fit"
3. Monitor progress in real-time:

   - Progress bar with iteration count
   - Live cost function plot
   - Current parameter values

Click "Abort" to cancel a running fit.

Results Display
---------------

The Results page shows comprehensive fit results after optimization completes.

Fit Summary
~~~~~~~~~~~

- **Success Status**: Whether optimization converged
- **Model Name**: The fitted function
- **Data Points**: Number of points used

Parameter Table
~~~~~~~~~~~~~~~

For each parameter:

- Optimal value
- Uncertainty (from covariance matrix diagonal)
- 95% Confidence interval

Fit Statistics
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Statistic
     - Description
   * - R-squared
     - Coefficient of determination (0-1)
   * - Adjusted R-squared
     - R-squared adjusted for number of parameters
   * - RMSE
     - Root mean squared error
   * - MAE
     - Mean absolute error
   * - AIC
     - Akaike Information Criterion
   * - BIC
     - Bayesian Information Criterion

Convergence Information
~~~~~~~~~~~~~~~~~~~~~~~

- Number of function evaluations (nfev)
- Final cost value
- Optimality measure
- Convergence message

Interactive Plots
~~~~~~~~~~~~~~~~~

All plots are interactive (zoom, pan, hover):

1. **Main Fit Plot**

   - Data points (scatter)
   - Fitted curve (line)
   - 95% confidence band (shaded region)

2. **Residuals Plot**

   - Residuals vs x
   - Zero reference line
   - Standard deviation bands (+/- 1 std, +/- 2 std)

3. **Residuals Histogram**

   - Distribution of residuals
   - Normal distribution overlay
   - Skewness and kurtosis statistics

Export
------

The Export page allows saving results in multiple formats.

Session Bundle (ZIP)
~~~~~~~~~~~~~~~~~~~~

Download a complete session bundle containing:

- Data snapshot (CSV)
- Configuration (YAML)
- Results (JSON)
- Plots (PNG and/or PDF)

Individual Exports
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Format
     - Description
   * - JSON
     - Full results with parameters, statistics, metadata
   * - CSV
     - Parameter table with values and uncertainties
   * - HTML
     - Interactive Plotly charts (self-contained)

Python Code Generation
~~~~~~~~~~~~~~~~~~~~~~

Generate a standalone Python script that reproduces the fit:

- Includes all imports
- Data definition
- Model function
- curve_fit call with parameters

Use the "Copy to Clipboard" button or download as a ``.py`` file.

Troubleshooting
---------------

Data Loading Issues
~~~~~~~~~~~~~~~~~~~

**Problem**: "Invalid data format" error

- Ensure numeric columns contain only numbers
- Check for missing values or text in data rows
- Verify the correct delimiter is detected

**Problem**: NaN values detected

- Remove or impute missing values before loading
- Check for divide-by-zero in your source data

Fitting Issues
~~~~~~~~~~~~~~

**Problem**: Fit fails to converge

- Try different initial parameters (p0)
- Enable multi-start optimization
- Add parameter bounds to constrain the search space
- Use the "Robust" or "Quality" preset

**Problem**: Poor fit quality (low R-squared)

- Verify you selected the appropriate model
- Check data for outliers
- Consider using a different loss function (e.g., 'huber' for robust fitting)

**Problem**: Fit takes too long

- For large datasets, enable streaming optimization
- Reduce max_iterations
- Use the "Fast" preset for exploratory analysis

Performance Tips
~~~~~~~~~~~~~~~~

1. **Cold Start**: The first fit may take longer due to JIT compilation.
   Subsequent fits reuse compiled code.

2. **Large Datasets**: For datasets over 1M points, the streaming optimizer
   is automatically enabled.

3. **Memory**: Monitor memory usage for very large datasets. Consider using
   data chunking.

Getting Help
~~~~~~~~~~~~

- GitHub Issues: https://github.com/imewei/NLSQ/issues
- Documentation: https://nlsq.readthedocs.io

See Also
--------

- :doc:`/tutorials/01_first_fit` - Python API tutorial
- :doc:`/reference/index` - API reference
- :doc:`/howto/troubleshooting` - General troubleshooting guide

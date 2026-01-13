Workflow Pages
==============

The GUI guides you through a 5-page workflow for curve fitting.

Page 1: Data Loading
--------------------

**Purpose**: Import your data into NLSQ.

**Features:**

- Load CSV files (drag-and-drop or file dialog)
- Paste from clipboard
- Preview data table
- Select X and Y columns
- View data statistics

**How to use:**

1. Click "Open File" or drag a CSV onto the window
2. Select the X column (independent variable)
3. Select the Y column (dependent variable)
4. Optionally select a sigma column (uncertainties)
5. Click "Next" or Ctrl+2

**Supported formats:**

- CSV (comma, tab, or semicolon separated)
- TSV
- Excel (xlsx) with openpyxl installed

Page 2: Model Selection
-----------------------

**Purpose**: Choose a mathematical model for fitting.

**Features:**

- Built-in models (exponential, Gaussian, etc.)
- Polynomial (select degree)
- Custom Python model (advanced)
- Model preview with formula

**How to use:**

1. Select a model category (Decay, Peak, Growth, etc.)
2. Choose a specific model
3. View the mathematical formula
4. Click "Next" or Ctrl+3

**Built-in models:**

- Exponential decay
- Gaussian peak
- Lorentzian peak
- Logistic growth
- Power law
- Polynomial (degree 1-10)

Page 3: Fitting Options
-----------------------

**Purpose**: Configure fit parameters and run the fit.

**Features:**

- Workflow preset selection (Fast/Robust/Quality)
- Initial parameter guess (auto or manual)
- Parameter bounds (optional)
- Advanced options (tolerances, iterations)
- Live cost function plot

**How to use:**

1. Select a preset:
   - **Fast**: Quick fit with default settings
   - **Robust**: Multi-start global optimization
   - **Quality**: Thorough search with tight tolerances

2. Set initial guess (or use auto-estimate)

3. Optionally set bounds on parameters

4. Click "Run Fit" (or Ctrl+R)

5. Watch the live cost plot during fitting

**Guided vs Advanced Mode:**

- **Guided**: Simple preset selection
- **Advanced**: Full control over tolerances, iterations, etc.

Page 4: Results
---------------

**Purpose**: View and analyze fit results.

**Features:**

- Fitted parameters with uncertainties
- Goodness-of-fit statistics (R², chi-squared)
- Interactive fit plot
- Residual plot
- Correlation matrix

**How to use:**

1. Review fitted parameters
2. Examine the fit plot (zoom, pan, export)
3. Check residuals for systematic patterns
4. View statistics panel
5. Click "Next" to export or "Back" to adjust

**Quality indicators:**

- R² close to 1.0 indicates good fit
- Reduced chi-squared ~1.0 with known uncertainties
- Random residuals (no patterns)

Page 5: Export
--------------

**Purpose**: Save results in various formats.

**Features:**

- ZIP session bundle (complete backup)
- JSON (parameters and metadata)
- CSV (data with fit curve)
- Python code generation
- Plot export (PNG, PDF, SVG)

**Export formats:**

1. **ZIP Bundle**: Complete session including:
   - Original data
   - Fitted parameters
   - Covariance matrix
   - Settings and configuration
   - Session state for reload

2. **JSON**: Structured results:

   .. code-block:: json

      {
        "parameters": {"A": 2.5, "k": 0.5, "c": 0.3},
        "uncertainties": {"A": 0.02, "k": 0.01, "c": 0.005},
        "statistics": {"r_squared": 0.998, "chi_squared": 45.2}
      }

3. **CSV**: Data table with fit values:

   .. code-block:: text

      x,y,y_fit,residual
      0,1.0,1.02,-0.02
      1,0.62,0.61,0.01
      ...

4. **Python Code**: Reproducible script:

   .. code-block:: python

      from nlsq import fit
      import jax.numpy as jnp


      def model(x, A, k, c):
          return A * jnp.exp(-k * x) + c


      popt, pcov = fit(model, xdata, ydata, p0=[2.5, 0.5, 0.3])

Navigation
----------

**Keyboard shortcuts:**

- Ctrl+1 to Ctrl+5: Jump to specific page
- Ctrl+R: Run fit (from any page)
- Ctrl+S: Save session

**Page guards:**

- Cannot proceed to Results without running fit
- Cannot export without valid results

Next Steps
----------

- :doc:`presets` - Learn about fitting presets
- :doc:`../troubleshooting/common_issues` - Troubleshooting

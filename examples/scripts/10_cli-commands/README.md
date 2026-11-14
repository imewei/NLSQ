# NLSQ CLI Command Demonstrations

This directory contains comprehensive examples demonstrating all NLSQ command-line interface features.

## Quick Start

```bash
cd examples/scripts/10_cli-commands

# Step 0: generate the sample data. polynomial_data.npz and large_dataset.h5
# (workflows 04 and 08) are gitignored and only exist after this step.
python data/generate_sample_data.py

./run_all_demos.sh
```

The generator is seeded (`np.random.seed(42)`), so it reproduces the committed
CSV/TXT files byte for byte. `run_all_demos.sh` runs it automatically when
`data/polynomial_data.npz` or `data/large_dataset.h5` is missing.

## Directory Structure

```
10_cli-commands/
├── README.md                 # This file
├── run_all_demos.sh          # Master script running all demos
├── aggregate_batch_results.py  # Collect batch results into one table
├── data/                     # Sample datasets
│   ├── generate_sample_data.py   # Seeded generator for everything below
│   ├── exponential_decay.csv
│   ├── gaussian_peak.txt
│   ├── damped_oscillation.csv
│   ├── enzyme_kinetics.csv
│   ├── radioactive_decay.csv
│   ├── reaction_kinetics.csv
│   ├── surface_2d.txt
│   ├── batch_data_*.csv
│   ├── polynomial_data.npz       # generated (gitignored)
│   └── large_dataset.h5          # generated (gitignored)
├── models/                   # Custom model definitions
│   ├── physics_models.py
│   ├── biology_models.py
│   ├── engineering_models.py
│   └── custom_models.py
├── workflows/                # YAML workflow configurations
│   ├── 01_basic_fit.yaml
│   ├── 02_builtin_models.yaml
│   ├── 03_custom_model.yaml
│   ├── 04_polynomial_fit.yaml
│   ├── 05_weighted_fit.yaml
│   ├── 06_global_optimization.yaml  # historical name: bounded local fit
│   ├── 07_2d_surface_fit.yaml
│   ├── 08_hdf5_data.yaml
│   └── batch_example/
│       ├── batch_workflow_1.yaml
│       ├── batch_workflow_2.yaml
│       └── batch_workflow_3.yaml
└── output/                   # Generated results (created at runtime)
```

## CLI Commands

### `nlsq info`

Display system information including JAX version, available devices, and GPU status.

```bash
nlsq info
```

### `nlsq config`

Generate configuration templates for workflows and custom models.

```bash
# Show help
nlsq config --help

# Generate workflow template
nlsq config --workflow -o my_workflow.yaml

# Generate custom model template
nlsq config --model -o my_model.py

# Force overwrite existing files
nlsq config --workflow -o my_workflow.yaml -f
```

### `nlsq fit`

Execute curve fitting from a YAML workflow configuration.

```bash
# Basic usage
nlsq fit workflow.yaml

# Override output file
nlsq fit workflow.yaml --output results.json

# Output JSON to stdout (for piping)
nlsq fit workflow.yaml --stdout
```

### `nlsq batch`

Process multiple workflow files in parallel.

```bash
# Process all YAML files in directory
nlsq batch workflows/*.yaml

# Generate summary file
nlsq batch workflows/*.yaml --summary summary.json

# Control parallelism
nlsq batch workflows/*.yaml --workers 4
```

Note that `--verbose` is a global flag, so it goes *before* the subcommand:
`nlsq --verbose batch workflows/*.yaml`.

The summary file records counts and failures only. Fitted parameters go to each
workflow's own `export.results_file`; see `aggregate_batch_results.py` below for
collecting them into one table.

### `aggregate_batch_results.py`

Collect the per-workflow result files a batch run produced into a single
parameter table, optionally as CSV.

```bash
nlsq batch workflows/batch_example/*.yaml --summary output/batch_summary.json
python aggregate_batch_results.py output --summary output/batch_summary.json --csv output/batch_parameters.csv
```

```text
workflow              status                     a                 b                 c     R^2
----------------------------------------------------------------------------------------------
batch_exp_decay_1     success         5.58 +-0.0693    0.3954 +-0.0127    0.9781 +-0.0449   0.993
batch_exp_decay_2     success        6.061 +-0.104     0.5344 +-0.0197     1.091 +-0.0446   0.988
batch_exp_decay_3     success        6.547 +-0.111     0.5816 +-0.0204    0.9538 +-0.0431   0.989
```

Passing `--summary` also reports the workflows that failed, since those leave no
result file of their own.

## Data Formats

### CSV (Comma-Separated Values)

```yaml
data:
  input_file: "data/exponential_decay.csv"
  format: "csv"
  columns:
    x: 0        # Column index for x data
    y: 1        # Column index for y data
    sigma: 2    # Column index for uncertainties (optional)
  csv:
    header: true
    delimiter: ","
```

### ASCII Text

```yaml
data:
  input_file: "data/damped_oscillation.txt"
  format: "ascii"
  columns:
    x: 0
    y: 1
  ascii:
    skip_rows: 1
    delimiter: "whitespace"
```

### NPZ (NumPy Archive)

```yaml
data:
  input_file: "data/polynomial_data.npz"
  format: "npz"
  npz:
    x_key: "x"
    y_key: "y"
    sigma_key: "sigma"  # Optional
```

### HDF5

```yaml
data:
  input_file: "data/large_dataset.h5"
  format: "hdf5"
  hdf5:
    x_path: "/data/x"
    y_path: "/data/y"
    sigma_path: "/data/sigma"  # Optional
```

### 2D Surface Data (CSV)

For 2D fitting, provide x, y coordinates and z values:

```yaml
data:
  input_file: "data/beam_profile_2d.csv"
  format: "csv"
  columns:
    x: 0    # X coordinate
    y: 1    # Y coordinate (becomes second row of xdata)
    z: 2    # Z value (the dependent variable)
```

## Model Types

### Builtin Models

Available models from `nlsq.functions`:

| Name                 | Function                        | Parameters        |
|----------------------|---------------------------------|-------------------|
| `linear`             | `a*x + b`                       | a, b              |
| `exponential_decay`  | `a*exp(-b*x) + c`               | a, b, c           |
| `exponential_growth` | `a*exp(b*x) + c`                | a, b, c           |
| `gaussian`           | `amp*exp(-(x-mu)^2/(2*sigma^2))`| amp, mu, sigma    |
| `lorentzian`         | Lorentzian peak function        | amp, x0, gamma    |
| `sigmoid`            | `L/(1+exp(-k*(x-x0))) + b`      | L, x0, k, b       |
| `power_law`          | `a*x^b`                         | a, b              |

```yaml
model:
  type: "builtin"
  name: "exponential_decay"
fitting:
  p0: auto          # Use the model's estimate_p0 helper
# Omitted fitting.bounds uses the model's bounds helper when available.
```

### Custom Models

Define models in Python files with JAX-compatible functions:

```python
# models/my_model.py
import jax.numpy as jnp


def my_model(x, a, b, c):
    """Custom model function."""
    return a * jnp.exp(-b * x) * jnp.cos(c * x)


def estimate_p0(x, y):
    """Optional: automatic initial parameter estimation."""
    return [jnp.max(y), 0.1, 1.0]


def bounds():
    """Optional: parameter bounds."""
    return ([0, 0, 0], [jnp.inf, jnp.inf, jnp.inf])
```

```yaml
model:
  type: "custom"
  path: "models/my_model.py"          # Resolved relative to the CWD, not the workflow file
  function: "my_model"
  # estimate_p0() and bounds() in the module are picked up automatically
  # (no separate p0_function/bounds_function keys)
```

### Polynomial Models

```yaml
model:
  type: "polynomial"
  degree: 3
```

## Workflow Examples

### 1. Basic Fit (01_basic_fit.yaml)

Simple exponential decay with builtin model:

```yaml
model:
  type: "builtin"
  name: "exponential_decay"

fitting:
  p0: auto
  ftol: 1.0e-8
  max_nfev: 100
```

### 2. Custom Model (03_custom_model.yaml)

Physics model with automatic initial guesses:

```yaml
model:
  type: "custom"
  path: "models/physics_models.py"
  function: "damped_oscillator"
```

### 3. Weighted Fitting (05_weighted_fit.yaml)

Include measurement uncertainties:

```yaml
data:
  columns:
    x: 0
    y: 1
    sigma: 2  # Uncertainties column

fitting:
  absolute_sigma: true
```

### 4. Bounded Fitting (06_global_optimization.yaml)

This historical filename now contains a bounded local fit with explicit
`fitting.p0` and `fitting.bounds`. The CLI calls `curve_fit`; it does not dispatch
multi-start or CMA-ES from YAML. For global optimization, use the Python API
examples in `../07_global_optimization/`.

### 5. 2D Surface Fitting (07_2d_surface_fit.yaml)

Fit 2D functions (beam profiles, surfaces):

```yaml
data:
  columns:
    x: 0  # X coordinate
    y: 1  # Y coordinate
    z: 2  # Z value (dependent variable)

model:
  type: "custom"
  path: "models/physics_models.py"
  function: "gaussian_2d"
```

### 6. Batch Processing

Process multiple datasets with shared configuration:

```bash
nlsq batch workflows/batch_example/*.yaml --summary output/batch_summary.json
python aggregate_batch_results.py output --summary output/batch_summary.json
```

Workflows run in parallel on a thread pool (not processes - JAX is not
fork-safe on GPU). Give each workflow a distinct `export.results_file` and
`visualization.filename_prefix`, or concurrent workflows will overwrite each
other's output.

## Visualization

All workflow examples include comprehensive visualization configurations that automatically generate publication-quality plots.

### Generated Outputs

Each workflow produces:
- **Combined plot**: Main fit + residuals in a single figure
- **Histogram**: Distribution of residuals with normal fit overlay
- **Multi-format**: Both PNG (raster) and PDF (vector) outputs

### Style Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `publication` | Serif fonts, 300 DPI, clean layout | Journal papers |
| `presentation` | Large sans-serif fonts, 150 DPI | Slides, talks |
| `nature` | 3.5" width, Arial font | Nature journal |
| `science` | Science journal specifications | Science journal |
| `minimal` | No spines/grid, clean look | Modern presentations |

### Color Schemes

| Scheme | Description |
|--------|-------------|
| `default` | Standard matplotlib colors |
| `colorblind` | Okabe-Ito palette (accessible) |
| `high_contrast` | Maximum contrast for projectors |
| `grayscale` | B&W printing compatible |
| `nature` | Nature journal style |
| `science` | Science journal style |

### Visualization Configuration Example

```yaml
visualization:
  enabled: true
  output_dir: "output/figures"
  filename_prefix: "my_fit"
  formats: ["png", "pdf"]
  dpi: 300
  figure_size: [6.0, 4.5]
  style: "publication"

  font:
    family: "serif"
    size: 10
    math_fontset: "cm"

  main_plot:
    x_label: "Time (s)"
    y_label: "Signal (V)"
    show_grid: true

    data:
      marker: "o"
      size: 20
      alpha: 0.7
      show_errorbars: true

    fit:
      linewidth: 1.5
      n_points: 500

    # Confidence band from covariance matrix error propagation
    confidence_band:
      enabled: true
      level: 0.95          # 95% confidence interval
      alpha: 0.2

    # Fit statistics annotation (R², RMSE, χ²)
    annotation:
      enabled: true
      show_r_squared: true
      show_rmse: true
      show_chi_squared: true
      location: "upper right"

  residuals_plot:
    enabled: true
    show_zero_line: true

    # Standard deviation bands
    std_bands:
      enabled: true
      levels: [1, 2]       # ±1σ and ±2σ

  # Histogram of residuals
  histogram:
    enabled: true
    bins: "auto"
    show_normal_fit: true

  # Colorblind-safe palette
  active_scheme: "colorblind"
```

### Demo Visualization Styles

Each demo showcases different visualization options:

| Workflow | Style | Color Scheme | Special Features |
|----------|-------|--------------|------------------|
| 01_basic_fit | `publication` | `default` | All features enabled |
| 02_builtin_models | `nature` | `colorblind` | Nature journal format |
| 03_custom_model | `science` | `science` | LaTeX math labels |
| 04_polynomial | `minimal` | `high_contrast` | No grid, square markers |
| 05_weighted_fit | `presentation` | `default` | 16:9 slides |
| 06_global_opt | `publication` | none set (library default) | Bounded fit, combined plot only |
| 07_2d_surface | `publication` | `colorblind` | Plots disabled (the CLI plotter is 1D only) |
| 08_hdf5_data | `publication` | `default` | Large dataset (10K points) |

## Output Format

Results are saved as JSON (real output of `workflows/01_basic_fit.yaml`, with
`pcov` trimmed):

```json
{
  "popt": [5.0176, 0.50042, 1.0079],
  "pcov": [[0.00147, 0.00012, -6.2e-05], [0.00012, 7.0e-05, 0.00011], [-6.2e-05, 0.00011, 0.00030]],
  "uncertainties": [0.0384, 0.00837, 0.0173],
  "statistics": {
    "chi_squared": 77.199,
    "rmse": 0.08786,
    "r_squared": 0.99518,
    "cost": 38.600
  },
  "convergence": {
    "iterations": 4,
    "function_evals": 5,
    "jacobian_evals": 5,
    "status": "success",
    "message": "`ftol` termination condition is satisfied.",
    "final_cost": 38.600
  },
  "metadata": {
    "workflow_name": "basic_fit_example",
    "dataset_id": "unknown",
    "model_id": "exponential_decay"
  }
}
```

`popt` is ordered like the model's parameters. The export section accepts
`results_file` (and `format`); there is no key that selects which of the blocks
above are written.

## Batch Summary

When using `nlsq batch --summary`, a JSON summary is generated:

```json
{
  "total": 4,
  "succeeded": 3,
  "failed": 1,
  "skipped": 0,
  "skipped_workflows": [],
  "cancelled": 0,
  "cancelled_workflows": [],
  "duration_seconds": 4.07,
  "start_time": "2026-09-03T18:12:02.723038",
  "end_time": "2026-09-03T18:12:07.671942",
  "max_workers": 3,
  "failures": [
    {
      "workflow_path": "workflows/batch_workflow_bad.yaml",
      "error": {
        "type": "DataLoadError",
        "message": "Data file not found: data/does_not_exist.csv",
        "context": {"file_path": "data/does_not_exist.csv"},
        "suggestion": "Check that the file path is correct and the file exists."
      }
    }
  ],
  "successes": [
    {"workflow_path": "workflows/batch_workflow_1.yaml"}
  ]
}
```

Note that `successes` carries workflow paths only - no parameters. Use
`aggregate_batch_results.py` to build a parameter table from the individual
result files.

## Troubleshooting

### Common Issues

1. **"Model function not found"**: Ensure the function name in YAML matches the Python function exactly.

2. **"Data file not found"**: Paths in the workflow are resolved relative to the
   current working directory, not to the YAML file. Run from this directory.

3. **"Convergence failed"**: Try:
   - Adjusting initial parameters (`p0`)
   - Using global optimization
   - Checking data quality

4. **"Invalid bounds"**: Ensure lower bounds < upper bounds for all parameters.

### Debug Mode

Enable verbose output for troubleshooting:

```yaml
fitting:
  verbose: 2  # 0=silent, 1=summary, 2=detailed
```

## See Also

- [NLSQ Documentation](https://nlsq.readthedocs.io/)
- [CLI Reference](../../../docs/user_guide/cli_reference.md)
- [Workflow Template](../../../nlsq/cli/templates/workflow_config_template.yaml)

# NLSQ CLI Command Demonstrations

This directory contains comprehensive examples demonstrating all NLSQ command-line interface features.

## Quick Start

```bash
cd examples/scripts/10_cli-commands
./run_all_demos.sh
```

## Directory Structure

```
10_cli-commands/
├── README.md                 # This file
├── run_all_demos.sh          # Master script running all demos
├── data/                     # Sample datasets
│   ├── generate_sample_data.py
│   ├── exponential_decay.csv
│   ├── gaussian_peak.csv
│   ├── damped_oscillation.txt
│   ├── polynomial_data.npz
│   ├── enzyme_kinetics.csv
│   ├── double_gaussian.csv
│   ├── beam_profile_2d.csv
│   ├── large_dataset.h5
│   └── batch_data_*.csv
├── models/                   # Custom model definitions
│   ├── physics_models.py
│   ├── biology_models.py
│   └── engineering_models.py
├── workflows/                # YAML workflow configurations
│   ├── 01_basic_fit.yaml
│   ├── 02_builtin_models.yaml
│   ├── 03_custom_model.yaml
│   ├── 04_polynomial_fit.yaml
│   ├── 05_weighted_fit.yaml
│   ├── 06_global_optimization.yaml
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
  auto_p0: true      # Automatic initial parameter estimation
  auto_bounds: true  # Automatic parameter bounds
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
  custom:
    file: "models/my_model.py"
    function: "my_model"
    p0_function: "estimate_p0"      # Optional
    bounds_function: "bounds"       # Optional
```

### Polynomial Models

```yaml
model:
  type: "polynomial"
  polynomial:
    degree: 3
```

## Workflow Examples

### 1. Basic Fit (01_basic_fit.yaml)

Simple exponential decay with builtin model:

```yaml
model:
  type: "builtin"
  name: "exponential_decay"
  auto_p0: true

fitting:
  termination:
    ftol: 1.0e-8
    max_iterations: 100
```

### 2. Custom Model (03_custom_model.yaml)

Physics model with automatic initial guesses:

```yaml
model:
  type: "custom"
  custom:
    file: "../models/physics_models.py"
    function: "damped_oscillator"
    p0_function: "estimate_p0"
    bounds_function: "bounds"
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
  use_weights: true
```

### 4. Global Optimization (06_global_optimization.yaml)

Multi-start optimization for complex landscapes:

```yaml
fitting:
  global_optimization:
    n_starts: 15
    sampling_method: "latin_hypercube"
```

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
  custom:
    function: "gaussian_2d"
```

### 6. Batch Processing

Process multiple datasets with shared configuration:

```bash
nlsq batch workflows/batch_example/*.yaml --summary output/batch_summary.json
```

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
  layout: "combined"

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
| 05_weighted_fit | `presentation` | `default` | 16:9 slides, weighted residuals |
| 06_global_opt | `publication` | `grayscale` | B&W printing |
| 07_2d_surface | `publication` | `colorblind` | Dense data visualization |
| 08_hdf5_data | `publication` | `default` | Large dataset (10K points) |

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "metadata": {
    "workflow_name": "basic_fit",
    "timestamp": "2024-01-15T10:30:00",
    "nlsq_version": "1.0.0"
  },
  "parameters": {
    "names": ["amplitude", "decay_rate", "offset"],
    "values": [10.5, 0.15, 0.5],
    "uncertainties": [0.1, 0.005, 0.02]
  },
  "statistics": {
    "chi_squared": 45.2,
    "reduced_chi_squared": 1.02,
    "r_squared": 0.998,
    "rmse": 0.12
  },
  "convergence": {
    "success": true,
    "iterations": 15,
    "termination_reason": "ftol"
  }
}
```

## Batch Summary

When using `nlsq batch --summary`, a JSON summary is generated:

```json
{
  "total": 4,
  "succeeded": 4,
  "failed": 0,
  "duration_seconds": 2.5,
  "results": [...]
}
```

## Troubleshooting

### Common Issues

1. **"Model function not found"**: Ensure the function name in YAML matches the Python function exactly.

2. **"Data file not found"**: Use relative paths from the workflow file location.

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

#!/usr/bin/env bash
# ==============================================================================
# NLSQ CLI Demonstration Script
# ==============================================================================
# This script demonstrates all NLSQ CLI commands and features.
#
# Usage:
#   cd examples/scripts/10_cli-commands
#   ./run_all_demos.sh
#
# Prerequisites:
#   - NLSQ installed: pip install nlsq
#   - Run from this directory (10_cli-commands)
#
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to script directory
cd "$(dirname "$0")"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       NLSQ CLI COMMAND DEMONSTRATIONS${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Create output directory
mkdir -p output

# ==============================================================================
# Demo 1: nlsq info
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 1: nlsq info - System Information${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Command: nlsq info"
echo ""
nlsq info
echo ""

# ==============================================================================
# Demo 2: nlsq config
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 2: nlsq config - Generate Configuration Templates${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Command: nlsq config --help"
nlsq config --help
echo ""

echo "Generating workflow template..."
echo "Command: nlsq config --workflow -o output/generated_workflow.yaml -f"
nlsq config --workflow -o output/generated_workflow.yaml -f
echo -e "${GREEN}Created: output/generated_workflow.yaml${NC}"
echo ""

echo "Generating custom model template..."
echo "Command: nlsq config --model -o output/generated_model.py -f"
nlsq config --model -o output/generated_model.py -f
echo -e "${GREEN}Created: output/generated_model.py${NC}"
echo ""

# ==============================================================================
# Demo 3: Basic nlsq fit
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 3: nlsq fit - Basic Curve Fitting${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Fitting exponential decay data with builtin model..."
echo "Command: nlsq fit workflows/01_basic_fit.yaml"
echo ""
nlsq fit workflows/01_basic_fit.yaml
echo ""
echo -e "${GREEN}Results saved to: output/01_basic_fit_results.json${NC}"
echo ""

# ==============================================================================
# Demo 4: Builtin models
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 4: nlsq fit - Builtin Gaussian Model${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Fitting Gaussian peak with builtin gaussian model..."
echo "Command: nlsq fit workflows/02_builtin_models.yaml"
echo ""
nlsq fit workflows/02_builtin_models.yaml
echo ""
echo -e "${GREEN}Results saved to: output/02_builtin_models_results.json${NC}"
echo ""

# ==============================================================================
# Demo 5: Custom model
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 5: nlsq fit - Custom Model from Python File${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Fitting damped oscillation with custom physics model..."
echo "Command: nlsq fit workflows/03_custom_model.yaml"
echo ""
nlsq fit workflows/03_custom_model.yaml
echo ""
echo -e "${GREEN}Results saved to: output/03_custom_model_results.json${NC}"
echo ""

# ==============================================================================
# Demo 6: Polynomial fit
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 6: nlsq fit - Polynomial Model (NPZ data)${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Fitting polynomial data from NPZ file..."
echo "Command: nlsq fit workflows/04_polynomial_fit.yaml"
echo ""
nlsq fit workflows/04_polynomial_fit.yaml
echo ""
echo -e "${GREEN}Results saved to: output/04_polynomial_fit_results.json${NC}"
echo ""

# ==============================================================================
# Demo 7: Weighted fit
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 7: nlsq fit - Weighted Fitting with Uncertainties${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Fitting enzyme kinetics with measurement uncertainties..."
echo "Command: nlsq fit workflows/05_weighted_fit.yaml"
echo ""
nlsq fit workflows/05_weighted_fit.yaml
echo ""
echo -e "${GREEN}Results saved to: output/05_weighted_fit_results.json${NC}"
echo ""

# ==============================================================================
# Demo 8: Global optimization
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 8: nlsq fit - Multi-Start Global Optimization${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Fitting with multi-start global optimization (15 starts)..."
echo "Command: nlsq fit workflows/06_global_optimization.yaml"
echo ""
nlsq fit workflows/06_global_optimization.yaml
echo ""
echo -e "${GREEN}Results saved to: output/06_global_optimization_results.json${NC}"
echo ""

# ==============================================================================
# Demo 9: 2D surface fitting
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 9: nlsq fit - 2D Surface Fitting${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Fitting 2D Gaussian surface (beam profile)..."
echo "Command: nlsq fit workflows/07_2d_surface_fit.yaml"
echo ""
nlsq fit workflows/07_2d_surface_fit.yaml
echo ""
echo -e "${GREEN}Results saved to: output/07_2d_surface_fit_results.json${NC}"
echo ""

# ==============================================================================
# Demo 10: HDF5 data
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 10: nlsq fit - HDF5 Data Format${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Fitting large dataset from HDF5 file (10,000 points)..."
echo "Command: nlsq fit workflows/08_hdf5_data.yaml"
echo ""
nlsq fit workflows/08_hdf5_data.yaml
echo ""
echo -e "${GREEN}Results saved to: output/08_hdf5_data_results.json${NC}"
echo ""

# ==============================================================================
# Demo 11: Output to stdout
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 11: nlsq fit --stdout - JSON Output to Terminal${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Outputting results as JSON to stdout (pipe to jq for formatting)..."
echo "Command: nlsq fit workflows/01_basic_fit.yaml --stdout | head -20"
echo ""
nlsq fit workflows/01_basic_fit.yaml --stdout 2>/dev/null | head -20
echo "..."
echo ""

# ==============================================================================
# Demo 12: Batch processing
# ==============================================================================
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Demo 12: nlsq batch - Parallel Batch Processing${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo "Processing multiple datasets in parallel..."
echo "Command: nlsq batch workflows/batch_example/*.yaml --summary output/batch_summary.json"
echo ""
nlsq batch workflows/batch_example/*.yaml --summary output/batch_summary.json
echo ""
echo -e "${GREEN}Individual results saved to: output/batch_results_*.json${NC}"
echo -e "${GREEN}Summary saved to: output/batch_summary.json${NC}"
echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       DEMONSTRATION COMPLETE${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "All demonstrations completed successfully!"
echo ""
echo "Generated output files:"
ls -la output/
echo ""
echo "CLI Commands Demonstrated:"
echo "  - nlsq info           : Display system information"
echo "  - nlsq config         : Generate configuration templates"
echo "  - nlsq fit            : Execute single curve fit"
echo "  - nlsq fit --output   : Override output file path"
echo "  - nlsq fit --stdout   : Output results to stdout"
echo "  - nlsq batch          : Execute parallel batch fitting"
echo "  - nlsq batch --summary: Generate aggregate summary"
echo ""
echo "Data Formats Demonstrated:"
echo "  - CSV with header"
echo "  - ASCII text"
echo "  - NPZ (NumPy archive)"
echo "  - HDF5"
echo ""
echo "Model Types Demonstrated:"
echo "  - Builtin models (exponential_decay, gaussian)"
echo "  - Custom models (damped_oscillator, michaelis_menten, gaussian_2d)"
echo "  - Polynomial models"
echo ""
echo "Features Demonstrated:"
echo "  - Weighted fitting with uncertainties"
echo "  - Multi-start global optimization"
echo "  - 2D surface fitting"
echo "  - Batch processing"
echo ""
echo -e "${GREEN}See README.md for detailed documentation.${NC}"

#!/bin/bash
# ==============================================================================
# NLSQ CLI Examples - Shell Script Demonstration
# ==============================================================================
# This script demonstrates various NLSQ CLI commands and options.
#
# Usage:
#   chmod +x run_cli_examples.sh
#   ./run_cli_examples.sh
# ==============================================================================

set -e  # Exit on first error

# Change to script directory
cd "$(dirname "$0")"

echo "=============================================================="
echo "NLSQ CLI Examples"
echo "=============================================================="
echo

# ==============================================================================
# Step 1: Generate sample data
# ==============================================================================
echo "Step 1: Generating sample data files..."
echo "--------------------------------------------------------------"
python generate_data.py
echo

# ==============================================================================
# Step 2: Display system information
# ==============================================================================
echo "Step 2: System Information (nlsq info)"
echo "--------------------------------------------------------------"
nlsq info
echo

# ==============================================================================
# Step 3: Run single fit - Radioactive Decay
# ==============================================================================
echo "Step 3: Single Fit - Radioactive Decay"
echo "--------------------------------------------------------------"
echo "Command: nlsq fit workflows/01_radioactive_decay.yaml"
echo
nlsq fit workflows/01_radioactive_decay.yaml
echo
echo "Results saved to: output/radioactive_decay_results.json"
echo

# ==============================================================================
# Step 4: Run single fit with verbose output
# ==============================================================================
echo "Step 4: Verbose Fit - Reaction Kinetics"
echo "--------------------------------------------------------------"
echo "Command: nlsq fit workflows/03_reaction_kinetics.yaml -v"
echo
nlsq fit workflows/03_reaction_kinetics.yaml -v
echo

# ==============================================================================
# Step 5: Run single fit with stdout output (for piping)
# ==============================================================================
echo "Step 5: Stdout Output (for piping)"
echo "--------------------------------------------------------------"
echo "Command: nlsq fit workflows/01_radioactive_decay.yaml --stdout"
echo
echo "Output (first 500 chars):"
nlsq fit workflows/01_radioactive_decay.yaml --stdout | head -c 500
echo
echo "..."
echo

# ==============================================================================
# Step 6: Run batch fitting
# ==============================================================================
echo "Step 6: Batch Fitting"
echo "--------------------------------------------------------------"
echo "Command: nlsq batch workflows/*.yaml --summary output/batch_summary.json"
echo
nlsq batch workflows/*.yaml --summary output/batch_summary.json
echo
echo "Summary saved to: output/batch_summary.json"
echo

# ==============================================================================
# Step 7: Display batch summary
# ==============================================================================
echo "Step 7: Batch Summary"
echo "--------------------------------------------------------------"
if command -v jq &> /dev/null; then
    echo "Using jq to format output:"
    cat output/batch_summary.json | jq '{total, succeeded, failed, duration_seconds}'
else
    echo "Install jq for formatted output. Raw JSON:"
    cat output/batch_summary.json
fi
echo

# ==============================================================================
# Step 8: Extract specific results with jq (if available)
# ==============================================================================
if command -v jq &> /dev/null; then
    echo "Step 8: Extract Results with jq"
    echo "--------------------------------------------------------------"
    echo "Radioactive decay fitted parameters:"
    cat output/radioactive_decay_results.json | jq '{popt, uncertainties, r_squared}'
    echo

    echo "Reaction kinetics fitted parameters:"
    cat output/reaction_kinetics_results.json | jq '{popt, uncertainties, r_squared}'
    echo
fi

# ==============================================================================
# Summary
# ==============================================================================
echo "=============================================================="
echo "CLI Examples Complete!"
echo "=============================================================="
echo
echo "Generated files:"
echo "  - data/radioactive_decay.csv"
echo "  - data/enzyme_kinetics.csv"
echo "  - data/reaction_kinetics.csv"
echo "  - data/damped_oscillation.csv"
echo
echo "Result files:"
echo "  - output/radioactive_decay_results.json"
echo "  - output/enzyme_kinetics_results.json"
echo "  - output/reaction_kinetics_results.json"
echo "  - output/damped_oscillator_results.json"
echo "  - output/batch_summary.json"
echo
echo "For more options, run: nlsq --help"
echo "=============================================================="

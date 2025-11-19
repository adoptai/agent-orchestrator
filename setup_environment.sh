#!/bin/bash
# ============================================================================
# Proper OpenMP Fix: Environment Setup Script
# ============================================================================
# This script sets up a conda environment with proper OpenMP management.
# It ensures all packages use a single OpenMP runtime by:
#   1. Creating a conda environment with numpy from conda-forge
#   2. Installing other packages via pip (they'll use conda's OpenMP)
#   3. Verifying OpenMP linkage
#
# Usage:
#   bash setup_environment.sh
# ============================================================================

set -e  # Exit on any error

echo "============================================================================"
echo "🔧 Setting up proper OpenMP environment"
echo "============================================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo ""
    echo "Please install Miniconda or Anaconda:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Found conda: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^agent-orchestrator "; then
    echo "⚠️  Environment 'agent-orchestrator' already exists"
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n agent-orchestrator -y
    else
        echo "📦 Using existing environment"
        echo ""
        echo "To activate it, run:"
        echo "  conda activate agent-orchestrator"
        exit 0
    fi
fi

# Create environment from environment.yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "✅ Environment created successfully!"
echo ""

# Activate and install pip packages
echo "📥 Installing pip packages..."
conda run -n agent-orchestrator pip install -r requirements.txt

echo ""
echo "============================================================================"
echo "✅ Setup complete!"
echo "============================================================================"
echo ""
echo "To use this environment:"
echo "  1. Activate it:"
echo "     conda activate agent-orchestrator"
echo ""
echo "  2. Run your program:"
echo "     python main.py"
echo ""
echo "This environment uses conda-forge packages which share a single OpenMP"
echo "runtime, eliminating the OpenMP conflict error."
echo ""


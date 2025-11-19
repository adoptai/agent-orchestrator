#!/bin/bash
# ============================================================================
# Alternative Fix: Fix OpenMP with pip (without conda)
# ============================================================================
# This script fixes OpenMP conflicts by:
#   1. Installing libomp via Homebrew (single OpenMP source)
#   2. Reinstalling packages to use the system OpenMP
#   3. Setting environment variables for proper linking
#
# This is an alternative if you prefer not to use conda.
# ============================================================================

set -e  # Exit on any error

echo "============================================================================"
echo "🔧 Fixing OpenMP conflicts with pip/Homebrew approach"
echo "============================================================================"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Error: Homebrew is not installed"
    echo ""
    echo "Please install Homebrew first:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✅ Found Homebrew: $(brew --version | head -n 1)"
echo ""

# Install libomp via Homebrew (single OpenMP source)
echo "📦 Installing libomp via Homebrew..."
if brew list libomp &> /dev/null; then
    echo "   libomp is already installed"
else
    brew install libomp
fi

echo ""
echo "📦 Reinstalling Python packages to use system OpenMP..."
echo ""

# Get the path to libomp
OMP_PATH=$(brew --prefix libomp)
export LDFLAGS="-L${OMP_PATH}/lib"
export CPPFLAGS="-I${OMP_PATH}/include"

echo "   Using OpenMP from: ${OMP_PATH}"
echo ""

# Uninstall packages that might have OpenMP conflicts
echo "🗑️  Uninstalling packages with potential OpenMP conflicts..."
pip uninstall -y numpy faiss-cpu chromadb sentence-transformers 2>/dev/null || true

# Reinstall numpy first (others depend on it)
echo "📥 Reinstalling numpy..."
pip install --no-cache-dir numpy

# Reinstall other packages
echo "📥 Reinstalling faiss-cpu..."
pip install --no-cache-dir faiss-cpu

echo "📥 Reinstalling chromadb..."
pip install --no-cache-dir chromadb

echo "📥 Reinstalling sentence-transformers..."
pip install --no-cache-dir sentence-transformers

# Install remaining requirements
echo "📥 Installing remaining requirements..."
pip install -r requirements.txt

echo ""
echo "============================================================================"
echo "✅ OpenMP fix complete!"
echo "============================================================================"
echo ""
echo "Packages have been reinstalled to use the system OpenMP library."
echo "You should no longer see OpenMP conflict errors."
echo ""
echo "Note: If you still see errors, you may need to:"
echo "  1. Use a virtual environment (recommended)"
echo "  2. Or use the conda approach (setup_environment.sh)"
echo ""


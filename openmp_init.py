import os
import sys

# ============================================================================
# OpenMP Environment Configuration
# ============================================================================
# These MUST be set before importing numpy, faiss, chromadb, or any other
# package that uses OpenMP. Once OpenMP is initialized, changing these has
# no effect.

def configure_openmp():
    """
    Configure OpenMP environment variables before any OpenMP-using libraries load.
    This prevents multiple initialization and controls thread usage.
    """
    
    # Set number of threads to 1 for all OpenMP libraries
    # This is the safest approach - prevents conflicts and reduces overhead
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Disable nested parallelism (use modern variable to avoid deprecation warning)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # Replaces deprecated OMP_NESTED
    
    # Only as a last resort, allow duplicate libraries
    # This is still not ideal but prevents crashes
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Disable Intel MKL threading (if present)
    os.environ["MKL_THREADING_LAYER"] = "sequential"
    
    print("🔧 OpenMP configuration applied:")
    print(f"   • OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    print(f"   • OMP_MAX_ACTIVE_LEVELS: {os.environ['OMP_MAX_ACTIVE_LEVELS']}")
    print(f"   • Duplicate lib handling: {os.environ['KMP_DUPLICATE_LIB_OK']}")
    print()

# Configure OpenMP immediately when this module is imported
configure_openmp()


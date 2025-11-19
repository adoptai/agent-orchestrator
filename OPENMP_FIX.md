# Proper OpenMP Fix Documentation

## Problem

On macOS, you may encounter this error:
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

This happens when multiple Python packages (numpy, faiss-cpu, chromadb, sentence-transformers) each statically link their own copy of the OpenMP library, causing conflicts.

## Root Cause

The error message suggests: *"The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library."*

This means we need to:
1. **Avoid static linking** - Don't let each package bundle its own OpenMP
2. **Use dynamic linking** - All packages should share one OpenMP library
3. **Ensure consistency** - All packages use the same OpenMP version

## Solutions

### Solution 1: Conda Environment (Recommended ✅)

**File**: `setup_environment.sh` and `environment.yml`

**How it works**:
- Creates a conda environment with numpy from conda-forge
- Conda-forge packages use consistent, dynamically-linked OpenMP
- Other packages installed via pip will use conda's OpenMP

**Why it's best**:
- ✅ Proper dependency management
- ✅ Ensures all packages share the same OpenMP
- ✅ No workarounds needed
- ✅ Industry best practice

**Usage**:
```bash
bash setup_environment.sh
conda activate agent-orchestrator
python main.py
```

### Solution 2: Homebrew + Pip (Alternative)

**File**: `fix_openmp_pip.sh`

**How it works**:
- Installs libomp via Homebrew (single source)
- Reinstalls packages with proper linking flags
- Packages use the system OpenMP library

**Why it's useful**:
- ✅ Works if you prefer pip over conda
- ✅ Uses system package manager
- ✅ Still ensures single OpenMP source

**Usage**:
```bash
bash fix_openmp_pip.sh
python main.py
```

## What NOT to Do

❌ **Don't use `KMP_DUPLICATE_LIB_OK=TRUE`** - This is a workaround that:
- Masks the problem instead of fixing it
- Can cause performance degradation
- May lead to incorrect results
- Is not a proper solution

## Technical Details

### Why Conda Works Better

Conda-forge packages are built with:
- Consistent build environments
- Shared system libraries (like OpenMP)
- Proper dependency resolution
- Dynamic linking instead of static bundling

### Why Pip Has Issues

Pip packages often:
- Bundle dependencies statically
- Each package includes its own OpenMP
- No coordination between packages
- Leads to multiple OpenMP copies

## Verification

After applying the fix, you should:
1. ✅ No longer see OpenMP errors
2. ✅ Program runs without warnings
3. ✅ No need for environment variable workarounds

## Files Created

- `environment.yml` - Conda environment definition
- `setup_environment.sh` - Automated conda setup script
- `fix_openmp_pip.sh` - Alternative pip-based fix
- `OPENMP_FIX.md` - This documentation

## References

- [OpenMP Official Documentation](https://www.openmp.org/)
- [Conda-forge Best Practices](https://conda-forge.org/docs/maintainer/adding_pkgs.html)
- [NumPy Build Configuration](https://numpy.org/devdocs/user/building.html)


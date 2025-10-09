# NLSQ Dependency Management

This document explains NLSQ's dependency management strategy and the different requirements files.

## Files Overview

### pyproject.toml (Source of Truth)
- **Purpose**: Declares minimum tested versions for library flexibility
- **Philosophy**: Allows users to use newer compatible versions
- **Updates**: Reflects tested minimum versions, not exact pins

### requirements.txt (Runtime Lock)
- **Purpose**: Exact runtime dependencies for reproducible installations
- **Use Case**: Production deployments requiring specific versions
- **Install**: `pip install -r requirements.txt`

### requirements-dev.txt (Development Lock)
- **Purpose**: Complete development environment with exact versions
- **Use Case**: Developer onboarding, ensuring consistent dev environment
- **Install**: `pip install -r requirements-dev.txt`

### requirements-full.txt (Complete Snapshot)
- **Purpose**: Complete pip freeze of entire environment
- **Use Case**: Exact environment replication, debugging
- **Install**: `pip install -r requirements-full.txt`

## Dependency Strategy

### Library vs Application Approach

NLSQ is a **library**, not an application. Therefore:

1. **pyproject.toml uses minimum versions** (e.g., `numpy>=2.0.0`)
   - Gives users flexibility to use their existing environments
   - Follows Python packaging best practices (PEP 621)
   - Allows downstream packages to resolve dependencies

2. **Requirements files use exact versions** (e.g., `numpy==2.3.3`)
   - Provides reproducibility when needed
   - Documents tested configurations
   - Enables CI/CD consistency

## Tested Versions (2025-10-08)

### Core Dependencies
- **Python**: 3.12.3
- **NumPy**: 2.3.3 (⚠️ Requires NumPy 2.x - see migration notes)
- **SciPy**: 1.16.2
- **JAX**: 0.7.2
- **JAXlib**: 0.7.2
- **Matplotlib**: 3.10.7

### Development Tools
- **pytest**: 8.4.2
- **black**: 25.9.0 (CalVer)
- **ruff**: 0.14.0
- **mypy**: 1.18.2
- **pre-commit**: 4.3.0

### Build System
- **setuptools**: 80.9.0
- **setuptools-scm**: 9.2.0

## Important Notes

### ⚠️ NumPy 2.0+ Required

NLSQ v0.1.1+ requires NumPy 2.0 or higher due to:
- Updated C API usage
- Removal of deprecated functions
- Performance improvements

**Migration from NumPy 1.x:**
```bash
# Upgrade NumPy
pip install --upgrade "numpy>=2.0"

# Reinstall NLSQ
pip install --upgrade nlsq
```

### JAX Version Updates

NLSQ is tested on JAX 0.7.2, which includes breaking changes from 0.4.x:
- Improved JIT compilation
- Updated API for transformations
- Better GPU/TPU support

Minimum supported version: JAX >= 0.6.0

### Optional Dependencies

Install with optional features:
```bash
# Development tools
pip install nlsq[dev]

# Documentation building
pip install nlsq[docs]

# Testing frameworks
pip install nlsq[test]

# Benchmarking tools
pip install nlsq[benchmark]

# Jupyter notebook support
pip install nlsq[jupyter]

# Everything
pip install nlsq[all]
```

## CI/CD Usage

### For Testing (Flexible)
```yaml
# Use pyproject.toml dependencies (allows newer versions)
- pip install -e ".[test]"
```

### For Production (Locked)
```yaml
# Use exact versions from requirements.txt
- pip install -r requirements.txt
```

### For Development (Complete)
```yaml
# Use complete dev environment
- pip install -r requirements-dev.txt
```

## Updating Dependencies

### When to Update pyproject.toml
- After validating new minimum versions work
- When fixing security vulnerabilities
- When leveraging new features from dependencies
- After running full test suite (817/820 tests)

### How to Update
1. Update local environment: `pip install --upgrade <package>`
2. Run tests: `make test` or `pytest`
3. Update pyproject.toml minimum version
4. Regenerate requirements files:
   ```bash
   pip freeze > requirements-full.txt
   # Manually update requirements.txt and requirements-dev.txt
   ```
5. Update this REQUIREMENTS.md with new tested versions
6. Commit changes

## Version History

### 2025-10-08 (v0.1.1)
- ✅ Updated to NumPy 2.3.3 (NumPy 2.x required)
- ✅ Updated to JAX 0.7.2
- ✅ Updated all development tools (Black 25.x, Ruff 0.14.0, etc.)
- ✅ Added Jupyter support as optional dependency
- ✅ Created requirements lock files for reproducibility

### Previous Versions
- NumPy >=1.26.0, JAX >=0.4.20 (pre-2025)

## Troubleshooting

### "NumPy version incompatible" Error
**Solution**: Upgrade to NumPy 2.x
```bash
pip install --upgrade "numpy>=2.0"
```

### "JAX version too old" Error
**Solution**: Upgrade JAX and JAXlib together
```bash
pip install --upgrade "jax>=0.6.0" "jaxlib>=0.6.0"
```

### Dependency Conflicts
**Solution**: Use requirements files for exact versions
```bash
pip install -r requirements-dev.txt
```

## References

- [Python Packaging User Guide](https://packaging.python.org/)
- [PEP 621 - pyproject.toml](https://peps.python.org/pep-0621/)
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [JAX Documentation](https://jax.readthedocs.io/)

---

**Last Updated**: 2025-10-08
**Maintainer**: Wei Chen (wchen@anl.gov)
**Version**: v0.1.1

# Test Commands Guide

## ⚠️ Important: Run ALL Tests for Coverage

The project requires **80% test coverage** as configured in `pyproject.toml`.

### ✅ Correct Commands (Run ALL Tests)

```bash
# Run all tests with coverage report
make test           # Runs: pytest
make test-cov       # Runs: pytest --cov-report=html

# Or directly:
pytest              # Runs all 116 tests
```

### ❌ Commands that Skip Tests

These commands will **deselect tests** and cause coverage to drop below 80%:

```bash
make test-slow      # Only runs 49 tests marked as "slow" (deselects 67 tests)
make test-debug     # Only runs slow tests with debug logging (deselects 67 tests)
make test-fast      # Excludes slow tests
```

### Coverage Results

- **With all tests**: 116 passed, **81.83% coverage** ✅
- **With only slow tests**: 49 passed, 67 deselected, **~72% coverage** ❌

### Makefile Targets Reference

| Command | Description | Tests Run | Coverage |
|---------|------------|-----------|----------|
| `make test` | Run all tests | 116 | 81.83% ✅ |
| `make test-cov` | Run all tests with HTML report | 116 | 81.83% ✅ |
| `make test-slow` | Only slow tests | 49 | ~72% ❌ |
| `make test-debug` | Only slow tests with debug | 49 | ~72% ❌ |
| `make test-fast` | Exclude slow tests | 67 | <80% ❌ |
.PHONY: install dev test lint format type-check clean docs help

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install the package in editable mode
	pip install -e .

dev:  ## Install all development dependencies
	pip install --upgrade pip
	pip install -e ".[dev,test,docs]"
	pre-commit install

dev-all:  ## Install ALL dependencies (dev, test, docs, benchmark)
	pip install --upgrade pip
	pip install -e ".[all]"
	pre-commit install

test:  ## Run all tests with pytest
	pytest

test-fast:  ## Run only fast tests (excludes optimization tests)
	pytest -m "not slow"

test-slow:  ## Run only slow optimization tests
	pytest -m "slow"

test-debug:  ## Run slow tests with debug logging
	NLSQ_DEBUG=1 pytest -m "slow" -s

test-cpu:  ## Run tests with CPU backend (avoids GPU compilation issues)
	NLSQ_FORCE_CPU=1 pytest

test-cpu-debug:  ## Run slow tests with debug logging on CPU backend
	NLSQ_DEBUG=1 NLSQ_FORCE_CPU=1 pytest -m "slow" -s

test-cov:  ## Run tests with coverage report
	pytest --cov-report=html
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html 2>/dev/null || open htmlcov/index.html 2>/dev/null || echo "Please open htmlcov/index.html manually"

test-cov-fast:  ## Run fast tests with coverage report
	pytest -m "not slow" --cov-report=html
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html 2>/dev/null || open htmlcov/index.html 2>/dev/null || echo "Please open htmlcov/index.html manually"

lint:  ## Run linting checks
	ruff check .

format:  ## Format code with ruff
	ruff format .
	ruff check --fix .

type-check:  ## Run type checking with mypy
	mypy nlsq

clean:  ## Clean build artifacts and cache files
	rm -rf build dist *.egg-info .coverage htmlcov
	find . -type d -name .nlsq_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .benchmarks -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .nlsq_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .coverage -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .hypothesis -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name checkpoints -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name "nlsq_debug_*.log" -delete
	find . -type f -name "build*.log" -delete
	rm -rf coverage.xml coverage.json checkpoint_iter_100.npz

clean-cache:  ## Clean only cache directories
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache .nlsq_cache

debug-modules:  ## Debug new modules with verbose logging
	NLSQ_DEBUG=1 python -c "from nlsq import stability, recovery, memory_manager, smart_cache; print('All new modules imported successfully')"

validate-install:  ## Validate that all modules can be imported
	python -c "import nlsq; print(f'NLSQ version: {nlsq.__version__}'); print('Available modules:', [m for m in dir(nlsq) if not m.startswith('_')])"

docs:  ## Build documentation
	cd docs && make clean html
	@echo "Opening documentation..."
	@python -m webbrowser docs/_build/html/index.html 2>/dev/null || open docs/_build/html/index.html 2>/dev/null || echo "Please open docs/_build/html/index.html manually"

benchmark:  ## Run performance benchmarks
	python benchmark/benchmark.py

benchmark-large:  ## Run large dataset benchmarks
	python benchmark/benchmark.py --large-datasets

profile:  ## Run memory profiling tests
	python -m memory_profiler benchmark/benchmark.py

build:  ## Build distribution packages
	python -m pip install --upgrade build
	python -m build

validate:  ## Validate package build
	python -m pip install --upgrade twine
	twine check dist/*

install-local:  ## Install from local build
	pip install dist/*.whl --force-reinstall

publish-test:  ## Publish to TestPyPI
	python -m pip install --upgrade twine
	twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	python -m pip install --upgrade twine
	twine upload dist/*

examples:  ## Test example notebooks
	python -c "import nbformat; from nbconvert.preprocessors import ExecutePreprocessor; [ExecutePreprocessor(timeout=600).preprocess(nbformat.read(f, as_version=4), {}) for f in ['examples/NLSQ Quickstart.ipynb', 'examples/NLSQ 2D Gaussian Demo.ipynb', 'examples/large_dataset_demo.ipynb']]"

test-large:  ## Run tests for large dataset features
	pytest tests/test_large_dataset.py tests/test_sparse_jacobian.py tests/test_streaming_optimizer.py -v

test-all:  ## Run all tests including large dataset tests
	pytest tests/ -v

test-modules:  ## Test specific new modules (stability, recovery, cache, etc)
	pytest tests/test_stability.py tests/test_stability_extended.py tests/test_init_module.py -v

test-memory:  ## Test memory management and leak detection
	pytest tests/ -k "memory" -v

test-cache:  ## Test caching functionality
	pytest tests/ -k "cache" -v

test-diagnostics:  ## Test diagnostics and monitoring
	pytest tests/ -k "diagnostic" -v

test-recovery:  ## Test optimization recovery mechanisms
	pytest tests/ -k "recovery" -v

test-validation:  ## Test input validation
	pytest tests/ -k "validat" -v

test-comprehensive:  ## Run comprehensive test suite with all new modules
	pytest tests/test_comprehensive_coverage.py tests/test_target_coverage.py tests/test_final_coverage.py -v

security-check:  ## Run security analysis with bandit
	bandit -r nlsq/ -ll --skip B101,B601,B602,B607

pre-commit-all:  ## Run all pre-commit hooks on all files
	pre-commit run --all-files

memory-profile:  ## Profile memory usage during tests
	python -m memory_profiler -m pytest tests/test_stability.py -v

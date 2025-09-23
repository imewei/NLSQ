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
	rm -rf build dist *.egg-info .coverage htmlcov .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name "nlsq_debug_*.log" -delete
	rm -rf coverage.xml .benchmarks

docs:  ## Build documentation
	cd docs && make clean html
	@echo "Opening documentation..."
	@python -m webbrowser docs/_build/html/index.html 2>/dev/null || open docs/_build/html/index.html 2>/dev/null || echo "Please open docs/_build/html/index.html manually"

benchmark:  ## Run performance benchmarks
	pytest benchmark/ --benchmark-only

profile:  ## Run memory profiling tests
	python -m memory_profiler benchmark/speed_comparison.py

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
	python -c "import nbformat; from nbconvert.preprocessors import ExecutePreprocessor; [ExecutePreprocessor(timeout=600).preprocess(nbformat.read(f, as_version=4), {}) for f in ['examples/NLSQ Quickstart.ipynb', 'examples/NLSQ 2D Gaussian Demo.ipynb']]"
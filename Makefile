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

test:  ## Run tests with pytest
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov-report=html
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
	rm -rf coverage.xml

docs:  ## Build documentation
	cd docs && make clean html
	@echo "Opening documentation..."
	@python -m webbrowser docs/_build/html/index.html 2>/dev/null || open docs/_build/html/index.html 2>/dev/null || echo "Please open docs/_build/html/index.html manually"

build:  ## Build distribution packages
	python -m pip install --upgrade build
	python -m build

publish-test:  ## Publish to TestPyPI
	python -m pip install --upgrade twine
	twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	python -m pip install --upgrade twine
	twine upload dist/*
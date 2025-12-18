# NLSQ Package Makefile
# ======================
# GPU-Accelerated Nonlinear Least Squares Curve Fitting
# Built on JAX for datasets up to 100M+ points

.PHONY: help install install-dev install-jax-gpu gpu-check env-info \
        test test-fast test-slow test-parallel test-all-parallel test-coverage \
        test-cpu test-debug test-large test-modules test-comprehensive \
        clean clean-all clean-pyc clean-build clean-test clean-venv \
        format lint type-check check quick docs build publish info version \
        benchmark benchmark-large security-check pre-commit-all validate-install

# ===================
# Configuration
# ===================
PYTHON := python
PYTEST := pytest
PACKAGE_NAME := nlsq
SRC_DIR := nlsq
TEST_DIR := tests
DOCS_DIR := docs
VENV := .venv

# ===================
# Platform detection
# ===================
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := windows
endif

# ===================
# Package manager detection (prioritize uv > conda/mamba > pip)
# ===================
UV_AVAILABLE := $(shell command -v uv 2>/dev/null)
CONDA_PREFIX := $(shell echo $$CONDA_PREFIX)
MAMBA_AVAILABLE := $(shell command -v mamba 2>/dev/null)

ifdef UV_AVAILABLE
    PKG_MANAGER := uv
    PIP := uv pip
    UNINSTALL_CMD := uv pip uninstall -y
    INSTALL_CMD := uv pip install
    RUN_CMD := uv run
else ifdef CONDA_PREFIX
    ifdef MAMBA_AVAILABLE
        PKG_MANAGER := mamba (using pip for JAX)
    else
        PKG_MANAGER := conda (using pip for JAX)
    endif
    PIP := pip
    UNINSTALL_CMD := pip uninstall -y
    INSTALL_CMD := pip install
    RUN_CMD :=
else
    PKG_MANAGER := pip
    PIP := pip
    UNINSTALL_CMD := pip uninstall -y
    INSTALL_CMD := pip install
    RUN_CMD :=
endif

# GPU installation command (platform-specific)
ifeq ($(PLATFORM),linux)
    JAX_GPU_PKG := jax[cuda12-local]==0.8.0 jaxlib==0.8.0
else
    JAX_GPU_PKG :=
endif

# ===================
# Colors for output
# ===================
BOLD := \033[1m
RESET := \033[0m
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
CYAN := \033[36m

# Default target
.DEFAULT_GOAL := help

# ===================
# Help target
# ===================
help:
	@echo "$(BOLD)$(BLUE)NLSQ Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET) make $(CYAN)<target>$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)ENVIRONMENT$(RESET)"
	@echo "  $(CYAN)env-info$(RESET)         Show detailed environment information"
	@echo "  $(CYAN)info$(RESET)             Show project and environment info"
	@echo "  $(CYAN)version$(RESET)          Show package version"
	@echo "  $(CYAN)validate-install$(RESET) Validate that all modules can be imported"
	@echo ""
	@echo "$(BOLD)$(GREEN)INSTALLATION$(RESET)"
	@echo "  $(CYAN)install$(RESET)          Install package in editable mode (CPU-only JAX)"
	@echo "  $(CYAN)install-dev$(RESET)      Install with development dependencies"
	@echo "  $(CYAN)install-jax-gpu$(RESET)  Install JAX with GPU support (Linux + CUDA 12+ only)"
	@echo ""
	@echo "$(BOLD)$(GREEN)GPU COMMANDS$(RESET)"
	@echo "  $(CYAN)gpu-check$(RESET)        Check GPU availability and CUDA setup"
	@echo ""
	@echo "$(BOLD)$(GREEN)TESTING$(RESET)"
	@echo "  $(CYAN)test$(RESET)             Run all tests"
	@echo "  $(CYAN)test-fast$(RESET)        Run tests excluding slow tests (-m 'not slow')"
	@echo "  $(CYAN)test-slow$(RESET)        Run only slow optimization tests"
	@echo "  $(CYAN)test-parallel$(RESET)    Run tests in parallel (-n auto)"
	@echo "  $(CYAN)test-all-parallel$(RESET) Run all tests in parallel"
	@echo "  $(CYAN)test-coverage$(RESET)    Run tests with coverage report"
	@echo "  $(CYAN)test-cpu$(RESET)         Run tests with CPU backend"
	@echo "  $(CYAN)test-debug$(RESET)       Run slow tests with debug logging"
	@echo "  $(CYAN)test-large$(RESET)       Run large dataset feature tests"
	@echo "  $(CYAN)test-modules$(RESET)     Run tests for specific modules"
	@echo "  $(CYAN)test-comprehensive$(RESET) Run comprehensive test suite"
	@echo ""
	@echo "$(BOLD)$(GREEN)CODE QUALITY$(RESET)"
	@echo "  $(CYAN)format$(RESET)           Format code with ruff"
	@echo "  $(CYAN)lint$(RESET)             Run linting checks (ruff)"
	@echo "  $(CYAN)type-check$(RESET)       Run type checking (mypy)"
	@echo "  $(CYAN)check$(RESET)            Run all checks (lint + type-check)"
	@echo "  $(CYAN)quick$(RESET)            Fast iteration: format + fast tests"
	@echo "  $(CYAN)security-check$(RESET)   Run security analysis with bandit"
	@echo "  $(CYAN)pre-commit-all$(RESET)   Run all pre-commit hooks"
	@echo ""
	@echo "$(BOLD)$(GREEN)DOCUMENTATION$(RESET)"
	@echo "  $(CYAN)docs$(RESET)             Build documentation with Sphinx"
	@echo ""
	@echo "$(BOLD)$(GREEN)BUILD & PUBLISH$(RESET)"
	@echo "  $(CYAN)build$(RESET)            Build distribution packages"
	@echo "  $(CYAN)publish$(RESET)          Publish to PyPI (requires credentials)"
	@echo ""
	@echo "$(BOLD)$(GREEN)BENCHMARKS$(RESET)"
	@echo "  $(CYAN)benchmark$(RESET)        Run performance benchmarks"
	@echo "  $(CYAN)benchmark-large$(RESET)  Run large dataset benchmarks"
	@echo ""
	@echo "$(BOLD)$(GREEN)CLEANUP$(RESET)"
	@echo "  $(CYAN)clean$(RESET)            Remove build artifacts and caches"
	@echo "  $(CYAN)clean-all$(RESET)        Deep clean of all caches"
	@echo "  $(CYAN)clean-pyc$(RESET)        Remove Python file artifacts"
	@echo "  $(CYAN)clean-build$(RESET)      Remove build artifacts"
	@echo "  $(CYAN)clean-test$(RESET)       Remove test and coverage artifacts"
	@echo "  $(CYAN)clean-venv$(RESET)       Remove virtual environment (use with caution)"
	@echo ""
	@echo "$(BOLD)Environment Detection:$(RESET)"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Package manager: $(PKG_MANAGER)"
	@echo ""

# ===================
# Installation targets
# ===================
install:
	@echo "$(BOLD)$(BLUE)Installing $(PACKAGE_NAME) in editable mode...$(RESET)"
	@$(INSTALL_CMD) -e .
	@echo "$(BOLD)$(GREEN)✓ Package installed!$(RESET)"

install-dev: install
	@echo "$(BOLD)$(BLUE)Installing development dependencies...$(RESET)"
	@$(INSTALL_CMD) -e ".[dev,test,docs]"
	@pre-commit install 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)✓ Dev dependencies installed!$(RESET)"

install-all: install
	@echo "$(BOLD)$(BLUE)Installing ALL dependencies (dev, test, docs, benchmark)...$(RESET)"
	@$(INSTALL_CMD) -e ".[all]"
	@pre-commit install 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)✓ All dependencies installed!$(RESET)"

# ===================
# GPU installation
# ===================
install-jax-gpu:
	@echo "$(BOLD)$(BLUE)Installing JAX with GPU support...$(RESET)"
	@echo "===================================="
	@echo "$(BOLD)Platform:$(RESET) $(PLATFORM)"
	@echo "$(BOLD)Package manager:$(RESET) $(PKG_MANAGER)"
	@echo ""
ifeq ($(PLATFORM),linux)
	@echo "$(BOLD)Step 1/4:$(RESET) Uninstalling CPU-only JAX..."
	@$(UNINSTALL_CMD) jax jaxlib 2>/dev/null || true
	@echo "  ✓ CPU JAX uninstalled"
	@echo ""
	@echo "$(BOLD)Step 2/4:$(RESET) Installing GPU-enabled JAX (CUDA 12.1-12.9)..."
	@echo "  Command: $(INSTALL_CMD) $(JAX_GPU_PKG)"
	@$(INSTALL_CMD) $(JAX_GPU_PKG)
	@echo "  ✓ GPU JAX installed"
	@echo ""
	@echo "$(BOLD)Step 3/4:$(RESET) Verifying GPU detection..."
	@$(MAKE) gpu-check
	@echo ""
	@echo "$(BOLD)Step 4/4:$(RESET) Installation complete!"
	@echo "$(BOLD)$(GREEN)✓ JAX GPU support installed successfully$(RESET)"
	@echo ""
	@echo "$(BOLD)Summary:$(RESET)"
	@echo "  Package manager: $(PKG_MANAGER)"
	@echo "  JAX version: 0.8.0 with CUDA 12 support"
	@echo "  Performance Impact: 150-270x speedup for large datasets (1M+ points)"
else
	@echo "$(BOLD)$(RED)✗ GPU acceleration only available on Linux with CUDA 12+$(RESET)"
	@echo "  Current platform: $(PLATFORM)"
	@echo "  Keeping CPU-only installation"
	@echo ""
	@echo "$(BOLD)Platform support:$(RESET)"
	@echo "  ✅ Linux + CUDA 12.1-12.9: Full GPU acceleration"
	@echo "  ❌ macOS: CPU-only (no NVIDIA GPU support)"
	@echo "  ❌ Windows: CPU-only (use WSL2 for GPU support)"
endif

# ===================
# GPU verification
# ===================
gpu-check:
	@echo "$(BOLD)$(BLUE)Checking GPU Configuration...$(RESET)"
	@echo "============================"
	@$(PYTHON) -c "import jax; print(f'JAX version: {jax.__version__}'); devices = jax.devices(); print(f'Available devices: {devices}'); gpu_devices = [d for d in devices if 'cuda' in str(d).lower() or 'gpu' in str(d).lower()]; print(f'✓ GPU detected: {len(gpu_devices)} device(s)') if gpu_devices else print('✗ No GPU detected - using CPU')"

# ===================
# Environment info
# ===================
env-info:
	@echo "$(BOLD)$(BLUE)Environment Information$(RESET)"
	@echo "======================"
	@echo ""
	@echo "$(BOLD)Platform Detection:$(RESET)"
	@echo "  OS: $(UNAME_S)"
	@echo "  Platform: $(PLATFORM)"
	@echo ""
	@echo "$(BOLD)Python Environment:$(RESET)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1 || echo 'not found')"
	@echo "  Python path: $(shell which $(PYTHON) 2>/dev/null || echo 'not found')"
	@echo ""
	@echo "$(BOLD)Package Manager Detection:$(RESET)"
	@echo "  Active manager: $(PKG_MANAGER)"
ifdef UV_AVAILABLE
	@echo "  ✓ uv detected: $(UV_AVAILABLE)"
	@echo "    Install command: $(INSTALL_CMD)"
	@echo "    Uninstall command: $(UNINSTALL_CMD)"
else
	@echo "  ✗ uv not found"
endif
ifdef CONDA_PREFIX
	@echo "  ✓ Conda environment detected"
	@echo "    CONDA_PREFIX: $(CONDA_PREFIX)"
ifdef MAMBA_AVAILABLE
	@echo "    Mamba available: $(MAMBA_AVAILABLE)"
else
	@echo "    Mamba: not found"
endif
	@echo "    Note: Using pip within conda for JAX installation"
else
	@echo "  ✗ Not in conda environment"
endif
	@echo "  pip: $(shell which pip 2>/dev/null || echo 'not found')"
	@echo ""
	@echo "$(BOLD)GPU Support:$(RESET)"
ifeq ($(PLATFORM),linux)
	@echo "  Platform: ✅ Linux (GPU support available)"
	@echo "  JAX GPU package: $(JAX_GPU_PKG)"
	@$(PYTHON) -c "import subprocess; r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, timeout=5); print(f'  GPU hardware: ✓ {r.stdout.strip()}') if r.returncode == 0 else print('  GPU hardware: ✗ Not detected')" 2>/dev/null || echo "  GPU hardware: ✗ nvidia-smi not found"
	@$(PYTHON) -c "import subprocess; r = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5); version = [line for line in r.stdout.split('\\n') if 'release' in line]; print(f'  CUDA: ✓ {version[0].split(\"release\")[1].split(\",\")[0].strip() if version else \"unknown\"}') if r.returncode == 0 else print('  CUDA: ✗ Not found')" 2>/dev/null || echo "  CUDA: ✗ nvcc not found"
else
	@echo "  Platform: ❌ $(PLATFORM) (GPU not supported)"
endif
	@echo ""
	@echo "$(BOLD)Installation Commands:$(RESET)"
	@echo "  Install GPU: make install-jax-gpu"
	@echo ""

# ===================
# Testing targets
# ===================
test:
	@echo "$(BOLD)$(BLUE)Running all tests...$(RESET)"
	$(RUN_CMD) $(PYTEST)

test-fast:
	@echo "$(BOLD)$(BLUE)Running fast tests (excluding slow)...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "not slow"
	@echo "$(BOLD)$(GREEN)✓ Fast tests passed!$(RESET)"

test-slow:
	@echo "$(BOLD)$(BLUE)Running slow optimization tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "slow"

test-parallel:
	@echo "$(BOLD)$(BLUE)Running tests in parallel...$(RESET)"
	$(RUN_CMD) $(PYTEST) -n auto

test-all-parallel:
	@echo "$(BOLD)$(BLUE)Running all tests in parallel...$(RESET)"
	$(RUN_CMD) $(PYTEST) -n auto
	@echo "$(BOLD)$(GREEN)✓ All tests passed!$(RESET)"

test-coverage:
	@echo "$(BOLD)$(BLUE)Running tests with coverage report...$(RESET)"
	$(RUN_CMD) $(PYTEST) --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(BOLD)$(GREEN)✓ Coverage report generated!$(RESET)"
	@echo "View HTML report: open htmlcov/index.html"

test-cpu:
	@echo "$(BOLD)$(BLUE)Running tests with CPU backend...$(RESET)"
	NLSQ_FORCE_CPU=1 $(RUN_CMD) $(PYTEST)

test-debug:
	@echo "$(BOLD)$(BLUE)Running slow tests with debug logging...$(RESET)"
	NLSQ_DEBUG=1 $(RUN_CMD) $(PYTEST) -m "slow" -s

test-large:
	@echo "$(BOLD)$(BLUE)Running large dataset feature tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) tests/test_large_dataset.py tests/test_sparse_jacobian.py tests/test_streaming_optimizer.py -v

test-modules:
	@echo "$(BOLD)$(BLUE)Running module-specific tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) tests/test_stability.py tests/test_stability_extended.py tests/test_init_module.py -v

test-comprehensive:
	@echo "$(BOLD)$(BLUE)Running comprehensive test suite...$(RESET)"
	$(RUN_CMD) $(PYTEST) tests/test_comprehensive_coverage.py tests/test_target_coverage.py tests/test_final_coverage.py -v

test-memory:
	@echo "$(BOLD)$(BLUE)Running memory tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -k "memory" -v

test-cache:
	@echo "$(BOLD)$(BLUE)Running cache tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -k "cache" -v

test-diagnostics:
	@echo "$(BOLD)$(BLUE)Running diagnostics tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -k "diagnostic" -v

test-recovery:
	@echo "$(BOLD)$(BLUE)Running recovery tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -k "recovery" -v

test-validation:
	@echo "$(BOLD)$(BLUE)Running validation tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -k "validat" -v

# ===================
# Code quality targets
# ===================
format:
	@echo "$(BOLD)$(BLUE)Formatting code with ruff...$(RESET)"
	$(RUN_CMD) ruff format $(PACKAGE_NAME) tests
	$(RUN_CMD) ruff check --fix $(PACKAGE_NAME) tests
	@echo "$(BOLD)$(GREEN)✓ Code formatted!$(RESET)"

lint:
	@echo "$(BOLD)$(BLUE)Running linting checks...$(RESET)"
	$(RUN_CMD) ruff check $(PACKAGE_NAME) tests
	@echo "$(BOLD)$(GREEN)✓ No linting errors!$(RESET)"

type-check:
	@echo "$(BOLD)$(BLUE)Running type checks...$(RESET)"
	$(RUN_CMD) mypy $(PACKAGE_NAME)
	@echo "$(BOLD)$(GREEN)✓ Type checking passed!$(RESET)"

check: lint type-check
	@echo "$(BOLD)$(GREEN)✓ All checks passed!$(RESET)"

quick: format test-fast
	@echo "$(BOLD)$(GREEN)✓ Quick iteration complete!$(RESET)"

security-check:
	@echo "$(BOLD)$(BLUE)Running security analysis...$(RESET)"
	$(RUN_CMD) bandit -r $(PACKAGE_NAME)/ -ll --skip B101,B601,B602,B607
	@echo "$(BOLD)$(GREEN)✓ Security check passed!$(RESET)"

pre-commit-all:
	@echo "$(BOLD)$(BLUE)Running all pre-commit hooks...$(RESET)"
	pre-commit run --all-files

# ===================
# Documentation targets
# ===================
docs:
	@echo "$(BOLD)$(BLUE)Building documentation...$(RESET)"
	cd $(DOCS_DIR) && $(MAKE) clean html
	@echo "$(BOLD)$(GREEN)✓ Documentation built!$(RESET)"
	@echo "Open: $(DOCS_DIR)/_build/html/index.html"

# ===================
# Build and publish targets
# ===================
build: clean-build
	@echo "$(BOLD)$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(BOLD)$(GREEN)✓ Build complete!$(RESET)"
	@echo "Distributions in dist/"

validate:
	@echo "$(BOLD)$(BLUE)Validating package build...$(RESET)"
	$(PYTHON) -m twine check dist/*
	@echo "$(BOLD)$(GREEN)✓ Package validation passed!$(RESET)"

publish: build validate
	@echo "$(BOLD)$(YELLOW)This will publish $(PACKAGE_NAME) to PyPI!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Publishing to PyPI...$(RESET)"; \
		$(PYTHON) -m twine upload dist/*; \
		echo "$(BOLD)$(GREEN)✓ Published to PyPI!$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

publish-test:
	@echo "$(BOLD)$(BLUE)Publishing to TestPyPI...$(RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*

# ===================
# Benchmark targets
# ===================
benchmark:
	@echo "$(BOLD)$(BLUE)Running performance benchmarks...$(RESET)"
	$(PYTHON) benchmark/benchmark.py

benchmark-large:
	@echo "$(BOLD)$(BLUE)Running large dataset benchmarks...$(RESET)"
	$(PYTHON) benchmark/benchmark.py --large-datasets

# ===================
# Cleanup targets
# ===================
clean-build:
	@echo "$(BOLD)$(BLUE)Removing build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .benchmarks/
	find . -type d -name "*.egg-info" \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true

clean-pyc:
	@echo "$(BOLD)$(BLUE)Removing Python file artifacts...$(RESET)"
	find . -type d -name __pycache__ \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*~" \) \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-delete 2>/dev/null || true

clean-test:
	@echo "$(BOLD)$(BLUE)Removing test and coverage artifacts...$(RESET)"
	find . -type d \( -name .pytest_cache -o -name .nlsq_cache -o -name .ruff_cache \
		-o -name .mypy_cache -o -name htmlcov -o -name .hypothesis -o -name .coverage \) \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d \( -name checkpoints -o -name checkpoints_example -o -name checkpoints_diagnostics \
		-o -name benchmark_results \) \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	rm -rf coverage.xml coverage.json
	find . -type f \( -name "nlsq_debug_*.log" -o -name "build*.log" -o -name "optimization.log" \) \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-delete 2>/dev/null || true

clean: clean-build clean-pyc clean-test
	@echo "$(BOLD)$(GREEN)✓ Cleaned!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/, .specify/, agent-os/"

clean-all: clean
	@echo "$(BOLD)$(BLUE)Performing deep clean of additional caches...$(RESET)"
	rm -rf .tox/ 2>/dev/null || true
	rm -rf .nox/ 2>/dev/null || true
	rm -rf .eggs/ 2>/dev/null || true
	rm -rf .cache/ 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)✓ Deep clean complete!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/, .specify/, agent-os/"

clean-venv:
	@echo "$(BOLD)$(YELLOW)WARNING: This will remove the virtual environment!$(RESET)"
	@echo "$(BOLD)You will need to recreate it manually.$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Removing virtual environment...$(RESET)"; \
		rm -rf $(VENV) venv; \
		echo "$(BOLD)$(GREEN)✓ Virtual environment removed!$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

# ===================
# Utility targets
# ===================
info:
	@echo "$(BOLD)$(BLUE)Project Information$(RESET)"
	@echo "===================="
	@echo "Project: $(PACKAGE_NAME)"
	@echo "Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "Platform: $(PLATFORM)"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo ""
	@echo "$(BOLD)$(BLUE)Directory Structure$(RESET)"
	@echo "===================="
	@echo "Source: $(SRC_DIR)/"
	@echo "Tests: $(TEST_DIR)/"
	@echo "Docs: $(DOCS_DIR)/"
	@echo ""
	@echo "$(BOLD)$(BLUE)JAX Configuration$(RESET)"
	@echo "=================="
	@$(PYTHON) -c "import jax; print('JAX version:', jax.__version__); print('Default backend:', jax.default_backend())" 2>/dev/null || echo "JAX not installed"

version:
	@$(PYTHON) -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__version__)" 2>/dev/null || \
		echo "$(BOLD)$(RED)Error: Package not installed. Run 'make install' first.$(RESET)"

validate-install:
	@echo "$(BOLD)$(BLUE)Validating installation...$(RESET)"
	@$(PYTHON) -c "import $(PACKAGE_NAME); print(f'NLSQ version: {$(PACKAGE_NAME).__version__}'); print('Available modules:', [m for m in dir($(PACKAGE_NAME)) if not m.startswith('_')])"
	@echo "$(BOLD)$(GREEN)✓ Installation validated!$(RESET)"

debug-modules:
	@echo "$(BOLD)$(BLUE)Debugging modules...$(RESET)"
	NLSQ_DEBUG=1 $(PYTHON) -c "from $(PACKAGE_NAME) import stability, recovery, memory_manager, smart_cache; print('All modules imported successfully')"

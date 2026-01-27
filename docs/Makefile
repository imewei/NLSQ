# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=

# Detect uv for virtual environment management
ifneq ($(shell command -v uv 2>/dev/null),)
    SPHINXBUILD   ?= uv run --extra docs sphinx-build
else
    SPHINXBUILD   ?= sphinx-build
endif
SOURCEDIR     = .
BUILDDIR      = _build

# Explicit Sphinx builder targets (replaces unsafe catch-all "%" rule
# which routes ANY unknown target to Sphinx and breaks when invoked
# with non-builder names like "clean-all").
.PHONY: help clean html dirhtml singlehtml latex latexpdf text man linkcheck doctest

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean up build caches and generated documentation
clean:
	@echo "Removing build directory..."
	@rm -rf $(BUILDDIR)
	@echo "Removing auto-generated API docs..."
	@rm -rf api/generated
	@echo "Removing Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"

# Sphinx builders — only known targets are routed to sphinx-build.
# $(O) is meant as a shortcut for $(SPHINXOPTS).
html dirhtml singlehtml latex latexpdf text man linkcheck doctest:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

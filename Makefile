# Developer convenience targets. The lint/format/test-cpu targets are the CPU
# quality gate and need no GPU; test-gpu and bench require a CUDA/ROCm device.

RUFF ?= ruff
PYTEST ?= pytest

.DEFAULT_GOAL := help
.PHONY: help install lint format format-check test test-cpu test-gpu typecheck check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with all dev extras
	pip install -e ".[all]"

lint:  ## Run ruff lint checks
	$(RUFF) check .

format:  ## Auto-format the codebase with ruff
	$(RUFF) format .

format-check:  ## Verify formatting without modifying files
	$(RUFF) format --check .

typecheck:  ## Run mypy (advisory)
	mypy

test-cpu:  ## Run the CPU-only test subset (GPU tests skip automatically)
	CUDA_VISIBLE_DEVICES="" $(PYTEST) -q

test-gpu:  ## Run the full suite including GPU kernel tests
	$(PYTEST) -q

test: test-cpu  ## Alias for test-cpu

check: lint format-check test-cpu  ## Full CPU quality gate (matches CI)

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

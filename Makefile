# edge-vision — common dev tasks

# Project venv interpreter; override with `make PY=python <target>` outside the venv.
PY ?= .venv/bin/python

.PHONY: help install install-dev install-gpu install-trt test test-fast smoke api ui export-onnx build-trt bench power-sweep distill clean format lint typecheck

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime deps only (CPU)
	$(PY) -m pip install -e .

install-dev: ## Install dev deps (pytest, ruff, mypy, jupyter, ui)
	$(PY) -m pip install -e ".[dev,ui]"

install-gpu: ## Install GPU deps (torch CUDA wheels assumed pre-installed)
	$(PY) -m pip install -e ".[dev,gpu]"

install-trt: ## Install TensorRT extras (Phase 2+)
	$(PY) -m pip install -e ".[trt]"
	@echo "Verify TensorRT: $(PY) -c 'import tensorrt; print(tensorrt.__version__)'"

test: ## Run pytest
	$(PY) -m pytest tests/ -v

test-fast: ## Skip slow tests
	$(PY) -m pytest tests/ -v -m "not slow"

smoke: ## End-to-end smoke (Phase 1+): RT-DETR baseline on 16 COCO images
	$(PY) scripts/run_baseline_smoke.py

export-onnx: ## Export PyTorch checkpoint -> ONNX (Phase 2+)
	$(PY) scripts/run_compile_smoke.py --stage onnx

build-trt: ## Build TensorRT engine from ONNX (Phase 2+)
	$(PY) scripts/run_compile_smoke.py --stage trt --precision fp16

bench: ## Latency p50/p95/p99 + FPS sweep (Phase 2+)
	$(PY) scripts/run_latency_sweep.py --num-images 100

power-sweep: ## Sustained 15-min NVML power + thermal sweep (Phase 5+)
	$(PY) scripts/run_power_sweep.py --duration-sec 900

distill: ## Run R50 -> R18 distillation smoke (Phase 4+)
	$(PY) scripts/run_distill_smoke.py

api: ## Run FastAPI dev server (Phase 6+)
	$(PY) -m uvicorn edgevision.api.main:app --reload

ui: ## Run Streamlit dashboard (Phase 5+)
	$(PY) -m streamlit run dashboard/pareto_plot.py

format: ## Auto-format with ruff
	$(PY) -m ruff format src tests scripts dashboard

lint: ## Lint with ruff
	$(PY) -m ruff check src tests scripts dashboard

typecheck: ## Type-check with mypy
	$(PY) -m mypy src

clean: ## Remove build/cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

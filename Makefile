.PHONY: install build-native serve benchmark test lint format docker engine clean

PYTHON ?= python
CONFIG ?= configs/server.yaml

install: ## Pure-python editable install
	$(PYTHON) -m pip install -e .

build-native: ## Build C++/CUDA scheduler extension inside the TRT container
	LLM_ENGINE_BUILD_CUDA=1 $(PYTHON) -m pip install -e . --no-build-isolation

engine: ## Compile a TensorRT-LLM engine (FP16/INT8) from a HF model
	$(PYTHON) scripts/build_engine.py --model $(MODEL) --dtype fp16 --tp 4

serve: ## Start the OpenAI-compatible serving endpoint
	$(PYTHON) -m engine.server --config $(CONFIG)

serve-multi: ## Multi-GPU launch (4 GPUs, NCCL tensor parallel)
	bash scripts/launch_distributed.sh 4 $(CONFIG)

benchmark: ## Latency / throughput benchmark against a running server
	$(PYTHON) benchmarks/benchmark_serving.py --host 127.0.0.1 --port 8000 --num-prompts 500

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check engine tests benchmarks
	mypy engine

format:
	ruff format engine tests benchmarks

docker:
	docker build -t llm-inference-engine:latest .

clean:
	rm -rf build dist *.egg-info csrc/build **/__pycache__ .pytest_cache .ruff_cache

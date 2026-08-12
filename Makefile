.PHONY: install install-train sandbox-images data train-reranker train-editor \
        serve eval table lint typecheck fmt clean

install:            ## base install: data + eval, no GPU needed
	python -m pip install -e ".[dev]"

install-train:      ## full install with CUDA deps (run on the GPU node)
	bash scripts/setup_env.sh

sandbox-images:     ## build the verification containers (the reward function)
	docker build -t autofix/sandbox-python:3.11 -f sandbox_images/python.Dockerfile sandbox_images
	docker build -t autofix/sandbox-node:20     -f sandbox_images/node.Dockerfile     sandbox_images
	docker build -t autofix/sandbox-go:1.22     -f sandbox_images/go.Dockerfile       sandbox_images

data:               ## build decontaminated training sets
	autofix-data --limit-per-source 20000

train-reranker:
	sbatch scripts/train_reranker.sbatch

train-editor:
	sbatch scripts/train_editor.sbatch

serve:
	bash scripts/serve_vllm.sh

eval:               ## baseline first, then the trained model
	autofix-eval --tag baseline --editor-model editor-base --reranker-model reranker-base --limit 50
	autofix-eval --tag sft

table:
	python -m autofix.eval.table

lint:
	ruff check src

fmt:
	ruff format src && ruff check --fix src

typecheck:
	mypy src

clean:
	rm -rf .ruff_cache .mypy_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

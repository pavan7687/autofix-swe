#!/bin/bash
# Serve the fine-tuned adapters for sampling and evaluation.
#
# Both adapters are served over their own base models. `--enable-lora` lets a
# request select an adapter by name via the `model` field, so one server backs
# both the reranker and the editor without a reload.
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-./artifacts/models}"
EDITOR_ADAPTER="${EDITOR_ADAPTER:-$MODEL_ROOT/editing-latest/adapter}"
EDITOR_BASE="${EDITOR_BASE:-Qwen/Qwen2.5-Coder-32B-Instruct}"

source .venv/bin/activate

# --max-lora-rank must be >= LORA_R from .env (default 32).
# --gpu-memory-utilization 0.90 leaves headroom for the sandbox containers that
# run on the same node during rejection sampling.
exec python -m vllm.entrypoints.openai.api_server \
  --model "$EDITOR_BASE" \
  --served-model-name editor-base \
  --enable-lora \
  --lora-modules editor="$EDITOR_ADAPTER" \
  --max-lora-rank 32 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --port 8000

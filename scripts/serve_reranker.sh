#!/bin/bash
# The reranker is small; serve it on a second port (or a second GPU) so the
# editor server does not have to be restarted when only the reranker changes.
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-./artifacts/models}"
RERANKER_ADAPTER="${RERANKER_ADAPTER:-$MODEL_ROOT/retrieval-latest/adapter}"
RERANKER_BASE="${RERANKER_BASE:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"

source .venv/bin/activate

exec python -m vllm.entrypoints.openai.api_server \
  --model "$RERANKER_BASE" \
  --served-model-name reranker-base \
  --enable-lora \
  --lora-modules reranker="$RERANKER_ADAPTER" \
  --max-lora-rank 32 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.25 \
  --dtype bfloat16 \
  --port 8001

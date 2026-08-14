#!/bin/bash
# Separate environment for vLLM inference. RUN ON THE LOGIN NODE.
#
#   bash scripts/setup_serving_env.sh
#
# Why a second environment: vLLM pins an exact torch version, and that pin
# routinely disagrees with the torch build the training stack needs (and with
# what this cluster's driver supports). Installing both together produces a
# resolver conflict and, worse, silently downgrades torch under the training
# code. Sampling and evaluation talk to vLLM over HTTP, so the two environments
# never need to share a process.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME=autofix-serve
PY_VERSION=3.11
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" \
  || conda create -y -q -n "$ENV_NAME" "python=$PY_VERSION" pip

PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
"$PY" -m pip install -q --upgrade pip

# Let vLLM choose its own torch, then verify it matches the driver. If vLLM
# demands a CUDA build newer than the driver supports, pin an older vLLM.
"$PY" -m pip install -q vllm
"$PY" -c "import torch, vllm; print(f'  vllm {vllm.__version__} on torch {torch.__version__} (CUDA {torch.version.cuda})')"

echo
echo "Driver supports CUDA 12.8 on this cluster. If the line above shows a"
echo "CUDA 13.x build, pin an older vLLM:"
echo "  $PY -m pip install 'vllm<0.27' --index-url $TORCH_CUDA_INDEX"
echo
echo "Serve with:  conda activate $ENV_NAME && bash scripts/serve_vllm.sh"

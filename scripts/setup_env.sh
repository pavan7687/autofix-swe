#!/bin/bash
# Build the Python environment. RUN THIS ON THE LOGIN NODE.
#
#   bash scripts/setup_env.sh
#
# Two deliberate choices, both learned the hard way on this cluster:
#
# 1. LOGIN NODE, not a compute node. Compute nodes here have no route to the
#    internet. Installing into shared home means every compute node sees the
#    finished environment without needing network itself. This is
#    download-and-unpack, not computation, so it is appropriate login node use.
#
# 2. PYTHON 3.11, not the conda base's 3.13. The ML stack lags new interpreter
#    releases: on 3.13 the cu121 wheel set is incomplete (nvidia-cudnn-cu12 has
#    no matching build) and vLLM/bitsandbytes support is patchy. 3.11 is the
#    version these libraries are actually tested against.
set -euo pipefail
cd "$(dirname "$0")/.."

PY_VERSION=3.11
ENV_NAME=autofix

echo "=== host: $(hostname) ==="

echo
echo "=== 1. Network reachability ==="
if timeout 20 python -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/', timeout=15)" 2>/dev/null; then
  echo "  PyPI reachable"
else
  echo "  ERROR: cannot reach PyPI from $(hostname)."
  echo "  Compute nodes have no internet here - run this on the login node."
  exit 1
fi

echo
echo "=== 2. Python $PY_VERSION environment ==="
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "  conda env '$ENV_NAME' exists"
  else
    echo "  creating conda env '$ENV_NAME' with python $PY_VERSION ..."
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
  fi
  conda activate "$ENV_NAME"
else
  echo "  conda not found; falling back to venv (interpreter will be $(python -V 2>&1))"
  [ -d .venv ] || python -m venv .venv
  source .venv/bin/activate
fi
python -m pip install -q --upgrade pip wheel setuptools
echo "  python $(python --version 2>&1 | cut -d' ' -f2) at $(which python)"

echo
echo "=== 3. PyTorch ==="
# Plain PyPI, NOT the cu121 index. Since torch 2.x the default Linux wheel is
# already CUDA-enabled and bundles its own nvidia-* dependencies, so it resolves
# cleanly. Pinning --index-url to a CUDA channel replaces PyPI entirely, and any
# dependency not mirrored there (nvidia-cudnn-cu12) then fails to resolve at all.
pip install torch
python -c "import torch; print(f'  torch {torch.__version__}')"
python -c "import torch; print(f'  bundled CUDA: {torch.version.cuda}')"

echo
echo "=== 4. Project and training dependencies ==="
pip install -e ".[train,serve,dev]"

echo
echo "=== 5. flash-attention (optional) ==="
# Needs nvcc at build time, which this cluster has no module for. PyTorch's
# built-in sdpa is slower but numerically identical; the training code reads
# AUTOFIX_ATTN to choose between them.
if command -v nvcc >/dev/null 2>&1 && pip install flash-attn --no-build-isolation 2>/dev/null; then
  ATTN=flash_attention_2
  echo "  flash-attn installed"
else
  ATTN=sdpa
  echo "  unavailable (no nvcc) - using sdpa"
fi

echo
echo "=== 6. Verify ==="
python - <<'PYEOF'
import importlib
for mod in ("torch", "transformers", "peft", "trl", "bitsandbytes",
            "accelerate", "datasets", "autofix"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<14} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"  {mod:<14} FAILED: {str(exc)[:70]}")
PYEOF

cat > .autofix-env <<EOF
# Written by setup_env.sh. Sourced automatically by scripts/activate_env.sh.
export AUTOFIX_ATTN=$ATTN
EOF
echo "  wrote .autofix-env (AUTOFIX_ATTN=$ATTN)"

echo
echo "=== Done ==="
echo "  source scripts/activate_env.sh"
echo "  autofix-data --limit-per-source 500"
echo
echo "GPU checks are skipped: login nodes have no GPU."
echo "Verify CUDA on a compute node with: sbatch scripts/smoke_test.sbatch"

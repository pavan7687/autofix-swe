#!/bin/bash
# Build the Python environment. RUN THIS ON THE LOGIN NODE.
#
#   bash scripts/setup_env.sh
#
# Why the login node and not a compute node: on most HPC clusters compute nodes
# have no route to the internet (or reach it only through a proxy), while login
# nodes do. Installing here writes into shared home storage, so every compute
# node sees the finished environment without needing network access itself.
#
# This is pure download-and-unpack, not computation, so it is appropriate login
# node usage. Nothing here needs a GPU: torch installs the CUDA runtime it needs
# as ordinary wheels, and only *uses* the driver at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== host: $(hostname) ==="
echo "If this is a compute node and installs fail, re-run on the login node."

# ---------------------------------------------------------------------------
echo
echo "=== 1. Network reachability ==="
if timeout 20 python -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/', timeout=15)" 2>/dev/null; then
  echo "  PyPI reachable"
else
  echo "  ERROR: cannot reach PyPI from $(hostname)."
  echo "  If you are on a compute node, run this on the login node instead."
  echo "  If the cluster uses a proxy, export http_proxy/https_proxy first."
  exit 1
fi

# ---------------------------------------------------------------------------
echo
echo "=== 2. Virtual environment ==="
[ -d .venv ] || python -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip wheel setuptools
echo "  python $(python --version 2>&1 | cut -d' ' -f2) at $(which python)"

# ---------------------------------------------------------------------------
echo
echo "=== 3. PyTorch ==="
# Torch first and on its own index, so pip pins the CUDA build. If torch is
# resolved later as a transitive dependency, pip may fetch a CPU-only wheel and
# everything afterwards silently compiles against the wrong runtime.
# cu121 works on the A40 (compute capability 8.6).
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('  torch', torch.__version__)"

# ---------------------------------------------------------------------------
echo
echo "=== 4. Project and training dependencies ==="
pip install -e ".[train,serve,dev]"

# ---------------------------------------------------------------------------
echo
echo "=== 5. flash-attention (optional) ==="
# Needs nvcc at build time. The cluster has no CUDA toolkit module, so this
# usually fails - which is fine. PyTorch's built-in `sdpa` is slower but
# numerically identical, and the training code reads AUTOFIX_ATTN to choose.
if command -v nvcc >/dev/null 2>&1; then
  pip install flash-attn --no-build-isolation \
    && ATTN=flash_attention_2 || ATTN=sdpa
else
  echo "  nvcc absent - skipping flash-attn"
  ATTN=sdpa
fi
echo "  attention backend: $ATTN"

# ---------------------------------------------------------------------------
echo
echo "=== 6. Verify ==="
for mod in transformers peft trl bitsandbytes accelerate datasets; do
  python -c "import $mod, sys; print(f'  $mod {getattr($mod, \"__version__\", \"ok\")}')" \
    || echo "  $mod FAILED"
done
python -c "import autofix; print('  autofix package importable')"

cat > .autofix-env <<EOF
# Sourced by the training scripts. Regenerate by re-running setup_env.sh.
export AUTOFIX_ATTN=$ATTN
EOF
echo "  wrote .autofix-env (AUTOFIX_ATTN=$ATTN)"

echo
echo "=== Done ==="
echo "  source .venv/bin/activate"
echo "  source .autofix-env"
echo "  autofix-data --limit-per-source 500"
echo
echo "NOTE: GPU checks are skipped here because login nodes have no GPU."
echo "      Verify CUDA on a compute node with: sbatch scripts/smoke_test.sbatch"

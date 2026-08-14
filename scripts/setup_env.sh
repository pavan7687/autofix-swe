#!/bin/bash
# Build the Python environment. RUN THIS ON THE LOGIN NODE.
#
#   bash scripts/setup_env.sh
#
# Three choices, each learned from a failure on this cluster:
#
# 1. LOGIN NODE. Compute nodes here have no route to the internet. Installing
#    into shared home means every compute node sees the finished environment.
#
# 2. PYTHON 3.11 via conda, not the base 3.13. The ML stack lags new interpreter
#    releases and 3.13 wheel coverage is still incomplete.
#
# 3. EXPLICIT INTERPRETER PATHS. `conda activate` inside a script does not
#    reliably win over an already-active virtualenv - PATH order decides, and a
#    stale `.venv` silently captured every install. Calling the environment's
#    python by absolute path removes the ambiguity entirely.
set -euo pipefail
cd "$(dirname "$0")/.."

PY_VERSION=3.11
ENV_NAME=autofix

echo "=== host: $(hostname) ==="

# Shed any environment already active in the calling shell, so nothing we do
# below depends on how the user happened to arrive here.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "  deactivating inherited virtualenv: $VIRTUAL_ENV"
  deactivate 2>/dev/null || true
  unset VIRTUAL_ENV
fi

echo
echo "=== 1. Network reachability ==="
if timeout 20 python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/', timeout=15)" 2>/dev/null; then
  echo "  PyPI reachable"
else
  echo "  ERROR: cannot reach PyPI from $(hostname)."
  echo "  Compute nodes have no internet here - run this on the login node."
  exit 1
fi

echo
echo "=== 2. Python $PY_VERSION environment ==="
command -v conda >/dev/null 2>&1 || { echo "  ERROR: conda not found"; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  conda env '$ENV_NAME' exists"
else
  conda create -y -q -n "$ENV_NAME" "python=$PY_VERSION" pip
fi

# Absolute paths from here on. This is the fix for the silent-capture bug.
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
[ -x "$PY" ] || { echo "  ERROR: $PY missing"; exit 1; }

ACTUAL="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "$ACTUAL" = "$PY_VERSION" ] || { echo "  ERROR: expected $PY_VERSION, got $ACTUAL"; exit 1; }
echo "  using $PY (python $ACTUAL)"

"$PY" -m pip install -q --upgrade pip wheel setuptools

echo
echo "=== 3. PyTorch ==="
# Plain PyPI: since torch 2.x the default Linux wheel is CUDA-enabled and
# bundles its own nvidia-* dependencies. Pinning --index-url to a CUDA channel
# replaces PyPI outright, and anything not mirrored there fails to resolve.
"$PY" -m pip install -q torch numpy
"$PY" -c "import torch; print(f'  torch {torch.__version__} (bundled CUDA {torch.version.cuda})')"

echo
echo "=== 4. Project and training dependencies ==="
"$PY" -m pip install -q -e ".[train,serve,dev]"

echo
echo "=== 5. flash-attention (optional, expected to fail here) ==="
if command -v nvcc >/dev/null 2>&1 \
   && "$PY" -m pip install -q flash-attn --no-build-isolation 2>/dev/null; then
  ATTN=flash_attention_2
  echo "  installed"
else
  ATTN=sdpa
  echo "  skipped (no nvcc) - using sdpa: slower, numerically identical"
fi

echo
echo "=== 6. Verify ==="
"$PY" - <<'PYEOF'
import importlib
ok = True
for mod in ("torch", "numpy", "transformers", "peft", "trl", "bitsandbytes",
            "accelerate", "datasets", "autofix"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<14} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        ok = False
        print(f"  {mod:<14} FAILED: {str(exc)[:70]}")
raise SystemExit(0 if ok else 1)
PYEOF

cat > .autofix-env <<EOF
# Written by setup_env.sh. Sourced automatically by scripts/activate_env.sh.
export AUTOFIX_ATTN=$ATTN
EOF
echo "  wrote .autofix-env (AUTOFIX_ATTN=$ATTN)"

echo
echo "=== Done ==="
echo "  conda activate $ENV_NAME"
echo "  autofix-data --limit-per-source 500"
echo
echo "The stale ./.venv is no longer used. Remove it to avoid confusion:"
echo "  rm -rf .venv .venv-cpu"

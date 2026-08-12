#!/bin/bash
# One-time environment setup. Run this ON A COMPUTE NODE, not the login node:
# flash-attn compiles against the GPU toolchain and needs nvcc present.
#
#   srun --partition=l40 --gres=gpu:1 --time=01:00:00 --pty bash
#   bash scripts/setup_env.sh
set -euo pipefail

echo "=== 1. CUDA module ==="
# Adjust to whatever `module avail cuda` showed. Harmless if there is no module
# system and CUDA is already on PATH.
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || \
  echo "  no cuda module loaded; assuming CUDA is already on PATH"
nvcc --version 2>/dev/null | tail -1 || echo "  warning: nvcc not found, flash-attn will fail to build"

echo "=== 2. Python environment ==="
# Conda is present on this cluster (the shell prompt shows an active base env),
# but a plain venv is used deliberately: it is isolated from whatever the base
# environment already has installed, which avoids a whole class of version
# conflicts that are painful to debug on a shared machine.
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel setuptools
python --version

echo "=== 3. PyTorch (must come before everything else) ==="
# Installing torch first pins the CUDA build. If pip resolves torch later as a
# transitive dependency it may pull a CPU wheel, and every subsequent package
# will silently compile against the wrong runtime.
# Change cu121 to match the CUDA version reported above.
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('  torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

echo "=== 4. Project + training dependencies ==="
pip install --quiet -e ".[train,serve,dev]"

echo "=== 5. flash-attention (must come last) ==="
# Compiles against the torch that is now installed; --no-build-isolation is
# required or it builds against a fresh torch download instead.
pip install --quiet flash-attn --no-build-isolation || \
  echo "  flash-attn build failed - training still works, set attn_implementation='sdpa' in training/run.py"

echo "=== 6. Verify ==="
python - <<'PYCHECK'
import torch
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  Device         : {p.name}")
    print(f"  VRAM           : {p.total_memory / 1024**3:.1f} GB")
    print(f"  bf16 supported : {torch.cuda.is_bf16_supported()}")
PYCHECK

python -c "from autofix.training.sizing import plan_editor, render_plan, detect_vram_gb; \
v=detect_vram_gb(); print(); print(render_plan(plan_editor(v), v))"

echo
echo "Setup complete. Activate with: source .venv/bin/activate"

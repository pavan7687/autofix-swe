#!/bin/bash
#SBATCH --job-name=autofix-bootstrap
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=25m0803
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=bootstrap-%j.out
#SBATCH --error=bootstrap-%j.out

# EVERYTHING remaining in setup, in one job.
#
#   sbatch scripts/bootstrap.sbatch
#   tail -f bootstrap-*.out
#
# Deliberately one script rather than several. Each separate submission costs a
# discover-fix-push-pull cycle, and that round-trip has been the real time sink
# so far - not the work itself. This finds the container runtime, builds the
# Python environment, and verifies the model configuration in a single pass.
#
# It never exits early on a soft failure: a missing optional component is
# recorded and the run continues, so ONE output file answers every remaining
# question instead of surfacing them one at a time.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "=============================================="
echo " autofix-swe bootstrap"
echo " host $(hostname) | $(date)"
echo "=============================================="

NOTES=""
note() { NOTES="${NOTES}
  - $1"; }

# ---------------------------------------------------------------- 1. GPU ----
echo
echo "### 1. GPU"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv || note "nvidia-smi missing"

# ---------------------------------------------------- 2. container runtime ----
echo
echo "### 2. Container runtime  (the reward function depends on this)"
RUNTIME="none"
for rt in apptainer singularity podman docker; do
  command -v "$rt" >/dev/null 2>&1 || { echo "  $rt: not installed"; continue; }
  echo -n "  $rt: installed -> "
  case "$rt" in
    docker)
      timeout 10 docker ps >/dev/null 2>&1 && { echo "USABLE"; RUNTIME=docker; } || echo "no socket permission" ;;
    podman)
      timeout 40 podman run --rm alpine true >/dev/null 2>&1 && { echo "USABLE (rootless)"; RUNTIME=podman; } || echo "cannot run" ;;
    apptainer|singularity)
      timeout 120 "$rt" exec docker://alpine true >/dev/null 2>&1 && { echo "USABLE"; RUNTIME="$rt"; } || echo "cannot run" ;;
  esac
  [ "$RUNTIME" != "none" ] && break
done
echo "  => usable runtime: $RUNTIME"
[ "$RUNTIME" = "none" ] && note "NO container runtime - needs the subprocess-isolation backend"

echo -n "  unprivileged user namespaces: "
unshare --user --pid --fork --mount-proc true 2>/dev/null && echo "WORK" || echo "blocked"

# ------------------------------------------------------------ 3. toolchain ----
echo
echo "### 3. CUDA toolchain"
if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -1; else
  echo "  nvcc absent - flash-attn cannot compile; sdpa fallback will be used"
fi

# ---------------------------------------------------------- 4. environment ----
echo
echo "### 4. Python environment"
[ -d .venv ] || python -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip wheel setuptools
echo "  python $(python --version 2>&1 | cut -d' ' -f2)"

echo "  installing torch (cu121) ..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu121 || note "torch install FAILED"
python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())" || note "torch import FAILED"

echo "  installing project + training deps ..."
pip install -q -e ".[train,serve]" || note "project deps FAILED"

echo "  attempting flash-attn (optional) ..."
if pip install -q flash-attn --no-build-isolation 2>/dev/null; then
  echo "  flash-attn OK"; ATTN=flash_attention_2
else
  echo "  flash-attn unavailable -> sdpa (slower, numerically identical)"
  ATTN=sdpa; note "flash-attn unavailable (non-fatal): export AUTOFIX_ATTN=sdpa"
fi

# --------------------------------------------------------------- 5. verify ----
echo
echo "### 5. Verify imports and device"
python - <<'PYEOF'
import torch
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  device         : {p.name}")
    print(f"  VRAM           : {p.total_memory / 1024**3:.2f} GiB")
    print(f"  bf16 supported : {torch.cuda.is_bf16_supported()}")
    print(f"  GPUs visible   : {torch.cuda.device_count()}")
for mod in ("transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets", "vllm"):
    try:
        m = __import__(mod)
        print(f"  {mod:<14}: {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"  {mod:<14}: FAILED ({str(exc)[:60]})")
PYEOF

echo
echo "### 6. Sizing plan for this GPU"
python -c "
from autofix.training.sizing import plan_editor, render_plan, detect_vram_gb
v = detect_vram_gb(); print(render_plan(plan_editor(v), v))
" || note "sizing check FAILED"

# --------------------------------------------------------------- summary ----
echo
echo "=============================================="
if [ -z "$NOTES" ]; then
  echo " BOOTSTRAP CLEAN"
else
  echo " BOOTSTRAP COMPLETE with notes:${NOTES}"
fi
echo
echo " Container runtime : $RUNTIME"
echo " Attention backend : $ATTN"
echo
echo " Next: source .venv/bin/activate && autofix-data --limit-per-source 500"
echo "=============================================="

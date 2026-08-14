#!/bin/bash
#SBATCH --job-name=autofix-bnb
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=25m0803
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=bnb-fix-%j.out
#SBATCH --error=bnb-fix-%j.out

# Find a bitsandbytes build that runs on this cluster's glibc.
#
#   sbatch scripts/fix_bitsandbytes.sbatch
#   cat bnb-fix-*.out
#
# This cluster is RHEL 8: glibc 2.28, kernel 4.18. bitsandbytes wheels from
# ~0.44 onward are built on manylinux_2_24+ and link against GLIBC_2.34, so
# they fail at dlopen with "version GLIBC_2.34 not found". Older releases used
# manylinux2014 (glibc 2.17) and should load.
#
# Without 4-bit quantisation a 32B model cannot fit in 45GB, so this determines
# which editor size is achievable. Each candidate is installed and then actually
# exercised on the GPU - importing successfully is not proof that the kernels run.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/activate_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== system ==="
ldd --version | head -1
echo "kernel $(uname -r)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo

test_bnb() {
  python - <<'PYEOF' 2>&1 | tail -5
import torch
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    layer = Linear4bit(256, 256, compute_dtype=torch.bfloat16).cuda()
    out = layer(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16))
    print(f"RESULT OK {bnb.__version__} output={tuple(out.shape)}")
except Exception as exc:
    print(f"RESULT FAIL {type(exc).__name__}: {str(exc)[:120]}")
PYEOF
}

WORKING=""
for version in 0.43.3 0.42.0 0.41.3.post2; do
  echo "--- trying bitsandbytes==$version ---"
  pip install -q "bitsandbytes==$version" 2>&1 | tail -2
  result="$(test_bnb)"
  echo "$result"
  if echo "$result" | grep -q "RESULT OK"; then
    WORKING="$version"
    break
  fi
  echo
done

echo
echo "=============================================="
if [ -n "$WORKING" ]; then
  echo " 4-bit WORKS with bitsandbytes==$WORKING"
  echo " Pin it in pyproject.toml and train the 32B editor as planned."
else
  echo " NO working bitsandbytes build on this glibc."
  echo
  echo " 4-bit quantisation is unavailable, so the editor must run in bf16."
  echo " On a 45GB A40 that means:"
  echo "   7B  bf16 @ 8K context  ~25GB  <- recommended"
  echo "   14B bf16 @ 4K context  ~40GB  (tight, and truncation hurts more"
  echo "                                  than the extra parameters help)"
  echo "   32B  does not fit without quantisation"
  echo
  echo " Set in .env:  EDITOR_SIZE_OVERRIDE=7b"
  echo "               QUANTIZATION=none"
fi
echo "=============================================="

#!/bin/bash
#SBATCH --job-name=autofix-bnbtest
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=25m0803
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=bnb-test-%j.out
#SBATCH --error=bnb-test-%j.out

# Verify the currently-installed bitsandbytes on real GPU hardware.
# Installs NOTHING - compute nodes have no network. Use try_bitsandbytes.sh
# on the login node first.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/activate_env.sh

echo "host $(hostname) | glibc $(ldd --version | head -1 | awk '{print $NF}')"
echo

python - <<'PYEOF'
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes {bnb.__version__}")
except Exception as exc:
    print(f"IMPORT FAILED: {exc}")
    raise SystemExit(1)

# Importing is not proof. The native library is dlopen'd lazily, so a glibc
# mismatch only surfaces when a kernel actually runs.
try:
    from bitsandbytes.nn import Linear4bit
    layer = Linear4bit(512, 512, compute_dtype=torch.bfloat16).cuda()
    out = layer(torch.randn(8, 512, device="cuda", dtype=torch.bfloat16))
    assert out.shape == (8, 512)
    print()
    print("  4-bit forward pass OK")
    print("  => QLoRA works. The 32B editor is viable.")
    print("     Keep QUANTIZATION=auto in .env")
except Exception as exc:
    print()
    print(f"  4-bit FAILED: {type(exc).__name__}: {str(exc)[:160]}")
    print("  => train in bf16 instead. Set in .env:")
    print("       QUANTIZATION=none")
    print("       EDITOR_SIZE_OVERRIDE=7b")
    raise SystemExit(1)
PYEOF

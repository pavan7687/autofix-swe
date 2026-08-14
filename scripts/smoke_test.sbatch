#!/bin/bash
#SBATCH --job-name=autofix-smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/smoke-%j.out
#SBATCH --error=artifacts/runs/smoke-%j.out
#
# ACCOUNT AND QOS - discovered via `bash scripts/slurm_info.sh`
#
# NOT the `debug` partition, despite these being short jobs: the `debug`
# QOS defines no limits of its own, so it falls back to a root-level
# GrpSubmitJobs cap of 20 that is saturated cluster-wide (93 submitted).
# Every submission there fails with AssocGrpSubmitJobsLimit regardless of
# account. `a40` has its own GrpSubmit=100 and usually has idle nodes.
#
# On this cluster the QOS name must MATCH THE PARTITION NAME. Every partition
# declares `AllowQos=<its own name>`, so a job on `a40` needs `--qos=a40`.
# A single exported SBATCH_QOS therefore cannot work across partitions, which
# is why the value is written explicitly in each script.
#
# Change --account if you are billing a different project. Your associations:
#   25m0803  personal, FairShare 0.43   <- used here, better priority
#   cccp     shared,   FairShare 0.17
#SBATCH --account=25m0803
#SBATCH --qos=a40

# End-to-end proof on a tiny scale, on the debug partition, in ~20 minutes.
# Run this BEFORE queuing a 24-hour job. It catches config errors, missing
# datasets, OOM at step 1, and broken loss masking - the four things that
# otherwise waste a full day of queue time.
set -euo pipefail

# SLURM runs a copy of this script from its spool directory. Relative paths like
# `.venv/bin/activate` only work if we return to the submission directory first.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "Smoke test on $(hostname), $(date)"

module load cuda/12.1 2>/dev/null || true
# Activates the conda env (or .venv) and loads AUTOFIX_ATTN.
source scripts/activate_env.sh

# Compute nodes have no DNS. Force offline so a missing weight fails instantly
# with a clear message instead of five DNS retries per file. Run
# scripts/fetch_models.sh on the login node first.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv


# Report what torch actually sees. `nvidia-smi` showing a GPU does not mean
# torch can use it - a CUDA/driver mismatch presents as "no GPU detected" in
# the sizing plan, which silently selects a smaller model profile.
python - <<'PYEOF'
import torch
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory / 1024**3:.2f} GiB  sm_{p.major}{p.minor}")
else:
    print()
    print("  FATAL: torch cannot see the GPU.")
    print("  Almost always a CUDA/driver mismatch. Compare the driver's maximum")
    print("  CUDA version (nvidia-smi, top right) with the build above.")
    print("  Fix by reinstalling torch against a matching CUDA, e.g.:")
    print("    TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128 \\")
    print("      bash scripts/setup_env.sh")
    raise SystemExit(1)
PYEOF

# Abort here rather than running a "successful" smoke test on CPU: a CPU run
# would take hours and prove nothing about the real training path.
if [ $? -ne 0 ]; then
  echo "Aborting: no usable GPU."
  exit 1
fi

echo "--- bitsandbytes 4-bit check (the 32B editor depends on this) ---"
python - <<'PYEOF'
# A login-node "compiled without GPU support" warning is meaningless - there is
# no GPU there. This is the check that counts: quantise a real tensor on the
# actual training hardware.
try:
    import torch, bitsandbytes as bnb
    print(f"  bitsandbytes {bnb.__version__}")
    from bitsandbytes.nn import Linear4bit
    layer = Linear4bit(256, 256, compute_dtype=torch.bfloat16).cuda()
    out = layer(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16))
    print(f"  4-bit Linear forward OK, output {tuple(out.shape)}")
    print("  => QLoRA will work on this node")
except Exception as exc:
    print(f"  FAILED: {type(exc).__name__}: {exc}")
    print("  => the 32B editor cannot be 4-bit quantised on this node.")
    print("     Try: pip install 'bitsandbytes==0.43.3'")
PYEOF

echo
echo "--- sizing plan ---"
autofix-train --task retrieval --dry-run

echo "--- 20 optimisation steps on the reranker ---"
# If loss does not move here, the loss mask is broken. Nothing else matters
# until this passes.
autofix-train --task retrieval --max-steps 20 --run-name smoke-reranker --no-resume

echo "Smoke test finished $(date)"

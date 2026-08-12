#!/bin/bash
#SBATCH --job-name=autofix-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/smoke-%j.out
#SBATCH --error=artifacts/runs/smoke-%j.err

# End-to-end proof on a tiny scale, on the debug partition, in ~20 minutes.
# Run this BEFORE queuing a 24-hour job. It catches config errors, missing
# datasets, OOM at step 1, and broken loss masking - the four things that
# otherwise waste a full day of queue time.
set -euo pipefail
echo "Smoke test on $(hostname), $(date)"

module load cuda/12.1 2>/dev/null || true
source .venv/bin/activate

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "--- sizing plan (no GPU work) ---"
autofix-train --task retrieval --dry-run

echo "--- 20 optimisation steps on the reranker ---"
# If loss does not move here, the loss mask is broken. Nothing else matters
# until this passes.
autofix-train --task retrieval --max-steps 20 --run-name smoke-reranker --no-resume

echo "Smoke test finished $(date)"

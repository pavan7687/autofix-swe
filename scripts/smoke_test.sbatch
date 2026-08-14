#!/bin/bash
#SBATCH --job-name=autofix-smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/smoke-%j.out
#SBATCH --error=artifacts/runs/smoke-%j.err
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

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "--- sizing plan (no GPU work) ---"
autofix-train --task retrieval --dry-run

echo "--- 20 optimisation steps on the reranker ---"
# If loss does not move here, the loss mask is broken. Nothing else matters
# until this passes.
autofix-train --task retrieval --max-steps 20 --run-name smoke-reranker --no-resume

echo "Smoke test finished $(date)"

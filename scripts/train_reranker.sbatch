#!/bin/bash
#SBATCH --job-name=autofix-reranker
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/runs/reranker-%j.out
#SBATCH --error=artifacts/runs/reranker-%j.out
#SBATCH --requeue
#
# ACCOUNT AND QOS - discovered via `bash scripts/slurm_info.sh`
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
#SBATCH --qos=l40

set -euo pipefail

# SLURM runs a copy of this script from its spool directory. Relative paths like
# `.venv/bin/activate` only work if we return to the submission directory first.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

module load cuda/12.1 2>/dev/null || true
# Activates the conda env (or .venv) and loads AUTOFIX_ATTN.
source scripts/activate_env.sh

# Compute nodes have no DNS. Force offline so a missing weight fails instantly
# with a clear message instead of five DNS retries per file. Run
# scripts/fetch_models.sh on the login node first.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


srun autofix-train --task retrieval --wandb
echo "Finished $(date)"

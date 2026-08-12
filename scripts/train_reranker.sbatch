#!/bin/bash
#SBATCH --job-name=autofix-reranker
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=artifacts/runs/reranker-%j.out
#SBATCH --error=artifacts/runs/reranker-%j.err
#SBATCH --requeue
#
# ACCOUNT / QOS
# This cluster rejects jobs with `Invalid qos specification` when the QOS is
# missing or not permitted. Discover yours with `bash scripts/slurm_info.sh`,
# then either uncomment the two lines below or export before submitting:
#     export SBATCH_ACCOUNT=<account>
#     export SBATCH_QOS=<qos>
# SLURM reads those environment variables automatically, so exporting them once
# per session configures every script here at no maintenance cost.
##SBATCH --account=CHANGE_ME
##SBATCH --qos=CHANGE_ME

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

module load cuda/12.1 2>/dev/null || true
source .venv/bin/activate

srun autofix-train --task retrieval --wandb
echo "Finished $(date)"

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

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

module load cuda/12.1 2>/dev/null || true
source .venv/bin/activate

srun autofix-train --task retrieval --wandb
echo "Finished $(date)"

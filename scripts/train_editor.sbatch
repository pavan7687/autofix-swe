#!/bin/bash
#SBATCH --job-name=autofix-editor
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/runs/editor-%j.out
#SBATCH --error=artifacts/runs/editor-%j.err
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

# Partitions on this cluster (from `sinfo -o "%P %G"`):
#   l40*  gpu:8   <- default, used here
#   dgx   gpu:8   <- try this if the L40s are queued; may be A100/H100
#   a40   gpu:4
#   debug gpu:4   <- short jobs only, good for the first smoke test
# GRES is untyped on this cluster, so `--gres=gpu:1` and NOT `gpu:a100:1`.
# We request ONE GPU: the training code is single-device by design.

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

module load cuda/12.1 2>/dev/null || echo "no module system; assuming CUDA on PATH"
source .venv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv

# --dry-run first so a bad config fails in 30 seconds, not 6 hours.
autofix-train --task editing --dry-run

# The training call resumes from the latest checkpoint automatically, which is
# what makes --requeue safe: a preempted job restarts where it left off.
srun autofix-train --task editing --wandb

echo "Finished $(date)"

#!/bin/bash
#SBATCH --job-name=autofix-editor
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=artifacts/runs/editor-%j.out
#SBATCH --error=artifacts/runs/editor-%j.err
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
#SBATCH --qos=a40

# Partitions on this cluster (from `sinfo -o "%P %G"`):
#   l40*  gpu:8   <- default, used here
#   dgx   gpu:8   <- try this if the L40s are queued; may be A100/H100
#   a40   gpu:4
#   debug gpu:4   <- short jobs only, good for the first smoke test
# GRES is untyped on this cluster, so `--gres=gpu:1` and NOT `gpu:a100:1`.
# We request ONE GPU: the training code is single-device by design.

set -euo pipefail

# SLURM runs a copy of this script from its spool directory. Relative paths like
# `.venv/bin/activate` only work if we return to the submission directory first.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

module load cuda/12.1 2>/dev/null || echo "no module system; assuming CUDA on PATH"
# Activates the conda env (or .venv) and loads AUTOFIX_ATTN.
source scripts/activate_env.sh

# Compute nodes have no DNS. Force offline so a missing weight fails instantly
# with a clear message instead of five DNS retries per file. Run
# scripts/fetch_models.sh on the login node first.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


nvidia-smi --query-gpu=name,memory.total --format=csv

# --dry-run first so a bad config fails in 30 seconds, not 6 hours.
autofix-train --task editing --dry-run

# The training call resumes from the latest checkpoint automatically, which is
# what makes --requeue safe: a preempted job restarts where it left off.
srun autofix-train --task editing --wandb

echo "Finished $(date)"

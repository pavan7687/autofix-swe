#!/bin/bash
#SBATCH --job-name=autofix-editor-mg
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=25m0803
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=artifacts/runs/editor-mg-%j.out
#SBATCH --error=artifacts/runs/editor-mg-%j.out
#SBATCH --requeue

# Editor training across all 4 GPUs of one a40 node.
#
# DDP, not model parallelism: a 4-bit 32B is ~18GB and fits on one 45GB A40, so
# the right move is a full copy per GPU with the batch split between them.
# Sharding one model across four cards would leave three idle at any moment.
#
# Expected speedup is ~3.5x, not 4x - gradient all-reduce costs something.
# 90h single-GPU becomes roughly 26h, which fits inside one 48h allocation
# instead of needing three preempted restarts.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

# Activates the conda env (or .venv) and loads AUTOFIX_ATTN.
source scripts/activate_env.sh

# Compute nodes have no DNS. Force offline so a missing weight fails instantly
# with a clear message instead of five DNS retries per file. Run
# scripts/fetch_models.sh on the login node first.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Set by bootstrap if flash-attn could not be built. Harmless when unset.
export AUTOFIX_ATTN="${AUTOFIX_ATTN:-flash_attention_2}"
echo "attention backend: $AUTOFIX_ATTN"

# Fail fast on a bad config: 30 seconds here beats discovering it 6 hours in.
autofix-train --task editing --dry-run

NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "launching DDP across $NGPU GPUs"

# --standalone: single node, no rendezvous server needed.
srun torchrun --standalone --nproc_per_node="$NGPU" \
  -m autofix.training.run --task editing --wandb

echo "Finished $(date)"

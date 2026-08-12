#!/bin/bash
# Run this FIRST, on a compute node, before configuring anything else.
#
# Interactive (this cluster allows --pty ONLY on the `interactive` partition):
#   srun --partition=interactive --gres=gpu:1 --time=00:10:00 --pty bash scripts/check_cluster.sh
#
# Batch, if the interactive partition is unavailable:
#   sbatch scripts/check_cluster.sbatch && sleep 60 && cat cluster-check-*.out
#
# It answers the four questions that determine whether this project can run
# here at all, and how it must be configured.
echo "=============================================="
echo " autofix-swe cluster capability check"
echo " host: $(hostname)   date: $(date)"
echo "=============================================="

echo
echo "--- 1. GPU: model and VRAM -------------------"
# Determines editor size and context length (see src/autofix/training/sizing.py).
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
else
  echo "  nvidia-smi NOT FOUND - are you on a compute node? Login nodes have no GPU."
fi

echo
echo "--- 2. Container runtime ---------------------"
# This is the reward function. Without one of these there is no resolve-rate
# evaluation and no rejection sampling.
FOUND_RUNTIME=""
for rt in docker podman apptainer singularity; do
  if command -v $rt >/dev/null 2>&1; then
    echo "  FOUND: $rt ($($rt --version 2>&1 | head -1))"
    FOUND_RUNTIME="$rt"
  fi
done
[ -z "$FOUND_RUNTIME" ] && echo "  NONE FOUND - ask your admin which is available."

if command -v docker >/dev/null 2>&1; then
  echo -n "  docker daemon reachable: "
  docker ps >/dev/null 2>&1 && echo "YES" || echo "NO (no socket permission - normal on shared clusters)"
fi

echo
echo "--- 3. CUDA toolchain ------------------------"
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 || echo "  nvcc not on PATH"
if command -v module >/dev/null 2>&1; then
  echo "  cuda modules available:"
  module avail cuda 2>&1 | head -20
else
  echo "  no environment-modules system"
fi

echo
echo "--- 4. Storage -------------------------------"
# Datasets + model weights + repo checkouts need ~200GB. Home quota is usually
# far smaller than scratch; point DATA_ROOT/MODEL_ROOT at scratch if so.
echo "  HOME:  $HOME"
df -h "$HOME" 2>/dev/null | tail -1
for candidate in /scratch /scratch/$USER /work /work/$USER "$SCRATCH"; do
  [ -d "$candidate" ] && { echo "  SCRATCH candidate: $candidate"; df -h "$candidate" | tail -1; }
done
command -v quota >/dev/null 2>&1 && { echo "  quota:"; quota -s 2>/dev/null | head -5; }

echo
echo "--- 5. Python --------------------------------"
python --version 2>&1
echo "  conda: $(command -v conda >/dev/null 2>&1 && conda --version || echo 'not found')"

echo
echo "=============================================="
echo " Report items 1, 2 and 4 back before training."
echo "=============================================="

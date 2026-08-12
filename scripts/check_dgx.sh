#!/bin/bash
# What GPU is in the DGX nodes?
#
#   bash scripts/check_dgx.sh
#
# This matters: the `dgx` partition allows 6-day jobs, and if those nodes hold
# A100-80GB or H100 cards, the editor can train at a 16K context instead of 8K
# (see src/autofix/training/sizing.py). Worth 60 seconds to find out.
#
# cn11-dgx is in the `interactive` partition, so it can be queried directly.
set -euo pipefail

echo "Querying cn11-dgx via the interactive partition..."
srun --partition=interactive --qos=interactive --account=25m0803 \
     --nodelist=cn11-dgx --gres=gpu:1 --time=00:05:00 \
     nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

echo
echo "If that shows A100-SXM4-80GB or H100, retrain on the dgx partition:"
echo "  sed -i 's/--partition=a40/--partition=dgx/; s/--qos=a40/--qos=dgx/' scripts/train_editor.sbatch"

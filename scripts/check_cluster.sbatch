#!/bin/bash
#SBATCH --job-name=autofix-check
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cluster-check-%j.out
#SBATCH --error=cluster-check-%j.out

# Non-interactive version of the capability check.
#
#   sbatch scripts/check_cluster.sbatch
#   cat cluster-check-*.out
#
# Use this when `srun --pty` is unavailable — this cluster restricts interactive
# jobs to the `interactive` partition, so batch is often the path of least
# resistance.
bash "$(dirname "$0")/check_cluster.sh"

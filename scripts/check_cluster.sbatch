#!/bin/bash
#SBATCH --job-name=autofix-check
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cluster-check-%j.out
#SBATCH --error=cluster-check-%j.out
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

# Non-interactive version of the capability check.
#
#   sbatch scripts/check_cluster.sbatch
#   cat cluster-check-*.out
#
# Use this when `srun --pty` is unavailable — this cluster restricts interactive
# jobs to the `interactive` partition, so batch is often the path of least
# resistance.
bash "$(dirname "$0")/check_cluster.sh"

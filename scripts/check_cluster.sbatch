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
#SBATCH --qos=debug

# Non-interactive version of the capability check.
#
#   sbatch scripts/check_cluster.sbatch
#   cat cluster-check-*.out
#
# Use this when `srun --pty` is unavailable — this cluster restricts interactive
# jobs to the `interactive` partition, so batch is often the path of least
# resistance.
bash "$(dirname "$0")/check_cluster.sh"

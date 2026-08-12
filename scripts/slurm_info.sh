#!/bin/bash
# Dump every scheduler constraint that affects how jobs must be submitted.
#
#   bash scripts/slurm_info.sh
#
# Runs on the LOGIN node - these are pure queries, no allocation required.
#
# Clusters differ in which of account / QOS / partition / time limit they
# require, and each missing one produces a different cryptic error. Getting all
# of them in one pass is faster than discovering them one rejection at a time.
echo "=============================================="
echo " SLURM configuration for $USER on $(hostname)"
echo "=============================================="

echo
echo "--- 1. Your associations (account / partition / QOS) ---"
# THE important one: which account and QOS you may actually use.
# Add --account=X and --qos=Y to every sbatch/srun accordingly.
sacctmgr -n show assoc user="$USER" \
  format=Cluster%15,Account%20,Partition%15,QOS%45,DefaultQOS%15 2>/dev/null \
  || echo "  sacctmgr unavailable (cluster may not use slurmdbd)"

echo
echo "--- 2. QOS definitions (limits per QOS) ---"
sacctmgr show qos format=Name%18,Priority%8,MaxWall%12,MaxSubmit%10,MaxTRESPU%28 2>/dev/null \
  | head -30 || echo "  unavailable"

echo
echo "--- 3. Partitions: availability, time limits, nodes ---"
sinfo -o "%20P %5a %12l %6D %10G %N" 2>/dev/null

echo
echo "--- 4. Interactive partition detail ---"
scontrol show partition interactive 2>/dev/null \
  | grep -Ei "PartitionName|AllowGroups|AllowAccounts|AllowQos|DefaultTime|MaxTime|State|TRES" \
  || echo "  no 'interactive' partition"

echo
echo "--- 5. Training partition detail (l40) ---"
scontrol show partition l40 2>/dev/null \
  | grep -Ei "PartitionName|AllowGroups|AllowAccounts|AllowQos|DefaultTime|MaxTime|State|TRES" \
  || echo "  no 'l40' partition"

echo
echo "--- 6. Fair-share / usage ---"
sshare -U 2>/dev/null | head -5 || echo "  sshare unavailable"

echo
echo "--- 7. What is currently running ---"
squeue -u "$USER" 2>/dev/null
echo "  (empty above = no jobs queued)"

echo
echo "=============================================="
echo " Add the Account and QOS from section 1 to"
echo " every sbatch script as:"
echo "   #SBATCH --account=<account>"
echo "   #SBATCH --qos=<qos>"
echo "=============================================="

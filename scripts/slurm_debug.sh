#!/bin/bash
# Diagnose a REJECTED or STUCK job submission.
#
#   bash scripts/slurm_debug.sh
#
# Two failure modes look similar from the outside and have opposite causes:
#
#   AssocGrpSubmitJobsLimit  - the scheduler thinks you already have jobs
#                              submitted, even when `squeue` looks empty. Usually
#                              a stale counter from a cancelled job, or a limit
#                              set on the parent account rather than on you.
#
#   queued but nothing running - the partition looks free because its nodes are
#                                DOWN or DRAINED, not because they are available.
#
# This script separates the two.
echo "=============================================="
echo " SLURM submission diagnostics for $USER"
echo "=============================================="

echo
echo "--- 1. Limits AND current usage counters ---"
# The authoritative view: shows GrpSubmitJobs alongside how many the scheduler
# currently believes you are using. A non-zero count with an empty squeue means
# a stale counter.
scontrol show assoc_mgr user="$USER" flags=assoc 2>/dev/null | head -60 \
  || echo "  unavailable"

echo
echo "--- 2. Association limits per partition ---"
sacctmgr show assoc user="$USER" \
  format=Account%12,Partition%12,QOS%22,GrpSubmit,MaxSubmit,GrpJobs,MaxJobs 2>/dev/null

echo
echo "--- 3. QOS limits (GrpSubmit is the one that bites) ---"
sacctmgr show qos format=Name%14,GrpSubmit,MaxSubmit,MaxJobsPU,MaxSubmitPU,MaxWall%12 2>/dev/null

echo
echo "--- 4. Node states: are they actually up? ---"
# STATE: idle=free, alloc=busy, drain/drng=being emptied, down=offline.
# A partition can look empty in squeue while every node is drained.
sinfo -p interactive,debug,l40,a40,dgx -o "%18N %10P %8t %10G %30E" 2>/dev/null

echo
echo "--- 5. Everything drained or down, with reasons ---"
sinfo -R 2>/dev/null | head -25
echo "  (empty above = all nodes healthy)"

echo
echo "--- 6. Your jobs in EVERY state, not just pending/running ---"
squeue -u "$USER" -t all -o "%.10i %.12P %.10j %.8T %.10M %.20R" 2>/dev/null
echo "  (compare with the counters in section 1)"

echo
echo "--- 7. Recent job history (did something fail or hang?) ---"
sacct -u "$USER" --starttime=now-1days \
  --format=JobID%12,JobName%16,Partition%12,State%20,Elapsed,ExitCode 2>/dev/null | head -20

echo
echo "=============================================="
echo " If section 1 shows a non-zero submit count but"
echo " section 6 is empty, the counter is stale: wait"
echo " a few minutes or ask the admin to reset it."
echo "=============================================="

#!/bin/bash
# One-screen status for every running job.
#
#   bash scripts/watch.sh          # snapshot
#   watch -n 30 bash scripts/watch.sh   # refresh every 30s
#
# Training runs are long and the interesting signals - step rate, loss, OOM -
# are buried in thousands of lines of library chatter. This pulls out only what
# tells you whether a run is healthy.
cd "$(dirname "$0")/.."

echo "=== queue ==="
if [ -z "$(squeue -u "$USER" -h)" ]; then
  echo "  (nothing queued or running)"
else
  squeue -u "$USER" -o "%.9i %.10P %.16j %.2t %.10M %.6D %R"
fi

# An empty queue is ambiguous - jobs may have succeeded, failed, been preempted
# or hit a time limit - and showing nothing at all is the least useful possible
# answer. sacct covers what squeue has already forgotten.
echo
echo "=== finished today ==="
sacct -u "$USER" --starttime=today --noheader \
      --format=JobID%14,JobName%18,State%22,ExitCode%8,Elapsed%10 2>/dev/null \
  | grep -vE "\.(batch|extern|[0-9]+) " | tail -8
echo "  (FAILED/OOM/TIMEOUT above means look at the log; CANCELLED means you did it)"

echo
echo "=== progress ==="
for f in artifacts/runs/*.out; do
  [ -f "$f" ] || continue
  # Only logs belonging to jobs that are still queued or running. A finished
  # run's traceback is not a current alert, and stale files made every check
  # look like a fire.
  # Live jobs, plus anything written in the last 30 minutes so a run that just
  # died is still reported rather than silently vanishing.
  jid=$(basename "$f" | grep -oE "[0-9]+")
  [ -n "$jid" ] || continue
  live=$(squeue -h -j "$jid" -o "%i" 2>/dev/null)
  recent=$(find "$f" -mmin -30 2>/dev/null)
  [ -n "$live" ] || [ -n "$recent" ] || continue
  [ -n "$live" ] || echo "  (job no longer running)"

  echo "--- $(basename "$f") ---"
  # Most recent step-rate line from the tqdm bar.
  last=$(grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s?/?it\]" "$f" | tail -1)
  if [ -n "$last" ]; then
    echo "  step : $last"
    # Turn the rate into a wall-clock estimate. This is the number that decides
    # whether a configuration is viable: a run projecting past the partition's
    # time limit will be killed mid-epoch, so it is better to find out at step
    # 20 than at hour 40.
    cur=$(echo "$last"  | grep -oE "^[0-9]+")
    tot=$(echo "$last"  | grep -oE "^[0-9]+/[0-9]+" | cut -d/ -f2)
    rate=$(echo "$last" | grep -oE "[0-9.]+s/it" | grep -oE "[0-9.]+")
    if [ -n "$rate" ] && [ -n "$tot" ] && [ -n "$cur" ]; then
      awk -v c="$cur" -v t="$tot" -v r="$rate" 'BEGIN {
        remain = (t - c) * r
        printf "  eta  : %.1f h remaining (%.1f h total at %.1fs/step)\n",
               remain/3600, (t*r)/3600, r
        if (t*r > 172800) print "  ALERT: projected run exceeds the 48h a40 limit - reduce epochs or subsample"
      }'
    fi
  fi

  # Most recent loss the Trainer logged.
  loss=$(grep -oE "'loss': [0-9.]+" "$f" | tail -3 | tr '\n' ' ')
  [ -n "$loss" ] && echo "  loss : $loss"

  # Anything that means the run is dead or dying.
  bad=$(grep -oE "(OutOfMemoryError|CUDA out of memory|RuntimeError|Traceback|CANCELLED|DUE TO TIME LIMIT|Killed|Segmentation fault)" "$f" | sort -u | tr '\n' ' ')
  if [ -n "$bad" ]; then
    echo "  ALERT: $bad"
    # The first exception line is almost always the actionable one; everything
    # after it is unwinding.
    first=$(grep -E "^(Error|OSError|RuntimeError|ValueError|ImportError|torch\.|.*Error:)" "$f" | head -2)
    [ -n "$first" ] && echo "$first" | sed 's/^/         /'
  fi

  [ -z "$last$loss$bad" ] && echo "  (starting up)"
done

echo
echo "=== pending jobs: why ==="
squeue -u "$USER" -h -t PD -o "  %.9i %R" | while read -r line; do
  echo "$line"
  case "$line" in
    *QOSMaxGRESPerJob*)  echo "      -> asking for more GPUs than the QOS allows per job" ;;
    *QOSMaxJobsPerUser*) echo "      -> already at your concurrent-job limit" ;;
    *Resources*)         echo "      -> waiting for a free node (normal)" ;;
    *Priority*)          echo "      -> queued behind higher-priority jobs (normal)" ;;
  esac
done

# GPU utilisation deliberately omitted: it needs ssh to the compute node, and
# without passwordless keys that prompts for a password and hangs the script.
# `sstat` works for running jobs and needs no ssh:
echo
echo "=== resource use (running jobs) ==="
for jid in $(squeue -u "$USER" -h -t R -o "%i"); do
  usage=$(sstat -j "${jid}.batch" --format=AveRSS,MaxRSS --noheader -P 2>/dev/null | head -1)
  [ -n "$usage" ] && echo "  $jid  RSS(avg/max): $usage"
done

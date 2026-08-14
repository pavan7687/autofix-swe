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
squeue -u "$USER" -o "%.9i %.10P %.16j %.2t %.10M %.6D %R"

echo
echo "=== progress ==="
for f in artifacts/runs/*.out; do
  [ -f "$f" ] || continue
  # Only files touched in the last 2 hours: old runs are noise.
  [ -n "$(find "$f" -mmin -120 2>/dev/null)" ] || continue

  echo "--- $(basename "$f") ---"
  # Most recent step-rate line from the tqdm bar.
  last=$(grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s?/?it\]" "$f" | tail -1)
  [ -n "$last" ] && echo "  step : $last"

  # Most recent loss the Trainer logged.
  loss=$(grep -oE "'loss': [0-9.]+" "$f" | tail -3 | tr '\n' ' ')
  [ -n "$loss" ] && echo "  loss : $loss"

  # Anything that means the run is dead or dying.
  bad=$(grep -oE "(OutOfMemoryError|CUDA out of memory|RuntimeError|Traceback|CANCELLED|DUE TO TIME LIMIT)" "$f" | sort -u | tr '\n' ' ')
  [ -n "$bad" ] && echo "  ALERT: $bad"

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

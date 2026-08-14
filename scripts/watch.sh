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
echo "=== gpu use on your nodes ==="
for node in $(squeue -u "$USER" -h -o "%N" | tr ',' ' ' | sort -u); do
  [ -n "$node" ] && echo "  $node: $(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$node" \
    'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>/dev/null | tr '\n' ' ' || echo 'unreachable')"
done

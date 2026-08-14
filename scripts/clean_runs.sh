#!/bin/bash
# Archive logs from finished jobs so the monitor only shows live runs.
#   bash scripts/clean_runs.sh
cd "$(dirname "$0")/.."
mkdir -p artifacts/runs/archive
moved=0
for f in artifacts/runs/*.out; do
  [ -f "$f" ] || continue
  jid=$(basename "$f" | grep -oE "[0-9]+")
  [ -n "$jid" ] || continue
  if ! squeue -h -j "$jid" 2>/dev/null | grep -q .; then
    mv "$f" artifacts/runs/archive/ && moved=$((moved + 1))
  fi
done
echo "archived $moved finished-job log(s) to artifacts/runs/archive/"

#!/bin/bash
# Pre-download model weights. RUN ON THE LOGIN NODE.
#
#   bash scripts/fetch_models.sh            # reranker + editor for this GPU
#   bash scripts/fetch_models.sh 7b 14b     # extra sizes as fallbacks
#
# Compute nodes on this cluster have no DNS, so anything that tries to reach
# huggingface.co during a job fails after five retries with
# "Name or service not known". Weights therefore have to be in the shared HF
# cache BEFORE the job starts; the training scripts then run fully offline.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/activate_env.sh

RERANKER="Qwen/Qwen2.5-Coder-1.5B-Instruct"
declare -A EDITORS=(
  [7b]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [14b]="Qwen/Qwen2.5-Coder-14B-Instruct"
  [32b]="Qwen/Qwen2.5-Coder-32B-Instruct"
)

# Default: the 32B chosen by the sizing table for a 45GB A40.
SIZES=("${@:-32b}")

echo "HF cache: ${HF_HOME:-$HOME/.cache/huggingface}"
echo

fetch() {
  local repo="$1"
  echo "--- $repo ---"
  python - "$repo" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

repo = sys.argv[1]
path = snapshot_download(
    repo_id=repo,
    # Skip duplicate formats: transformers reads safetensors, so pulling the
    # .bin copies as well would double the download for no benefit.
    ignore_patterns=["*.pth", "*.bin", "*.msgpack", "*.h5", "*consolidated*"],
    max_workers=8,
)
print(f"  cached at {path}")
PYEOF
}

fetch "$RERANKER"
for size in "${SIZES[@]}"; do
  repo="${EDITORS[$size]:-}"
  [ -z "$repo" ] && { echo "unknown size '$size' (use 7b, 14b or 32b)"; exit 1; }
  fetch "$repo"
done

echo
echo "Done. Cached models:"
du -sh "${HF_HOME:-$HOME/.cache/huggingface}/hub"/models--Qwen--* 2>/dev/null || true
echo
echo "Training jobs now run offline (HF_HUB_OFFLINE=1 is set in the sbatch scripts)."

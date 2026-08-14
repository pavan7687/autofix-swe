#!/bin/bash
# Install a candidate bitsandbytes version. RUN ON THE LOGIN NODE.
#
#   bash scripts/try_bitsandbytes.sh 0.43.3
#   sbatch scripts/test_bitsandbytes.sbatch     # then verify on a GPU node
#
# Split across two nodes on purpose: only the login node can reach PyPI, and
# only a compute node has a GPU to test against. An earlier attempt to do both
# inside one batch job silently failed - pip reported "from versions: none"
# (no network) and the test then re-checked the already-installed version three
# times, producing a confident but meaningless conclusion.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-0.43.3}"
source scripts/activate_env.sh

echo "Installing bitsandbytes==$VERSION ..."
pip install "bitsandbytes==$VERSION"

python -c "
import bitsandbytes, pathlib
p = pathlib.Path(bitsandbytes.__file__).parent
libs = sorted(x.name for x in p.glob('*.so'))
print(f'  installed {bitsandbytes.__version__}')
print(f'  native libs: {libs}')
" 2>&1 | grep -v "compiled without GPU support" || true

echo
echo "Now verify it actually loads on a GPU node:"
echo "  sbatch scripts/test_bitsandbytes.sbatch"

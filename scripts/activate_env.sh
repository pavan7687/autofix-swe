#!/bin/bash
# Shared environment activation, sourced by setup and every batch script.
#
#   source scripts/activate_env.sh
#
# Prefers a conda environment named `autofix` and falls back to a local .venv,
# so the batch scripts do not need to know which one setup created.
if [ -n "${CONDA_EXE:-}" ] || command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if conda env list | grep -qE '^autofix\s'; then
      conda activate autofix
      [ -f .autofix-env ] && source .autofix-env
      return 0 2>/dev/null || exit 0
    fi
  fi
fi

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  [ -f .autofix-env ] && source .autofix-env
  return 0 2>/dev/null || exit 0
fi

echo "No environment found. Run: bash scripts/setup_env.sh" >&2
return 1 2>/dev/null || exit 1

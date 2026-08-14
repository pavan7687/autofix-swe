#!/bin/bash
# Shared environment activation, sourced by setup and every batch script.
#
#   source scripts/activate_env.sh
#
# Prefers a conda environment named `autofix` and falls back to a local .venv,
# so the batch scripts do not need to know which one setup created.
# Drop any inherited virtualenv first: if one is active its bin/ sits earlier
# on PATH than the conda env, and `python` silently resolves to the wrong
# interpreter even though `conda activate` appeared to succeed.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate 2>/dev/null || true
  unset VIRTUAL_ENV
fi

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

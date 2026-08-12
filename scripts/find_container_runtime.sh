#!/bin/bash
# Exhaustive search for a usable container runtime.
#
#   sbatch --partition=a40 --qos=a40 --time=00:10:00 --gres=gpu:1 \
#          --wrap "bash $PWD/scripts/find_container_runtime.sh" \
#          --output=container-search-%j.out
#
# The sandbox is this project's reward function: it decides whether a generated
# patch actually passes a repository's tests. Without SOME isolation mechanism
# there is no resolve-rate metric and no rejection sampling.
#
# `docker` being installed is not enough - on a shared cluster the daemon socket
# is almost always root-only. What we need is a runtime that works UNPRIVILEGED.
echo "=== Container runtime search on $(hostname) ==="

echo
echo "--- Binaries on PATH ---"
for rt in docker podman apptainer singularity ch-run charliecloud shifter udocker nerdctl; do
  p=$(command -v $rt 2>/dev/null) && echo "  $rt -> $p" || echo "  $rt : not on PATH"
done

echo
echo "--- Common install locations (may not be on PATH) ---"
for d in /usr/bin /usr/local/bin /opt/apptainer/bin /opt/singularity/bin \
         /usr/local/apptainer/bin /cm/shared/apps /opt/ohpc/pub; do
  [ -d "$d" ] && find "$d" -maxdepth 2 \
      \( -name "apptainer*" -o -name "singularity*" -o -name "podman*" \) \
      2>/dev/null | head -5
done

echo
echo "--- Environment modules (some clusters hide runtimes here) ---"
if command -v module >/dev/null 2>&1; then
  module avail 2>&1 | grep -iE "apptainer|singularity|podman|container" || echo "  none found"
else
  echo "  no module system"
fi
command -v spack >/dev/null 2>&1 && spack find 2>/dev/null | grep -iE "apptainer|singularity"

echo
echo "--- Can we actually RUN anything? ---"
if command -v docker >/dev/null 2>&1; then
  echo -n "  docker:     "; timeout 10 docker ps >/dev/null 2>&1 && echo "USABLE" || echo "no socket permission"
fi
if command -v podman >/dev/null 2>&1; then
  echo -n "  podman:     "; timeout 20 podman run --rm alpine true >/dev/null 2>&1 && echo "USABLE (rootless)" || echo "present but failed"
fi
if command -v apptainer >/dev/null 2>&1; then
  echo -n "  apptainer:  "; timeout 60 apptainer exec docker://alpine true >/dev/null 2>&1 && echo "USABLE" || echo "present but failed"
fi
if command -v singularity >/dev/null 2>&1; then
  echo -n "  singularity:"; timeout 60 singularity exec docker://alpine true >/dev/null 2>&1 && echo "USABLE" || echo "present but failed"
fi

echo
echo "--- Unprivileged user namespaces (needed for rootless containers) ---"
# If this is 1 and max_user_namespaces > 0, rootless podman/apptainer can work
# even without a system install - and a plain-subprocess fallback can at least
# use namespace isolation.
for f in /proc/sys/kernel/unprivileged_userns_clone /proc/sys/user/max_user_namespaces; do
  [ -r "$f" ] && echo "  $f = $(cat $f)"
done
echo -n "  unshare test: "
unshare --user --pid --fork --mount-proc true 2>/dev/null && echo "WORKS" || echo "blocked"

echo
echo "--- Your groups (is 'docker' among them?) ---"
id

echo
echo "=== If nothing is USABLE, ask the admin: ==="
echo "  'Is Apptainer available, or can I be added to the docker group?'"
echo "  Apptainer is the standard HPC answer and is designed for exactly this."

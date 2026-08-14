"""Choosing a sandbox backend.

The rest of the codebase asks for "a sandbox" and gets whichever backend this
host can actually support. Keeping the choice in one place means the pipeline,
the rejection sampler and the evaluator never branch on it.

Selection order, best isolation first:

1. **Docker** - a real container per run. Preferred when the daemon socket is
   reachable, which on a shared cluster it usually is not.
2. **Local namespaces** - `unshare` user/PID/network namespaces plus rlimits.
   No daemon, no root, no image. Weaker filesystem isolation, identical network
   and process isolation.
3. **Plain subprocess** - LocalSandbox degrades to this automatically if even
   user namespaces are unavailable. Correctness still holds; isolation does not.

`SANDBOX_BACKEND` pins the choice explicitly, which matters for experiments:
a resolve rate measured under Docker and one measured under namespaces should
be reported as the same configuration, so the backend is recorded in the run
manifest rather than silently varying.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from autofix.config import Settings
from autofix.logging_conf import get_logger
from autofix.models import TestOutcome
from autofix.sandbox.detect import Toolchain

log = get_logger(__name__)


class Sandbox(Protocol):
    """What the pipeline requires of any backend."""

    async def __aenter__(self) -> Sandbox: ...
    async def __aexit__(self, *args: object) -> None: ...
    async def prepare(self) -> TestOutcome: ...
    async def run_tests(
        self, command: list[str] | None = ..., timeout: int | None = ...
    ) -> TestOutcome: ...
    async def run_single_test(self, target: str) -> TestOutcome: ...


def docker_usable(settings: Settings) -> bool:
    """True only if the daemon actually answers - installed is not enough."""
    try:
        import docker
        from docker.errors import DockerException
    except ImportError:
        return False
    try:
        client = (
            docker.DockerClient(base_url=settings.docker_host, timeout=10)
            if settings.docker_host
            else docker.from_env(timeout=10)
        )
        client.ping()
        client.close()
    except (DockerException, Exception):  # noqa: BLE001
        return False
    return True


def create_sandbox(
    settings: Settings, toolchain: Toolchain, repo_root: Path
) -> Sandbox:
    backend = (settings.sandbox_backend or "auto").lower()

    if backend in ("auto", "docker") and docker_usable(settings):
        from autofix.sandbox.runner import DockerSandbox

        log.info("sandbox.backend", backend="docker")
        return DockerSandbox(settings, toolchain, repo_root)

    if backend == "docker":
        raise RuntimeError(
            "SANDBOX_BACKEND=docker but the Docker daemon is unreachable. "
            "Set SANDBOX_BACKEND=local to use namespace isolation instead."
        )

    from autofix.sandbox.local import LocalSandbox, namespaces_available

    log.info("sandbox.backend", backend="local",
             namespaces=namespaces_available())
    return LocalSandbox(settings, toolchain, repo_root)


def describe_backend(settings: Settings) -> str:
    """One-line summary for run manifests and eval reports."""
    from autofix.sandbox.local import namespaces_available

    if docker_usable(settings):
        return "docker (container per run)"
    if namespaces_available():
        return "local (unshare user/pid/net namespaces + rlimits)"
    return "local (plain subprocess - NO isolation)"

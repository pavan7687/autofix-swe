"""Docker sandbox for executing untrusted repository code.

This module is the **reward function** for the whole project. A candidate patch
scores 1 if the tests pass inside here and 0 otherwise, so its correctness and
its isolation are both load-bearing.

Why Docker and not a subprocess
-------------------------------
Verification means executing arbitrary third-party repository code *and*
arbitrary code the model just generated, thousands of times, on a shared
cluster node. A subprocess with rlimits shares the filesystem, the network
namespace and the process table; one hostile `conftest.py` reaches everything
the training job can touch. Docker gives a separate mount, PID and network
namespace for roughly 300ms of startup. That trade is not close.

The network is disabled during test execution for a second reason beyond
security: a test that silently reaches the internet makes the reward signal
non-reproducible, and a noisy reward is worse than a strict one.

The container is configured hostile-by-default:

* ``network_mode=none`` during the test phase - no exfiltration, no flaky
  network tests. The dependency-install phase is the one exception and runs as a
  separate, earlier container with network on, after which the filesystem is
  committed and reused.
* ``read_only`` root filesystem with a small tmpfs at /tmp and a writable bind
  only for the checkout.
* ``cap_drop=ALL``, ``security_opt=no-new-privileges``, non-root ``uid 1000``.
* memory, CPU quota and pids-limit set, so a fork bomb or memory hog kills the
  container rather than the host.
* a wall-clock timeout enforced by us, because a hung test would otherwise pin
  a worker slot forever.

The Docker socket is deliberately *not* mounted into the sandbox container -
only the bot process talks to it.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import shlex
import tarfile
import time
import uuid
from pathlib import Path

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.containers import Container

from autofix.config import Settings
from autofix.logging_conf import get_logger
from autofix.models import TestOutcome
from autofix.sandbox.detect import Toolchain
from autofix.sandbox.parse import parse_test_output

log = get_logger(__name__)

_WORKDIR = "/workspace"


class SandboxError(RuntimeError):
    pass


class DockerSandbox:
    """One instance per run; owns a prepared image and disposable containers."""

    def __init__(self, settings: Settings, toolchain: Toolchain, repo_root: Path) -> None:
        self._s = settings
        self._tc = toolchain
        self._root = repo_root
        # An explicit DOCKER_HOST wins; otherwise fall back to environment
        # detection, which resolves the Unix socket on Linux/macOS and the
        # named pipe on Windows.
        self._client = (
            docker.DockerClient(base_url=settings.docker_host, timeout=120)
            if settings.docker_host
            else docker.from_env(timeout=120)
        )
        self._prepared_image: str | None = None
        self._run_tag = f"autofix-prepared:{uuid.uuid4().hex[:12]}"

    # --- lifecycle --------------------------------------------------------

    async def __aenter__(self) -> DockerSandbox:
        await asyncio.to_thread(self._ensure_base_image)
        return self

    async def __aexit__(self, *_: object) -> None:
        await asyncio.to_thread(self._cleanup)

    def _ensure_base_image(self) -> None:
        try:
            self._client.images.get(self._tc.image)
        except ImageNotFound as exc:
            raise SandboxError(
                f"Sandbox image {self._tc.image!r} is missing. "
                f"Build it with: make sandbox-images"
            ) from exc
        except DockerException as exc:
            raise SandboxError(f"Cannot reach Docker daemon: {exc}") from exc

    def _cleanup(self) -> None:
        if self._prepared_image:
            with contextlib.suppress(NotFound, DockerException):
                self._client.images.remove(self._prepared_image, force=True)
        with contextlib.suppress(DockerException):
            self._client.close()

    # --- dependency install ----------------------------------------------

    async def prepare(self) -> TestOutcome:
        """Install dependencies once, then commit the result as a warm image.

        Doing this per-iteration would multiply a 4-minute `npm ci` by the
        repair-loop count. Committing once means iterations 2 and 3 are pure
        test runs.
        """
        if not self._tc.install_cmd:
            self._prepared_image = self._tc.image
            return TestOutcome(True, 0, 0.0, "no install step required", "")

        outcome = await self._run_in_container(
            image=self._tc.image,
            command=self._tc.install_cmd,
            timeout=self._s.sandbox_install_timeout_seconds,
            network="bridge" if self._s.sandbox_allow_install_network else "none",
            read_only=False,
            commit_tag=self._run_tag,
        )
        if outcome.passed:
            self._prepared_image = self._run_tag
        return outcome

    # --- test execution ---------------------------------------------------

    async def run_tests(
        self, command: list[str] | None = None, timeout: int | None = None
    ) -> TestOutcome:
        image = self._prepared_image or self._tc.image
        cmd = command or self._tc.test_cmd
        outcome = await self._run_in_container(
            image=image,
            command=cmd,
            timeout=timeout or self._s.sandbox_timeout_seconds,
            network=self._s.sandbox_network_mode,
            read_only=False,
        )
        parsed = parse_test_output(self._tc.language, outcome.stdout, outcome.stderr)
        outcome.failed_tests = parsed.failed_tests
        outcome.passed_count = parsed.passed_count
        outcome.failed_count = parsed.failed_count
        return outcome

    async def run_single_test(self, target: str) -> TestOutcome:
        template = self._tc.single_test_cmd_template
        if template is None:
            return await self.run_tests()
        cmd = [part.replace("{target}", target) for part in template]
        return await self.run_tests(command=cmd)

    # --- core -------------------------------------------------------------

    async def _run_in_container(
        self,
        image: str,
        command: list[str],
        timeout: int,
        network: str,
        read_only: bool,
        commit_tag: str | None = None,
    ) -> TestOutcome:
        return await asyncio.to_thread(
            self._run_blocking, image, command, timeout, network, read_only, commit_tag
        )

    def _run_blocking(
        self,
        image: str,
        command: list[str],
        timeout: int,
        network: str,
        read_only: bool,
        commit_tag: str | None,
    ) -> TestOutcome:
        started = time.monotonic()
        container: Container | None = None
        try:
            container = self._client.containers.create(
                image=image,
                command=["sh", "-lc", shlex.join(command)],
                working_dir=_WORKDIR,
                network_mode=network,
                mem_limit=f"{self._s.sandbox_memory_mb}m",
                memswap_limit=f"{self._s.sandbox_memory_mb}m",  # disallow swap escape
                nano_cpus=int(self._s.sandbox_cpu_quota * 1e9),
                pids_limit=self._s.sandbox_pids_limit,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                user="1000:1000",
                read_only=read_only,
                tmpfs={"/tmp": "rw,noexec,nosuid,size=512m"},  # noqa: S108 - container-internal
                environment={
                    "HOME": "/tmp",  # noqa: S108 - container-internal
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "CI": "true",
                    "NO_COLOR": "1",
                },
                detach=True,
                auto_remove=False,
                labels={"autofix": "sandbox"},
            )
            self._copy_repo_into(container)
            container.start()

            timed_out = False
            try:
                result = container.wait(timeout=timeout)
                exit_code = int(result.get("StatusCode", 1))
            except Exception:  # docker raises ReadTimeout on wait() timeout
                timed_out = True
                exit_code = 124
                with contextlib.suppress(DockerException):
                    container.kill()

            stdout = _decode(container.logs(stdout=True, stderr=False))
            stderr = _decode(container.logs(stdout=False, stderr=True))

            if commit_tag and exit_code == 0 and not timed_out:
                container.commit(repository=commit_tag.split(":")[0],
                                 tag=commit_tag.split(":")[1])

            return TestOutcome(
                passed=exit_code == 0 and not timed_out,
                exit_code=exit_code,
                duration_seconds=round(time.monotonic() - started, 2),
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                error="wall-clock timeout" if timed_out else None,
            )
        except DockerException as exc:
            return TestOutcome(
                passed=False,
                exit_code=-1,
                duration_seconds=round(time.monotonic() - started, 2),
                stdout="",
                stderr=str(exc),
                error=f"docker error: {exc}",
            )
        finally:
            if container is not None:
                with contextlib.suppress(DockerException):
                    container.remove(force=True, v=True)

    def _copy_repo_into(self, container: Container) -> None:
        """Stream the checkout in as a tar rather than bind-mounting it.

        A bind mount would let container-side code write back into the host
        checkout. Copying in means the container's filesystem is genuinely
        disposable, and we read results back out through the diff we already
        hold rather than trusting anything the container wrote.
        """
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(self._root, arcname=".", filter=_tar_filter)
        buf.seek(0)
        container.put_archive(_WORKDIR, buf.getvalue())


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    skip = (
        "/.git/", "/node_modules/", "/.venv/", "/venv/", "/__pycache__/",
        "/.mypy_cache/", "/.pytest_cache/", "/.tox/", "/dist/", "/build/",
    )
    normalised = "/" + info.name.replace("\\", "/").lstrip("./") + "/"
    if any(part in normalised for part in skip):
        return None
    info.uid, info.gid = 1000, 1000
    info.uname, info.gname = "sandbox", "sandbox"
    return info


def _decode(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw

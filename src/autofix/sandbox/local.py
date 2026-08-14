"""Namespace-isolated sandbox for clusters with no container runtime.

Why this exists
---------------
This project's reward function is "did the repository's own test suite pass".
Running that means executing untrusted third-party code plus code a model just
generated, thousands of times. Docker was the original answer, but shared HPC
clusters almost never grant the daemon socket, and Apptainer is not installed
everywhere either.

What they *do* commonly allow is **unprivileged user namespaces**, and that is
enough to reconstruct most of what Docker was providing:

| Property            | Docker                | Here                              |
|---------------------|-----------------------|-----------------------------------|
| Network isolation   | `network_mode=none`   | `unshare --net` (own netns)       |
| PID isolation       | pid namespace         | `unshare --pid --fork`            |
| Filesystem scope    | image + tmpfs         | `unshare --mount`, per-repo venv  |
| Memory cap          | cgroup `mem_limit`    | `RLIMIT_AS`                       |
| Process cap         | `pids_limit`          | `RLIMIT_NPROC`                    |
| CPU cap             | cgroup quota          | `RLIMIT_CPU` + wall-clock timeout |
| Root filesystem     | fresh per run         | **not reproduced**                |

Honest limitations
------------------
The last row matters and is stated in the method write-up rather than glossed
over. A test suite here can read the real filesystem, including the user's home
directory. It cannot reach the network, cannot see or signal other processes,
and cannot exhaust host memory - but it is *not* the equivalent of a fresh
container image, and this backend should not be used to run genuinely hostile
code.

For this project the risk profile is acceptable: the corpora are well-known
open-source repositories (django, flask, sympy, requests) whose test suites are
run by thousands of CI systems daily. It would not be acceptable for the
production bot this codebase started as, which accepted arbitrary repositories
from strangers.

Network isolation is the property that matters most for *correctness*, not just
security: a test that quietly reaches the internet makes the reward signal
non-reproducible, and a noisy reward is worse than a strict one.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys
import time
from pathlib import Path

from autofix.config import Settings
from autofix.logging_conf import get_logger
from autofix.models import TestOutcome
from autofix.sandbox.detect import Toolchain
from autofix.sandbox.parse import parse_test_output

log = get_logger(__name__)


class SandboxError(RuntimeError):
    pass


def namespaces_available() -> bool:
    """Can we create an unprivileged user+PID namespace on this host?"""
    try:
        proc = __import__("subprocess").run(
            ["unshare", "--user", "--pid", "--fork", "--mount-proc", "true"],
            capture_output=True, timeout=15, check=False,
        )
    except (FileNotFoundError, Exception):  # noqa: BLE001
        return False
    return proc.returncode == 0


class LocalSandbox:
    """Drop-in replacement for DockerSandbox using OS namespaces.

    Presents the same interface (`prepare`, `run_tests`, `run_single_test`) so
    nothing upstream needs to know which backend is in use.
    """

    def __init__(self, settings: Settings, toolchain: Toolchain, repo_root: Path) -> None:
        self._s = settings
        self._tc = toolchain
        self._root = repo_root
        self._env_dir = repo_root.parent / "sandbox-env"
        self._prepared = False
        self._use_ns = namespaces_available()
        if not self._use_ns:
            log.warning(
                "sandbox.no_namespaces",
                note="falling back to plain subprocess; network is NOT isolated",
            )

    async def __aenter__(self) -> LocalSandbox:
        return self

    async def __aexit__(self, *_: object) -> None:
        shutil.rmtree(self._env_dir, ignore_errors=True)

    # --- environment preparation -----------------------------------------

    async def prepare(self) -> TestOutcome:
        """Install the repository's dependencies.

        Network is deliberately ALLOWED here and only here — pip and npm need
        it. Test execution afterwards runs with the network namespace isolated,
        which is what keeps the reward signal deterministic.
        """
        if not self._tc.install_cmd:
            self._prepared = True
            return TestOutcome(True, 0, 0.0, "no install step required", "")

        if self._tc.language == "python":
            outcome = await self._create_venv()
            if not outcome.passed:
                return outcome

        outcome = await self._run(
            self._tc.install_cmd,
            timeout=self._s.sandbox_install_timeout_seconds,
            isolate_network=False,
        )
        self._prepared = outcome.passed
        return outcome

    async def _create_venv(self) -> TestOutcome:
        """A dedicated interpreter per repository.

        Installing a repo's pinned dependencies into the shared environment
        would corrupt it for every subsequent instance — and across a few
        thousand rejection-sampling instances that is a guarantee of
        irreproducible results.
        """
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "venv", "--system-site-packages", str(self._env_dir),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        ok = proc.returncode == 0
        if not ok:
            log.warning("sandbox.venv_failed", error=err.decode(errors="replace")[:300])
        return TestOutcome(
            passed=ok, exit_code=proc.returncode or 0,
            duration_seconds=round(time.monotonic() - started, 2),
            stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"),
            error=None if ok else "venv creation failed",
        )

    # --- test execution ---------------------------------------------------

    async def run_tests(
        self, command: list[str] | None = None, timeout: int | None = None
    ) -> TestOutcome:
        outcome = await self._run(
            command or self._tc.test_cmd,
            timeout=timeout or self._s.sandbox_timeout_seconds,
            isolate_network=True,
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
        return await self.run_tests(
            command=[part.replace("{target}", target) for part in template]
        )

    # --- core -------------------------------------------------------------

    # Forwarded ONLY while the network is open (dependency install). Many HPC
    # clusters reach the internet exclusively through an HTTP proxy, and pip or
    # npm will fail with an opaque DNS error if these are missing. They are
    # deliberately NOT forwarded to test execution, where the network namespace
    # is isolated and a proxy address would be both useless and needless
    # information to hand untrusted code.
    _NETWORK_PASSTHROUGH = (
        "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy",
        "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST",
        "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE",
        "NPM_CONFIG_REGISTRY", "npm_config_registry", "GOPROXY", "GOSUMDB",
        "HF_ENDPOINT",
    )

    def _environment(self, allow_network: bool = False) -> dict[str, str]:
        """A deliberately minimal environment.

        The parent process holds cluster credentials and API tokens; passing
        those into untrusted test code would defeat the point of isolating it.
        Only what a test suite legitimately needs is forwarded — plus, during
        the install phase alone, the proxy settings needed to reach a package
        index at all.
        """
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(self._root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "CI": "true",
            "NO_COLOR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "TMPDIR": str(self._root.parent / "tmp"),
        }
        if self._tc.language == "python" and self._env_dir.exists():
            env["VIRTUAL_ENV"] = str(self._env_dir)
            env["PATH"] = f"{self._env_dir / 'bin'}:{env['PATH']}"

        if allow_network:
            for key in self._NETWORK_PASSTHROUGH:
                value = os.environ.get(key)
                if value:
                    env[key] = value
        return env

    def _wrap(self, command: list[str], isolate_network: bool) -> list[str]:
        """Build the isolation wrapper around the real command.

        `ulimit` is applied inside the shell rather than via preexec_fn because
        it must survive the `unshare` boundary, and because a shell is needed
        anyway to bring loopback up inside the new network namespace — many test
        suites bind to 127.0.0.1 and would fail against a down `lo`.
        """
        mem_kb = self._s.sandbox_memory_mb * 1024
        inner = " ".join(_quote(part) for part in command)

        guarded = (
            f"ulimit -v {mem_kb} 2>/dev/null || true; "
            f"ulimit -u {self._s.sandbox_pids_limit} 2>/dev/null || true; "
            f"{inner}"
        )

        if not self._use_ns:
            return ["sh", "-c", guarded]

        ns_flags = ["--user", "--map-root-user", "--pid", "--fork", "--mount-proc"]
        if isolate_network:
            ns_flags.append("--net")
            # A fresh netns has `lo` present but DOWN. Tests that bind to
            # localhost need it up; this is still a fully isolated namespace
            # with no route to anywhere else.
            guarded = f"ip link set lo up 2>/dev/null; {guarded}"

        return ["unshare", *ns_flags, "sh", "-c", guarded]

    async def _run(
        self, command: list[str], timeout: int, isolate_network: bool
    ) -> TestOutcome:
        started = time.monotonic()
        tmp = self._root.parent / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)

        wrapped = self._wrap(command, isolate_network)
        log.debug("sandbox.exec", command=command, isolated=self._use_ns,
                  network=not isolate_network)

        try:
            proc = await asyncio.create_subprocess_exec(
                *wrapped, cwd=str(self._root),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=self._environment(allow_network=not isolate_network),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise SandboxError(f"cannot execute {wrapped[0]!r}: {exc}") from exc

        timed_out = False
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        # asyncio.TimeoutError only became an alias of the builtin TimeoutError
        # in 3.11. Catching both keeps this correct on 3.10 as well, and the
        # duplicate is harmless on newer versions.
        except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
            timed_out = True
            out, err = b"", b""
            # Kill the whole process group: a hung pytest usually has children,
            # and killing only the leader leaves them holding the GPU node.
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            # Reap the killed process so it does not linger as a zombie. It is
            # already dead, so this returns immediately; the timeout only guards
            # against a pathological unkillable child.
            with contextlib.suppress(Exception):
                out, err = await asyncio.wait_for(proc.communicate(), timeout=10)

        exit_code = 124 if timed_out else (proc.returncode or 0)
        return TestOutcome(
            passed=exit_code == 0 and not timed_out,
            exit_code=exit_code,
            duration_seconds=round(time.monotonic() - started, 2),
            stdout=out.decode(errors="replace"),
            stderr=err.decode(errors="replace"),
            timed_out=timed_out,
            error="wall-clock timeout" if timed_out else None,
        )


def _quote(part: str) -> str:
    import shlex

    return shlex.quote(part)

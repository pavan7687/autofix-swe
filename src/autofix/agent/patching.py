"""Applying and reverting model-generated diffs.

`git apply` is used rather than a pure-Python patcher, for three reasons:

* it is the same implementation that will judge the PR later, so "applies here"
  means "applies there";
* ``--check`` gives us a dry run, letting us reject a bad diff before touching
  the working tree;
* ``-3`` (three-way) recovers from the single most common LLM diff defect -
  slightly wrong ``@@`` line numbers with correct context - without us having to
  re-implement fuzzy hunk matching.

Every apply is verified after the fact by re-reading `git diff`, so what we
commit is what git actually did, never what the model claimed it would do.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from autofix.logging_conf import get_logger

log = get_logger(__name__)


class GitError(RuntimeError):
    pass


@dataclass(slots=True)
class ApplyResult:
    ok: bool
    message: str
    normalised_diff: str = ""


async def run_git(
    repo: Path, *args: str, stdin: str | None = None, timeout: int = 120
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(repo),
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true", "PATH": "/usr/bin:/bin"},
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(stdin.encode() if stdin is not None else None),
            timeout=timeout,
        )
    except TimeoutError as exc:
        proc.kill()
        raise GitError(f"git {' '.join(args)} timed out") from exc
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def apply_patch(repo: Path, diff: str) -> ApplyResult:
    """Dry-run then apply. Returns the diff as git actually recorded it."""
    if not diff.strip():
        return ApplyResult(False, "empty patch")

    payload = diff if diff.endswith("\n") else diff + "\n"

    code, _, err = await run_git(repo, "apply", "--check", "-p1", "-", stdin=payload)
    strategy = ["apply", "-p1", "--whitespace=nowarn", "-"]
    if code != 0:
        # Retry with three-way merge: tolerates stale hunk offsets.
        code3, _, err3 = await run_git(
            repo, "apply", "--check", "-3", "-p1", "-", stdin=payload
        )
        if code3 != 0:
            return ApplyResult(
                False, f"patch does not apply: {err.strip() or err3.strip()}"
            )
        strategy = ["apply", "-3", "-p1", "--whitespace=nowarn", "-"]

    code, _, err = await run_git(repo, *strategy, stdin=payload)
    if code != 0:
        return ApplyResult(False, f"git apply failed: {err.strip()}")

    _, actual, _ = await run_git(repo, "diff", "--no-color")
    if not actual.strip():
        return ApplyResult(False, "patch applied but produced no change")

    return ApplyResult(True, "applied", normalised_diff=actual)


async def reset_worktree(repo: Path) -> None:
    """Return the checkout to pristine HEAD between repair iterations."""
    await run_git(repo, "checkout", "--", ".")
    await run_git(repo, "clean", "-fd")


async def changed_files(repo: Path) -> list[str]:
    _, out, _ = await run_git(repo, "diff", "--name-only")
    return [line.strip() for line in out.splitlines() if line.strip()]


async def clone(
    clone_url: str, token: str, dest: Path, branch: str, depth: int = 1
) -> None:
    """Shallow-clone over HTTPS with the installation token.

    The token is injected into the URL rather than a config file so it never
    lands on disk in `.git/config`; we rewrite the remote immediately after.
    An empty token clones anonymously, which is what the CLI does against a
    public repository.
    """
    authed = _authed_url(clone_url, token)
    proc = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth", str(depth), "--branch", branch,
        "--single-branch", authed, str(dest),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise GitError(f"clone failed: {err.decode(errors='replace')[:400]}")

    await run_git(dest, "remote", "set-url", "origin", clone_url)
    await run_git(dest, "config", "user.name", "autofix-bot[bot]")
    await run_git(
        dest, "config", "user.email",
        "autofix-bot[bot]@users.noreply.github.com",
    )


async def commit_and_push(
    repo: Path, branch: str, message: str, clone_url: str, token: str
) -> str:
    await run_git(repo, "checkout", "-b", branch)
    await run_git(repo, "add", "-A")

    code, _, err = await run_git(repo, "commit", "-m", message, "--no-verify")
    if code != 0:
        raise GitError(f"commit failed: {err.strip()}")

    _, sha, _ = await run_git(repo, "rev-parse", "HEAD")

    authed = _authed_url(clone_url, token)
    code, _, err = await run_git(
        repo, "push", authed, f"{branch}:{branch}", timeout=180
    )
    if code != 0:
        raise GitError(f"push failed: {_redact(err, token)}")

    return sha.strip()


def _authed_url(clone_url: str, token: str) -> str:
    if not token:
        return clone_url
    return clone_url.replace("https://", f"https://x-access-token:{token}@")


def _redact(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text

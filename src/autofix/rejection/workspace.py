"""Per-instance repository checkout.

Rejection sampling and evaluation both need the target repository at an exact
commit, in a directory nothing else touches. This context manager owns that
lifecycle: clone at `base_commit`, apply the instance's `test_patch` so the
gating tests exist, hand back the path, delete it afterwards.

Applying `test_patch` before sampling is essential and easy to get wrong. The
benchmark's failing tests live in that patch, not in the repository at
`base_commit` — without it the tests the fix is graded against simply do not
exist, and every candidate would be scored against a suite that cannot fail.
The model never sees the test patch; only the grader does.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from autofix.agent.patching import GitError, apply_patch, run_git
from autofix.config import Settings
from autofix.logging_conf import get_logger
from autofix.models import Instance

log = get_logger(__name__)

_GITHUB = "https://github.com"


class InstanceWorkspace:
    """`async with` a clean checkout of one instance's repository."""

    def __init__(self, settings: Settings, instance: Instance) -> None:
        self._s = settings
        self._inst = instance
        self._root = settings.workspace_root / f"{instance.instance_id[:60]}-{uuid.uuid4().hex[:8]}"
        self.repo_dir: Path = self._root / "repo"
        self.error: str | None = None

    async def __aenter__(self) -> InstanceWorkspace:
        try:
            await self._checkout()
        except (GitError, OSError) as exc:
            self.error = f"checkout failed: {exc}"
            log.warning("workspace.checkout_failed", instance=self._inst.instance_id,
                        error=str(exc)[:200])
        return self

    async def __aexit__(self, *_: object) -> None:
        shutil.rmtree(self._root, ignore_errors=True)

    async def _checkout(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        url = f"{_GITHUB}/{self._inst.repo}.git"

        # Full clone: a shallow one cannot check out an arbitrary historical
        # commit, and benchmark instances are pinned to old SHAs.
        code, _, err = await run_git(
            self._root, "clone", "--quiet", url, str(self.repo_dir), timeout=600
        )
        if code != 0:
            raise GitError(f"clone {self._inst.repo}: {err.strip()[:300]}")

        code, _, err = await run_git(
            self.repo_dir, "checkout", "--quiet", "--force", self._inst.base_commit
        )
        if code != 0:
            raise GitError(f"checkout {self._inst.base_commit[:12]}: {err.strip()[:200]}")

        await run_git(self.repo_dir, "config", "user.email", "autofix@local")
        await run_git(self.repo_dir, "config", "user.name", "autofix")

        if self._inst.test_patch.strip():
            applied = await apply_patch(self.repo_dir, self._inst.test_patch)
            if not applied.ok:
                raise GitError(f"test_patch would not apply: {applied.message[:200]}")
            # Commit it so `git checkout -- .` between candidates does not
            # revert the gating tests along with the candidate patch.
            await run_git(self.repo_dir, "add", "-A")
            await run_git(self.repo_dir, "commit", "--quiet", "-m", "tests", "--no-verify")

        log.debug("workspace.ready", instance=self._inst.instance_id)

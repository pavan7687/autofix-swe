"""Execution-based verification: the reward function.

A candidate patch is correct if and only if, inside a clean container:

  * every test in FAIL_TO_PASS now passes, and
  * every test in PASS_TO_PASS still passes.

That definition is what makes this project reinforcement learning from
*verifiable* feedback rather than from a learned preference model. There is no
reward model to exploit — the tests either pass or they do not, and the model
cannot argue with a non-zero exit code.

The cheap filters run first and in strict order, because each rejects
candidates for a fraction of the cost of the one after it:

    parse (µs) -> scope check (ms) -> git apply (10ms) -> test suite (30-300s)

Roughly 40-60% of samples from an early-checkpoint model die before the
expensive step, which is the difference between a sampling run that finishes
overnight and one that does not.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from autofix.agent.patching import apply_patch, reset_worktree
from autofix.agent.responses import extract_diff
from autofix.config import Settings
from autofix.guardrails.scope import ScopeGuard
from autofix.logging_conf import get_logger
from autofix.models import Candidate, Instance, TestOutcome
from autofix.sandbox.detect import Toolchain
from autofix.sandbox.runner import DockerSandbox

log = get_logger(__name__)


class CandidateVerifier:
    """Verifies sampled patches against one instance's repository checkout."""

    def __init__(
        self,
        settings: Settings,
        instance: Instance,
        repo_dir: Path,
        sandbox: DockerSandbox,
        toolchain: Toolchain,
    ) -> None:
        self._s = settings
        self._inst = instance
        self._repo = repo_dir
        self._sandbox = sandbox
        self._toolchain = toolchain
        self._scope = ScopeGuard(settings)
        self._baseline: TestOutcome | None = None

    async def establish_baseline(self) -> TestOutcome:
        """Run the suite on unpatched code.

        Without this, "the tests pass" is unfalsifiable — they may have passed
        already. The baseline also catches instances whose environment is
        simply broken, which get discarded rather than scored as failures.
        """
        await reset_worktree(self._repo)
        self._baseline = await self._sandbox.run_tests()
        return self._baseline

    async def verify(self, candidate: Candidate) -> Candidate:
        # 1. Parse. Free.
        diff = extract_diff(candidate.raw_output)
        if diff is None:
            candidate.rejection = "no valid unified diff in model output"
            return candidate
        candidate.diff, candidate.parsed = diff, True

        # 2. Scope. Microseconds. Also rejects test-file edits, which would
        #    otherwise let the model "fix" a bug by deleting the test.
        report = self._scope.inspect(diff)
        if not report.ok:
            candidate.rejection = "scope: " + "; ".join(report.violations[:3])
            return candidate
        candidate.scope_ok = True

        # 3. Apply. Milliseconds.
        await reset_worktree(self._repo)
        applied = await apply_patch(self._repo, diff)
        if not applied.ok:
            candidate.rejection = f"apply failed: {applied.message}"
            return candidate
        candidate.applied = True
        candidate.diff = applied.normalised_diff

        # 4. Execute. Seconds to minutes — the reason for the ordering above.
        outcome = await self._sandbox.run_tests()
        candidate.outcome = outcome
        candidate.resolved = self._is_resolved(outcome)
        if not candidate.resolved:
            candidate.rejection = self._explain(outcome)

        await reset_worktree(self._repo)
        return candidate

    def _is_resolved(self, outcome: TestOutcome) -> bool:
        failed = set(outcome.failed_tests)

        # FAIL_TO_PASS: the tests the fix must turn green.
        if self._inst.fail_to_pass:
            if any(_matches(t, failed) for t in self._inst.fail_to_pass):
                return False
        elif not outcome.passed:
            # No explicit test list (mined instances): fall back to a green suite.
            return False

        # PASS_TO_PASS: nothing previously green may have broken.
        baseline_failed = set(self._baseline.failed_tests) if self._baseline else set()
        for test in self._inst.pass_to_pass:
            if _matches(test, failed) and not _matches(test, baseline_failed):
                return False
        return True

    def _explain(self, outcome: TestOutcome) -> str:
        if outcome.timed_out:
            return "test suite timed out"
        still_failing = [
            t for t in self._inst.fail_to_pass
            if _matches(t, set(outcome.failed_tests))
        ]
        if still_failing:
            return f"fail_to_pass still failing: {still_failing[:3]}"
        broken = sorted(
            set(outcome.failed_tests)
            - set(self._baseline.failed_tests if self._baseline else [])
        )
        if broken:
            return f"regression, newly failing: {broken[:3]}"
        return f"tests failed (exit {outcome.exit_code})"


def _matches(test_name: str, failed: set[str]) -> bool:
    """Fuzzy test-identity match across reporter formats.

    pytest prints `path/to/test_x.py::TestC::test_m`, but the FAIL_TO_PASS
    field may hold any suffix of that. Comparing on the last two segments is
    the tolerant-but-not-sloppy middle ground.
    """
    if test_name in failed:
        return True
    tail = "::".join(test_name.split("::")[-2:])
    return any(f.endswith(tail) or tail.endswith("::".join(f.split("::")[-2:]))
               for f in failed)


async def verify_batch(
    verifier: CandidateVerifier, candidates: list[Candidate], concurrency: int = 1
) -> list[Candidate]:
    """Verify candidates for one instance.

    Concurrency is 1 by default and that is not an oversight: all candidates
    for an instance share a single git worktree, so applying two patches at
    once would corrupt both. Parallelism belongs at the instance level, where
    each worker owns its own checkout and container.
    """
    if concurrency <= 1:
        return [await verifier.verify(c) for c in candidates]

    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(candidate: Candidate) -> Candidate:
        async with semaphore:
            return await verifier.verify(candidate)

    return list(await asyncio.gather(*(guarded(c) for c in candidates)))

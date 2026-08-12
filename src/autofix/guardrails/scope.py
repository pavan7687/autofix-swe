"""Scope enforcement on a generated patch.

This is the last line of defence between an LLM and someone else's repository.
It runs on the *parsed diff*, before anything touches the filesystem, and it is
deliberately paranoid:

* path traversal and absolute paths are rejected outright;
* protected globs (CI config, lockfiles, packaging metadata, secrets) are
  untouchable, because a change there is either unreviewable or dangerous;
* file/line ceilings turn "the model decided to refactor the module" into a
  clean rejection instead of a 4,000-line PR;
* new-file creation is allowed but capped, and deletions are refused entirely;
* **test files are immutable.** The bot is verified against the repository's own
  test cases, so a patch that edits, weakens or deletes a test is rewriting the
  specification it is being judged against. Detection lives here rather than in
  the pipeline so that every scope rule is auditable in one file.

A rejection here is a *successful* outcome - it means the guardrail worked.
"""
from __future__ import annotations

import fnmatch
import posixpath
from dataclasses import dataclass, field

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from autofix.config import Settings


@dataclass(slots=True)
class ScopeReport:
    ok: bool
    files: list[str] = field(default_factory=list)
    added_lines: int = 0
    removed_lines: int = 0
    violations: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        return self.added_lines + self.removed_lines


_TEST_SUFFIXES = (
    "_test.py", "_test.go", ".test.js", ".test.jsx", ".test.ts", ".test.tsx",
    ".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx", "_spec.rb", "Test.java",
)


def is_test_path(path: str) -> bool:
    """True if a path is part of a repository's test suite."""
    lower = path.lower()
    padded = f"/{lower}"
    return (
        "/tests/" in padded
        or "/test/" in padded
        or "/spec/" in padded
        or "/__tests__/" in padded
        or "/test_" in padded
        or lower.endswith(tuple(s.lower() for s in _TEST_SUFFIXES))
    )


class ScopeGuard:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._protected = settings.protected_paths

    def _is_protected(self, path: str) -> bool:
        """Match a repo-relative path against the protected globs.

        `fnmatch` has no concept of `**`, and its `*` happily crosses `/`. That
        means the natural-looking pattern `**/*.lock` fails to match a
        root-level `poetry.lock`, because the pattern still demands a literal
        slash. Silently under-matching a *protection* rule is the dangerous
        direction of failure, so each pattern is tried three ways: as written,
        with a leading `**/` stripped, and against the basename alone.
        """
        name = path.rsplit("/", 1)[-1]
        for pattern in self._protected:
            bare = pattern[3:] if pattern.startswith("**/") else pattern
            if (
                fnmatch.fnmatch(path, pattern)
                or fnmatch.fnmatch(path, bare)
                or fnmatch.fnmatch(name, bare)
            ):
                return True
        return False

    @staticmethod
    def _is_unsafe_path(path: str) -> bool:
        if path.startswith(("/", "\\")) or ":" in path.split("/")[0]:
            return True
        normalised = posixpath.normpath(path)
        return normalised.startswith("..") or normalised == "."

    def inspect(self, diff_text: str) -> ScopeReport:
        report = ScopeReport(ok=True)

        if not diff_text.strip():
            report.ok = False
            report.violations.append("patch is empty")
            return report

        try:
            patch = PatchSet(diff_text)
        except (UnidiffParseError, Exception) as exc:  # noqa: BLE001
            report.ok = False
            report.violations.append(f"patch is not a valid unified diff: {exc}")
            return report

        if not patch:
            report.ok = False
            report.violations.append("patch contains no file sections")
            return report

        for pfile in patch:
            path = pfile.path
            report.files.append(path)

            if self._is_unsafe_path(path):
                report.violations.append(f"unsafe path outside repository: {path}")
            if self._is_protected(path):
                report.violations.append(f"protected path may not be modified: {path}")
            if pfile.is_removed_file:
                report.violations.append(f"file deletion is not permitted: {path}")
            if pfile.is_rename:
                report.violations.append(f"file rename is not permitted: {path}")
            if is_test_path(path):
                report.test_files.append(path)
                report.violations.append(
                    f"test files are immutable and may not be patched: {path}"
                )

            file_lines = pfile.added + pfile.removed
            if file_lines > self._s.max_single_file_lines:
                report.violations.append(
                    f"{path}: {file_lines} changed lines exceeds per-file cap "
                    f"{self._s.max_single_file_lines}"
                )

            report.added_lines += pfile.added
            report.removed_lines += pfile.removed

        unique_files = sorted(set(report.files))
        report.files = unique_files

        if len(unique_files) > self._s.max_files_changed:
            report.violations.append(
                f"{len(unique_files)} files changed exceeds cap "
                f"{self._s.max_files_changed}"
            )
        if report.total_lines > self._s.max_lines_changed:
            report.violations.append(
                f"{report.total_lines} changed lines exceeds cap "
                f"{self._s.max_lines_changed}"
            )

        report.ok = not report.violations
        return report

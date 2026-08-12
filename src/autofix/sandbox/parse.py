"""Test-output parsing.

We need three facts from a test run: did it pass, how many tests failed, and
which ones. Exit code alone is not enough - a collection error and a genuine
assertion failure both exit non-zero but demand different next moves from the
agent, and the failing test names are the highest-signal thing we can put in the
repair prompt.

Parsing is intentionally regex-based over stdout rather than machine-readable
report plugins: we cannot install `pytest-json-report` into someone else's
repository without changing their dependency set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedTests:
    passed_count: int = 0
    failed_count: int = 0
    failed_tests: list[str] = field(default_factory=list)
    collection_error: bool = False


_PYTEST_SUMMARY = re.compile(
    r"^(?:FAILED|ERROR)\s+(?P<test>[\w./\\:\-\[\]]+)", re.MULTILINE
)
_PYTEST_COUNTS = re.compile(
    r"(?:(?P<failed>\d+)\s+failed)|(?:(?P<passed>\d+)\s+passed)|(?:(?P<errors>\d+)\s+error)"
)
_UNITTEST_FAIL = re.compile(r"^(?:FAIL|ERROR):\s+(?P<test>\S+)", re.MULTILINE)
_JEST_FAIL = re.compile(r"^\s*●\s+(?P<test>.+?)$", re.MULTILINE)
_JEST_COUNTS = re.compile(
    r"Tests:\s+(?:(?P<failed>\d+)\s+failed,\s*)?(?:\d+\s+skipped,\s*)?(?P<passed>\d+)\s+passed"
)
_GO_FAIL = re.compile(r"^\s*---\s+FAIL:\s+(?P<test>\S+)", re.MULTILINE)


def _parse_python(text: str) -> ParsedTests:
    result = ParsedTests()
    result.failed_tests = list(dict.fromkeys(_PYTEST_SUMMARY.findall(text)))
    if not result.failed_tests:
        result.failed_tests = list(dict.fromkeys(_UNITTEST_FAIL.findall(text)))

    for match in _PYTEST_COUNTS.finditer(text):
        if match.group("failed"):
            result.failed_count += int(match.group("failed"))
        if match.group("passed"):
            result.passed_count += int(match.group("passed"))
        if match.group("errors"):
            result.failed_count += int(match.group("errors"))

    if not result.failed_count:
        result.failed_count = len(result.failed_tests)

    result.collection_error = (
        "error during collection" in text.lower()
        or "ImportError while importing test module" in text
        or "ModuleNotFoundError" in text
    )
    return result


def _parse_node(text: str) -> ParsedTests:
    result = ParsedTests()
    result.failed_tests = [t.strip() for t in dict.fromkeys(_JEST_FAIL.findall(text))]
    if (counts := _JEST_COUNTS.search(text)) is not None:
        result.failed_count = int(counts.group("failed") or 0)
        result.passed_count = int(counts.group("passed") or 0)
    else:
        result.failed_count = len(result.failed_tests)
    return result


def _parse_go(text: str) -> ParsedTests:
    result = ParsedTests()
    result.failed_tests = list(dict.fromkeys(_GO_FAIL.findall(text)))
    result.failed_count = len(result.failed_tests)
    result.passed_count = text.count("--- PASS:")
    return result


_PARSERS = {"python": _parse_python, "node": _parse_node, "go": _parse_go}


def parse_test_output(language: str, stdout: str, stderr: str) -> ParsedTests:
    text = f"{stdout}\n{stderr}"
    parser = _PARSERS.get(language, _parse_python)
    return parser(text)

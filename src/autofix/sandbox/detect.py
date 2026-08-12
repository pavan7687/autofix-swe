"""Toolchain detection.

A production bot pointed at real repositories cannot assume `pytest`. It has to
look at what is actually checked in and derive: which base image, how to install
dependencies, and how to run tests.

Detection is evidence-ranked, not first-match: a repo containing both `tox.ini`
and `pyproject.toml` should prefer the pyproject path, and a repo with a
`Makefile` target named `test` is usually the most reliable signal of all
because it is what CI runs.

Everything returned here is a *proposal*. The runner verifies it by executing
the repository's existing suite before any patch is applied; if that baseline
does not run, the whole attempt is abandoned rather than guessed at.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Toolchain:
    language: str
    image: str
    install_cmd: list[str]
    test_cmd: list[str]
    single_test_cmd_template: list[str] | None
    confidence: float
    evidence: list[str]

    def describe(self) -> str:
        return (
            f"{self.language} | image `{self.image}` | "
            f"install `{' '.join(self.install_cmd) or '(none)'}` | "
            f"test `{' '.join(self.test_cmd)}`"
        )


_PY_IMAGE = "autofix/sandbox-python:3.11"
_NODE_IMAGE = "autofix/sandbox-node:20"
_GO_IMAGE = "autofix/sandbox-go:1.22"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _makefile_test_target(root: Path) -> bool:
    for name in ("Makefile", "makefile", "GNUmakefile"):
        content = _read(root / name)
        if content and re.search(r"^test:", content, re.MULTILINE):
            return True
    return False


def _detect_python(root: Path) -> Toolchain | None:
    evidence: list[str] = []
    install: list[str] = []

    pyproject = root / "pyproject.toml"
    setup_py = root / "setup.py"
    reqs = [p for p in ("requirements.txt", "requirements-dev.txt", "requirements/dev.txt")
            if (root / p).exists()]

    if pyproject.exists():
        evidence.append("pyproject.toml")
        content = _read(pyproject)
        extra = "[dev]" if "dev =" in content or '"dev"' in content else ""
        install = ["python", "-m", "pip", "install", "-e", f".{extra}" if extra else "."]
    elif setup_py.exists():
        evidence.append("setup.py")
        install = ["python", "-m", "pip", "install", "-e", "."]

    for req in reqs:
        evidence.append(req)
        install = ["sh", "-c", f"python -m pip install -r {req} && " +
                   (" ".join(install) if install else "true")]
        break

    has_pytest = (
        (root / "pytest.ini").exists()
        or "pytest" in _read(pyproject)
        or "pytest" in _read(root / "tox.ini")
        or "pytest" in _read(root / "setup.cfg")
        or any(root.glob("tests/**/test_*.py"))
        or any(root.glob("test_*.py"))
    )
    if not (evidence or has_pytest):
        return None

    if has_pytest:
        evidence.append("pytest signals")
        test_cmd = ["python", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
        single = ["python", "-m", "pytest", "-q", "--no-header", "{target}"]
    else:
        test_cmd = ["python", "-m", "unittest", "discover", "-v"]
        single = ["python", "-m", "unittest", "-v", "{target}"]

    return Toolchain(
        language="python",
        image=_PY_IMAGE,
        install_cmd=install,
        test_cmd=test_cmd,
        single_test_cmd_template=single,
        confidence=0.9 if has_pytest and evidence else 0.6,
        evidence=evidence,
    )


def _detect_node(root: Path) -> Toolchain | None:
    pkg_path = root / "package.json"
    if not pkg_path.exists():
        return None
    try:
        pkg = json.loads(_read(pkg_path) or "{}")
    except json.JSONDecodeError:
        return None

    evidence = ["package.json"]
    scripts = pkg.get("scripts", {})
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

    if (root / "pnpm-lock.yaml").exists():
        install, runner = ["pnpm", "install", "--frozen-lockfile"], "pnpm"
        evidence.append("pnpm-lock.yaml")
    elif (root / "yarn.lock").exists():
        install, runner = ["yarn", "install", "--frozen-lockfile"], "yarn"
        evidence.append("yarn.lock")
    else:
        install, runner = ["npm", "ci", "--no-audit", "--no-fund"], "npm"
        if not (root / "package-lock.json").exists():
            install = ["npm", "install", "--no-audit", "--no-fund"]

    if "test" in scripts:
        test_cmd = [runner, "run", "test"] if runner != "npm" else ["npm", "test", "--silent"]
        evidence.append("scripts.test")
    elif "vitest" in deps:
        test_cmd = ["npx", "vitest", "run"]
    elif "jest" in deps:
        test_cmd = ["npx", "jest", "--ci"]
    else:
        return None

    single = [*test_cmd, "--", "{target}"] if "jest" in deps or "vitest" in deps else None

    return Toolchain(
        language="node",
        image=_NODE_IMAGE,
        install_cmd=install,
        test_cmd=test_cmd,
        single_test_cmd_template=single,
        confidence=0.85,
        evidence=evidence,
    )


def _detect_go(root: Path) -> Toolchain | None:
    if not (root / "go.mod").exists():
        return None
    return Toolchain(
        language="go",
        image=_GO_IMAGE,
        install_cmd=["go", "mod", "download"],
        test_cmd=["go", "test", "./...", "-count=1"],
        single_test_cmd_template=["go", "test", "-run", "{target}", "./...", "-count=1"],
        confidence=0.9,
        evidence=["go.mod"],
    )


def detect(repo_root: Path) -> Toolchain | None:
    """Return the highest-confidence toolchain, or None if unsupported."""
    candidates = [
        detector(repo_root)
        for detector in (_detect_python, _detect_node, _detect_go)
    ]
    found = [c for c in candidates if c is not None]
    if not found:
        return None

    best = max(found, key=lambda c: c.confidence)
    if _makefile_test_target(repo_root):
        best.evidence.append("Makefile:test target present")
    return best

"""Domain types shared across data building, training, sampling and evaluation.

The central object is `Instance`: one (issue, repository state, gold patch)
triple. Everything downstream — a retrieval example, an editing example, a
rejection-sampling candidate, an eval task — is derived from it, so the schema
is defined once here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Instance(BaseModel):
    """One issue-resolution task, from any source dataset."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str = ""                      # gold source patch
    test_patch: str = ""                 # tests that gate the fix
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    environment_setup_commit: str = ""
    source: str = "unknown"
    gold_files: list[str] = Field(default_factory=list)

    @property
    def contamination_key(self) -> str:
        """Repo + commit identifies a benchmark task regardless of dataset.

        Both halves are lowercased. Git SHAs are conventionally lowercase but
        the public corpora are not consistent about it, and a case variant that
        slips past this check is a silent contamination leak.
        """
        return f"{self.repo.strip().lower()}@{self.base_commit.strip().lower()[:12]}"


class RetrievalExample(BaseModel):
    """Train the reranker: which of these candidate files hold the defect?"""

    instance_id: str
    repo: str
    problem_statement: str
    candidates: list[str]
    gold_files: list[str]
    bm25_hit: bool = False               # did BM25 surface any gold file at all
    bm25_rank_of_first_gold: int | None = None

    @property
    def is_trainable(self) -> bool:
        # If BM25 missed every gold file, the reranker cannot recover it and the
        # example teaches nothing but noise.
        return self.bm25_hit


class EditingExample(BaseModel):
    """Train the editor: given the issue and the buggy files, write the patch."""

    instance_id: str
    repo: str
    problem_statement: str
    file_contents: dict[str, str]
    patch: str
    reasoning: str = ""
    token_estimate: int = 0


@dataclass(slots=True)
class TestOutcome:
    """Result of one sandbox invocation. This is the reward signal."""

    passed: bool
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str
    timed_out: bool = False
    failed_tests: list[str] = field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    error: str | None = None

    def tail(self, limit: int = 6000) -> str:
        combined = (self.stdout or "") + ("\n" + self.stderr if self.stderr else "")
        return combined[-limit:]


@dataclass(slots=True)
class Candidate:
    """One sampled patch, and whether execution proved it correct."""

    instance_id: str
    sample_index: int
    raw_output: str
    diff: str = ""
    parsed: bool = False
    scope_ok: bool = False
    applied: bool = False
    resolved: bool = False               # fail_to_pass now pass, pass_to_pass intact
    outcome: TestOutcome | None = None
    rejection: str | None = None

    @property
    def reward(self) -> float:
        """Binary, execution-derived. No learned reward model, nothing to hack."""
        return 1.0 if self.resolved else 0.0


@dataclass(slots=True)
class CodeChunk:
    """A retrieved slice of a repository, with provenance for the prompt."""

    path: str
    start_line: int
    end_line: int
    text: str
    symbol: str | None = None
    score: float = 0.0

    def render(self) -> str:
        header = f"--- {self.path}:{self.start_line}-{self.end_line}"
        if self.symbol:
            header += f"  ({self.symbol})"
        return f"{header}\n{self.text}"

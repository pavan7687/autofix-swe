"""Held-out set protection.

If a single SWE-bench test instance leaks into training, every number this
project reports is void — and it is the first thing a reviewer or interviewer
will check. Contamination is silent: the loss curve looks healthy, eval looks
excellent, and the result is worthless.

Three independent filters, because each catches leaks the others miss:

1. **Exact instance id.** Trivial, catches the obvious overlap between public
   training corpora and the benchmarks (SWE-Gym and SWE-bench train both draw
   from the same repositories).
2. **repo@commit.** Different datasets assign different ids to the same
   underlying task. Repo plus base commit identifies the task itself.
3. **Repository-level.** The strictest option: drop every instance from any
   repository that appears in the eval set at all. Costs training data, but it
   is the only filter that survives the objection "your model memorised
   `django/django` internals from a neighbouring commit".

The default is repo+commit; repository-level is available via `--strict` and is
what should be reported in a writeup if the numbers are close.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from autofix.logging_conf import get_logger
from autofix.models import Instance

log = get_logger(__name__)


@dataclass
class ContaminationIndex:
    """Fingerprints of everything that must never appear in training."""

    instance_ids: set[str] = field(default_factory=set)
    repo_commits: set[str] = field(default_factory=set)
    repos: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)

    def add_instance(self, inst: Instance) -> None:
        self.instance_ids.add(inst.instance_id.strip().lower())
        self.repo_commits.add(inst.contamination_key)
        self.repos.add(inst.repo.strip().lower())

    def is_contaminated(self, inst: Instance, strict: bool = False) -> str | None:
        """Return the reason this instance must be dropped, or None.

        All comparisons are case-normalised: the corpora disagree on the casing
        of repository names and SHAs, and a leak that survives on a casing
        technicality invalidates every number downstream.
        """
        if inst.instance_id.strip().lower() in self.instance_ids:
            return "instance_id"
        if inst.contamination_key in self.repo_commits:
            return "repo_commit"
        if strict and inst.repo.strip().lower() in self.repos:
            return "repo"
        return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sources": self.sources,
                    "instance_ids": sorted(self.instance_ids),
                    "repo_commits": sorted(self.repo_commits),
                    "repos": sorted(self.repos),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("decontam.index_saved", path=str(path),
                 instances=len(self.instance_ids), repos=len(self.repos))

    @classmethod
    def load(cls, path: Path) -> ContaminationIndex:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            instance_ids=set(data["instance_ids"]),
            repo_commits=set(data["repo_commits"]),
            repos=set(data["repos"]),
            sources=data.get("sources", []),
        )


def build_index(benchmark_names: list[str]) -> ContaminationIndex:
    """Load every held-out benchmark and fingerprint all of its instances."""
    from datasets import load_dataset  # imported late: heavy, optional at runtime

    index = ContaminationIndex(sources=list(benchmark_names))
    for name in benchmark_names:
        loaded_any = False
        for split in ("test", "dev", "validation", "train"):
            try:
                ds = load_dataset(name, split=split)
            except Exception as exc:  # noqa: BLE001 - split simply may not exist
                log.debug("decontam.split_absent", benchmark=name, split=split,
                          error=str(exc)[:120])
                continue
            loaded_any = True
            for row in ds:
                index.add_instance(
                    Instance(
                        instance_id=row.get("instance_id", ""),
                        repo=row.get("repo", ""),
                        base_commit=row.get("base_commit", ""),
                        problem_statement=row.get("problem_statement", ""),
                        source=name,
                    )
                )
            log.info("decontam.loaded", benchmark=name, split=split, rows=len(ds))
        if not loaded_any:
            raise RuntimeError(
                f"Could not load any split of held-out benchmark {name!r}. "
                "Refusing to continue: an unverified held-out set is worse than none."
            )
    return index


@dataclass
class FilterReport:
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def note_drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return self.kept + sum(self.dropped.values())

    def render(self) -> str:
        lines = [f"  kept:    {self.kept:>7,}  ({self.kept / max(self.total, 1):.1%})"]
        for reason, count in sorted(self.dropped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  dropped: {count:>7,}  ({reason})")
        return "\n".join(lines)


def filter_instances(
    instances: list[Instance], index: ContaminationIndex, strict: bool = False
) -> tuple[list[Instance], FilterReport]:
    report = FilterReport()
    kept: list[Instance] = []
    for inst in instances:
        reason = index.is_contaminated(inst, strict=strict)
        if reason:
            report.note_drop(f"contaminated:{reason}")
        else:
            kept.append(inst)
            report.kept += 1
    log.info("decontam.filtered", kept=report.kept, dropped=report.total - report.kept)
    return kept, report

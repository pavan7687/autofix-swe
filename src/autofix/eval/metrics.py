"""Metrics.

Four numbers, each answering a different question. Reporting only the first is
the most common way an ML writeup becomes unfalsifiable.

* **resolve rate** — the headline. Fraction of instances where a sampled patch
  made FAIL_TO_PASS pass without breaking PASS_TO_PASS. Execution-derived.
* **localisation acc@k** — did retrieval put a truly-buggy file in the top k?
  Isolates the reranker from the editor. If this is low, no editor can succeed.
* **BM25 recall@N** — the ceiling the reranker operates under. The reranker
  cannot recover a file BM25 never surfaced, so this bounds the whole system
  and must be reported alongside acc@k, or acc@k looks better than it is.
* **apply rate** — fraction of generated diffs that were syntactically valid
  and applied cleanly. Separates "the model cannot fix bugs" from "the model
  cannot emit a well-formed diff", which have completely different remedies.

`pass@k` uses the unbiased estimator from the Codex paper rather than the naive
"any of k succeeded", which is biased upward when n > k.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb


@dataclass
class EvalMetrics:
    total: int = 0
    resolved: int = 0
    applied: int = 0
    parsed: int = 0
    samples: int = 0
    sample_resolved: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    localisation_hits_at: dict[int, int] = field(default_factory=dict)
    bm25_recall_hits: int = 0
    localisation_evaluated: int = 0

    def note_error(self, reason: str) -> None:
        self.errors[reason] = self.errors.get(reason, 0) + 1

    @property
    def resolve_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0

    @property
    def apply_rate(self) -> float:
        return self.applied / self.samples if self.samples else 0.0

    @property
    def parse_rate(self) -> float:
        return self.parsed / self.samples if self.samples else 0.0

    @property
    def sample_solve_rate(self) -> float:
        return self.sample_resolved / self.samples if self.samples else 0.0

    def acc_at(self, k: int) -> float:
        if not self.localisation_evaluated:
            return 0.0
        return self.localisation_hits_at.get(k, 0) / self.localisation_evaluated

    @property
    def bm25_recall(self) -> float:
        if not self.localisation_evaluated:
            return 0.0
        return self.bm25_recall_hits / self.localisation_evaluated

    def as_dict(self) -> dict:
        return {
            "instances": self.total,
            "resolved": self.resolved,
            "resolve_rate": round(self.resolve_rate, 4),
            "localisation": {
                "acc@1": round(self.acc_at(1), 4),
                "acc@3": round(self.acc_at(3), 4),
                "acc@5": round(self.acc_at(5), 4),
                "bm25_recall@candidates": round(self.bm25_recall, 4),
                "evaluated_on": self.localisation_evaluated,
            },
            "generation": {
                "samples": self.samples,
                "parse_rate": round(self.parse_rate, 4),
                "apply_rate": round(self.apply_rate, 4),
                "sample_solve_rate": round(self.sample_solve_rate, 4),
            },
            "errors": dict(sorted(self.errors.items(), key=lambda kv: -kv[1])),
        }


def record_localisation(
    metrics: EvalMetrics, predicted: list[str], bm25_candidates: list[str],
    gold: list[str],
) -> None:
    if not gold:
        return
    metrics.localisation_evaluated += 1
    gold_set = set(gold)

    if gold_set & set(bm25_candidates):
        metrics.bm25_recall_hits += 1

    for k in (1, 3, 5):
        if gold_set & set(predicted[:k]):
            metrics.localisation_hits_at[k] = metrics.localisation_hits_at.get(k, 0) + 1


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021).

    n = samples drawn, c = samples that passed, k = the k being reported.
    Returns the probability that at least one of k random draws passes.
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def aggregate_pass_at_k(per_instance: list[tuple[int, int]], k: int) -> float:
    """Mean unbiased pass@k over instances, each given as (n_samples, n_passed)."""
    if not per_instance:
        return 0.0
    return sum(pass_at_k(n, c, k) for n, c in per_instance) / len(per_instance)

"""Assemble every eval run into the ablation table.

    python -m autofix.eval.table

Reads `artifacts/runs/eval-*/results.json` and emits markdown ready to paste
into a README or report. The table *is* the deliverable of this project: a
single resolve-rate number proves nothing without the untrained baseline and
the retrieval ablation sitting next to it.
"""
from __future__ import annotations

import json
from pathlib import Path

from autofix.config import get_settings

# Canonical row order. Anything not listed is appended alphabetically.
_PREFERRED = ["baseline", "sft", "sft-rs1", "sft-rs2", "bm25-only", "oracle-files"]


def collect(run_root: Path) -> list[dict]:
    runs = []
    for results in sorted(run_root.glob("eval-*/results.json")):
        try:
            runs.append(json.loads(results.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return runs


def sort_runs(runs: list[dict]) -> list[dict]:
    """Canonical row order, independent of filesystem or caller ordering.

    `baseline` must come first: a resolve rate is meaningless without the
    untrained control directly above it for comparison.
    """
    order = {tag: i for i, tag in enumerate(_PREFERRED)}
    return sorted(runs, key=lambda r: (order.get(r.get("tag", ""), 99), r.get("tag", "")))


def render(runs: list[dict]) -> str:
    if not runs:
        return "No eval runs found. Run `autofix-eval --tag <name>` first."
    runs = sort_runs(runs)

    lines = [
        "| Configuration | Resolve rate | acc@1 | acc@3 | acc@5 | BM25 recall | Apply rate | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        loc = run.get("localisation", {})
        gen = run.get("generation", {})
        lines.append(
            f"| `{run.get('tag', '?')}` "
            f"| **{_pct(run.get('resolve_rate'))}** "
            f"| {_pct(loc.get('acc@1'))} | {_pct(loc.get('acc@3'))} "
            f"| {_pct(loc.get('acc@5'))} | {_pct(loc.get('bm25_recall@candidates'))} "
            f"| {_pct(gen.get('apply_rate'))} | {run.get('instances', 0)} |"
        )

    lines.append("")
    lines.append("**Reading this table**")
    lines.append("")
    lines.append("- `baseline` is the untrained base model. Every other row must beat it "
                 "or the fine-tuning did nothing.")
    lines.append("- `acc@k` isolates retrieval: if it is low, no editor can succeed, and "
                 "resolve rate is retrieval-bound rather than generation-bound.")
    lines.append("- `BM25 recall` is the ceiling on `acc@k` — the reranker cannot recover "
                 "a file stage-1 never surfaced.")
    lines.append("- `apply rate` separates 'cannot fix bugs' from 'cannot emit a valid "
                 "diff'. A low apply rate is a formatting problem, not a reasoning one.")
    return "\n".join(lines)


def _pct(value: object) -> str:
    return f"{value:.1%}" if isinstance(value, int | float) else "—"


def main() -> None:
    settings = get_settings()
    runs = collect(settings.run_root)
    table = render(runs)
    print(table)
    out = settings.run_root / "RESULTS.md"
    out.write_text(f"# Results\n\n{table}\n", encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()

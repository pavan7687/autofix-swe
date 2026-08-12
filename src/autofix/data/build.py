"""Dataset construction: raw instances -> two training sets.

    autofix-data --limit-per-source 5000
    autofix-data --strict          # repository-level decontamination

Produces, under DATA_ROOT:

    contamination_index.json   fingerprints of every held-out benchmark instance
    retrieval/                 reranker training data
    editing/                   editor training data
    stats.json                 counts, drop reasons, length distribution

Order of operations is deliberate and must not be rearranged: the contamination
index is built and saved **before** any training example is written, so there
is no window in which an unfiltered example can reach disk.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from autofix.config import get_settings
from autofix.data.decontaminate import (
    FilterReport,
    build_index,
    filter_instances,
)
from autofix.data.sources import SOURCES, deduplicate, load_all
from autofix.logging_conf import configure_logging, get_logger
from autofix.models import EditingExample, Instance, RetrievalExample, Split
from autofix.prompting import estimate_tokens

log = get_logger(__name__)

# An editing example whose prompt exceeds the training context is unusable; a
# trivially short one teaches nothing. Both ends are trimmed.
_MIN_PATCH_CHARS = 20
_MAX_PATCH_CHARS = 20_000
_MAX_PROMPT_TOKENS = 30_000


def build_retrieval_examples(
    instances: list[Instance], n_candidates: int, rng: random.Random
) -> list[RetrievalExample]:
    """Build reranker training data.

    Candidate lists are synthesised rather than produced by running BM25 over
    every repository at every commit — that would require checking out tens of
    thousands of snapshots. Instead each example mixes the gold files with
    distractors sampled from other instances in the same repository, which
    preserves the discrimination the reranker must learn (real paths from a
    real project) at a tiny fraction of the cost.

    The eval harness *does* run true BM25, so reported retrieval numbers are
    honest; this shortcut affects training data only.
    """
    by_repo: dict[str, list[str]] = {}
    for inst in instances:
        by_repo.setdefault(inst.repo, []).extend(inst.gold_files)
    for repo in by_repo:
        by_repo[repo] = sorted(set(by_repo[repo]))

    examples: list[RetrievalExample] = []
    for inst in instances:
        gold = [f for f in inst.gold_files if f]
        if not gold:
            continue

        pool = [f for f in by_repo.get(inst.repo, []) if f not in set(gold)]
        if len(pool) < 5:
            continue  # too few distractors to make the task non-trivial

        k = min(n_candidates - len(gold), len(pool))
        distractors = rng.sample(pool, k)
        candidates = gold + distractors
        rng.shuffle(candidates)

        first_gold = min(
            (candidates.index(g) for g in gold if g in candidates), default=None
        )
        examples.append(
            RetrievalExample(
                instance_id=inst.instance_id,
                repo=inst.repo,
                problem_statement=inst.problem_statement,
                candidates=candidates,
                gold_files=gold,
                bm25_hit=True,
                bm25_rank_of_first_gold=first_gold,
            )
        )
    return examples


def build_editing_examples(instances: list[Instance]) -> tuple[list[EditingExample], Counter]:
    """Build editor training data.

    File contents are reconstructed from the patch's own context lines rather
    than by checking out each repository. A unified diff carries the surrounding
    source, which is exactly the context the model needs and no more — and it
    sidesteps cloning thousands of repos at specific commits.
    """
    drops: Counter = Counter()
    examples: list[EditingExample] = []

    for inst in instances:
        patch = inst.patch.strip()
        if len(patch) < _MIN_PATCH_CHARS:
            drops["patch_too_short"] += 1
            continue
        if len(patch) > _MAX_PATCH_CHARS:
            drops["patch_too_long"] += 1
            continue
        if not inst.problem_statement.strip():
            drops["no_problem_statement"] += 1
            continue
        if any(_is_test_file(f) for f in inst.gold_files):
            drops["patch_touches_tests"] += 1
            continue

        contents = reconstruct_context(patch)
        if not contents:
            drops["no_reconstructable_context"] += 1
            continue

        tokens = estimate_tokens(inst.problem_statement + "".join(contents.values()) + patch)
        if tokens > _MAX_PROMPT_TOKENS:
            drops["prompt_too_long"] += 1
            continue

        examples.append(
            EditingExample(
                instance_id=inst.instance_id,
                repo=inst.repo,
                problem_statement=inst.problem_statement,
                file_contents=contents,
                patch=patch,
                token_estimate=tokens,
            )
        )
    return examples, drops


def reconstruct_context(patch: str) -> dict[str, str]:
    """Recover pre-patch source fragments from a unified diff.

    For each hunk we keep the context lines and the removed lines, which
    together are the file as it looked *before* the fix — the state the model
    must reason about at inference time.
    """
    files: dict[str, list[str]] = {}
    current: str | None = None

    for line in patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            current = parts[-1][2:] if len(parts) >= 4 else None
            if current:
                files.setdefault(current, [])
        elif line.startswith("--- ") and current is None:
            path = line[4:].strip()
            current = path[2:] if path.startswith("a/") else path
            files.setdefault(current, [])
        elif line.startswith(("+++", "index ", "new file", "deleted file", "similarity")):
            continue
        elif line.startswith("@@"):
            if current:
                files[current].append("...")
        elif current is not None and line[:1] in (" ", "-"):
            files[current].append(line[1:])

    return {
        path: "\n".join(lines).strip()
        for path, lines in files.items()
        if len([ln for ln in lines if ln != "..."]) >= 3
    }


def _is_test_file(path: str) -> bool:
    lower = path.lower()
    padded = f"/{lower}"
    return (
        "/tests/" in padded or "/test/" in padded or "/test_" in padded
        or lower.endswith(("_test.py", "_test.go", ".test.js", ".spec.ts"))
    )


def assign_splits(
    examples: list, rng: random.Random, val_fraction: float = 0.02
) -> dict[str, list]:
    """Split by repository, never by example.

    A random example-level split leaks: two instances from the same repository
    at nearby commits share almost all their code, so the model would see
    validation repositories during training and the validation loss would be
    optimistic. Grouping by repo is the only defensible split here.
    """
    repos = sorted({e.repo for e in examples})
    rng.shuffle(repos)
    n_val = max(1, int(len(repos) * val_fraction))
    val_repos = set(repos[:n_val])

    out: dict[str, list] = {Split.TRAIN: [], Split.VALIDATION: []}
    for example in examples:
        key = Split.VALIDATION if example.repo in val_repos else Split.TRAIN
        out[key].append(example)
    return out


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")
    log.info("data.written", path=str(path), rows=len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autofix-data",
        description="Build decontaminated retrieval and editing training sets.",
    )
    parser.add_argument("--sources", nargs="*", default=None,
                        choices=[s.name for s in SOURCES],
                        help="subset of sources (default: all)")
    parser.add_argument("--limit-per-source", type=int, default=None,
                        help="cap instances per source; useful for a smoke run")
    parser.add_argument("--strict", action="store_true",
                        help="repository-level decontamination (drops more data, "
                             "strongest guarantee)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    settings.ensure_dirs()
    rng = random.Random(args.seed if args.seed is not None else settings.seed)

    # 1. Held-out fingerprints FIRST. Nothing is written before this succeeds.
    print(f"\nBuilding contamination index from {settings.eval_benchmarks} ...")
    index = build_index(settings.eval_benchmarks)
    index.save(settings.contamination_index)

    # 2. Raw instances.
    print("\nLoading source corpora ...")
    instances = load_all(args.sources, args.limit_per_source)
    if not instances:
        raise SystemExit("No instances loaded. Check network access to HuggingFace.")
    instances = deduplicate(instances)

    # 3. Filter.
    print(f"\nDecontaminating ({'repository-level' if args.strict else 'repo+commit'}) ...")
    clean, report = filter_instances(instances, index, strict=args.strict)
    print(report.render())
    if not clean:
        raise SystemExit("Every instance was filtered out. Check the index.")

    # 4. Build both task datasets.
    print("\nBuilding retrieval examples ...")
    retrieval = build_retrieval_examples(clean, settings.retrieval_candidates, rng)
    retrieval = [e for e in retrieval if e.is_trainable]

    print("Building editing examples ...")
    editing, edit_drops = build_editing_examples(clean)

    # 5. Repo-grouped splits, then write.
    for name, examples, out_dir in (
        ("retrieval", retrieval, settings.retrieval_dataset),
        ("editing", editing, settings.editing_dataset),
    ):
        splits = assign_splits(examples, rng)
        for split_name, rows in splits.items():
            write_jsonl(out_dir / f"{split_name}.jsonl", rows)
        print(f"  {name}: {len(splits[Split.TRAIN]):,} train / "
              f"{len(splits[Split.VALIDATION]):,} val")

    _write_stats(settings.data_root, report, retrieval, editing, edit_drops, args.strict)
    print(f"\nDone. Artifacts in {settings.data_root}")


def _write_stats(
    root: Path, report: FilterReport, retrieval: list, editing: list,
    edit_drops: Counter, strict: bool,
) -> None:
    lengths = sorted(e.token_estimate for e in editing) or [0]
    stats = {
        "decontamination": {
            "strict_repo_level": strict,
            "kept": report.kept,
            "dropped": dict(report.dropped),
        },
        "retrieval_examples": len(retrieval),
        "editing_examples": len(editing),
        "editing_drop_reasons": dict(edit_drops),
        "editing_token_estimate": {
            "p50": lengths[len(lengths) // 2],
            "p90": lengths[int(len(lengths) * 0.9)],
            "p99": lengths[int(len(lengths) * 0.99)],
            "max": lengths[-1],
        },
    }
    (root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\nDataset statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

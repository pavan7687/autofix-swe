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
    ContaminationIndex,
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
        # Cap gold files at the candidate budget. A patch touching more files
        # than the whole candidate list is a refactor, not a bug fix, and it
        # would leave no room for distractors - which is what made
        # `n_candidates - len(gold)` go negative and crash rng.sample.
        gold = [f for f in inst.gold_files if f][:n_candidates]
        if not gold:
            continue

        pool = [f for f in by_repo.get(inst.repo, []) if f not in set(gold)]
        if len(pool) < 5:
            continue  # too few distractors to make the task non-trivial

        k = max(0, min(n_candidates - len(gold), len(pool)))
        if k == 0:
            continue  # no distractors -> the answer is "all of them", teaches nothing
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
    examples: list, rng: random.Random, val_fraction: float = 0.03
) -> dict[str, list]:
    """Split by repository, targeting a fraction of EXAMPLES not of repos.

    Grouping by repository is non-negotiable: two instances from one repo at
    nearby commits share almost all their code, so an example-level split leaks
    and the validation loss comes out optimistic.

    But repository sizes are extremely skewed — a handful of projects supply
    most instances. Taking a fixed *fraction of repositories* therefore gives
    wildly variable splits: picking 2% of repos once produced a 29% validation
    set because one large repo happened to be selected.

    So: shuffle repositories, then accumulate them into validation until the
    target share of *examples* is reached, skipping any repo that would
    overshoot the cap. Grouping is preserved, size is controlled.
    """
    by_repo: dict[str, int] = {}
    for example in examples:
        by_repo[example.repo] = by_repo.get(example.repo, 0) + 1

    total = len(examples)
    target = max(1, int(total * val_fraction))
    ceiling = int(total * val_fraction * 2)  # never more than 2x the target

    # Only consider repositories that could fit inside the ceiling at all.
    # Filtering BEFORE the loop matters: an "unless the set is still empty"
    # escape hatch inside the loop lets the first shuffled repo in regardless of
    # size, which is exactly how a 3% target produced a 33% validation set.
    eligible = [r for r in sorted(by_repo) if by_repo[r] <= ceiling]
    rng.shuffle(eligible)

    val_repos: set[str] = set()
    val_count = 0
    for repo in eligible:
        if val_count >= target:
            break
        size = by_repo[repo]
        if val_count + size > ceiling:
            continue  # would overshoot; a smaller repo may still fit
        val_repos.add(repo)
        val_count += size

    if not val_repos:
        # Every repository is larger than the ceiling (very few, very large
        # repos). Take the smallest and accept the oversized split rather than
        # returning no validation data at all.
        smallest = min(by_repo, key=lambda r: by_repo[r])
        val_repos = {smallest}
        val_count = by_repo[smallest]
        log.warning("split.oversized", repo=smallest, examples=val_count,
                    note="all repos exceed the validation ceiling")

    out: dict[str, list] = {Split.TRAIN: [], Split.VALIDATION: []}
    for example in examples:
        key = Split.VALIDATION if example.repo in val_repos else Split.TRAIN
        out[key].append(example)

    log.info("split.assigned", total=total, val=len(out[Split.VALIDATION]),
             val_repos=len(val_repos), target=target)
    return out


def selftest_filter(index: ContaminationIndex, rng: random.Random) -> dict:
    """Positive control for the decontamination filter.

    Real training corpora are often already disjoint from the benchmarks, so
    `dropped == 0` does not by itself prove the filter works — it could equally
    mean the filter is broken. This injects synthetic instances built from the
    held-out index and asserts every one is caught, which distinguishes the two
    cases. It runs on every build and is recorded in stats.json.
    """
    if not index.repo_commits:
        return {"ran": False, "reason": "empty contamination index"}

    sample_keys = rng.sample(sorted(index.repo_commits), min(5, len(index.repo_commits)))
    canaries: list[Instance] = []
    for key in sample_keys:
        repo, _, commit = key.partition("@")
        canaries.append(
            Instance(instance_id=f"canary-{commit}", repo=repo, base_commit=commit,
                     problem_statement="synthetic canary", patch="x", source="selftest")
        )
    # A clean instance must survive, or the filter is simply dropping everything.
    control = Instance(instance_id="canary-clean", repo="definitely/not-a-benchmark-repo",
                       base_commit="0" * 40, problem_statement="synthetic control",
                       patch="x", source="selftest")

    kept, _ = filter_instances([*canaries, control], index, strict=False)
    kept_ids = {i.instance_id for i in kept}
    caught = len(canaries) - len([i for i in canaries if i.instance_id in kept_ids])
    passed = caught == len(canaries) and "canary-clean" in kept_ids

    return {
        "ran": True,
        "passed": passed,
        "canaries_injected": len(canaries),
        "canaries_caught": caught,
        "clean_control_survived": "canary-clean" in kept_ids,
    }


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

    selftest = selftest_filter(index, rng)
    if selftest.get("ran") and not selftest.get("passed"):
        raise SystemExit(
            f"Decontamination self-test FAILED: {selftest}. "
            "Refusing to build a training set that cannot be trusted."
        )
    print(f"  self-test: {selftest['canaries_caught']}/{selftest['canaries_injected']} "
          f"injected benchmark instances correctly rejected, clean control survived")

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
    if not report.dropped:
        # Expected, not alarming: the public corpora are curated to avoid the
        # benchmark repositories. The self-test above is what proves the filter
        # is actually working.
        print("  (zero overlap - the source corpora are already disjoint from "
              "the benchmarks; the self-test above confirms the filter works)")
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

    _write_stats(settings.data_root, report, retrieval, editing, edit_drops,
                 args.strict, selftest)
    print(f"\nDone. Artifacts in {settings.data_root}")


def _write_stats(
    root: Path, report: FilterReport, retrieval: list, editing: list,
    edit_drops: Counter, strict: bool, selftest: dict | None = None,
) -> None:
    lengths = sorted(e.token_estimate for e in editing) or [0]
    stats = {
        "decontamination": {
            "strict_repo_level": strict,
            "kept": report.kept,
            "dropped": dict(report.dropped),
            "filter_self_test": selftest,
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

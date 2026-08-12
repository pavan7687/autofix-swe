"""Rejection sampling: turn the model's own successes into round-2 training data.

    autofix-sample --round 1 --instances 2000

The loop, per training instance:

    sample k patches (temp 0.8)  ->  verify each in the sandbox
                                       |
                        keep ONLY the ones whose tests pass
                                       |
                         write as new editing examples

This is STaR / rejection-sampling fine-tuning. Its appeal for this problem is
that the filter is **execution, not judgement**: there is no reward model, no
LLM-as-judge, and nothing for the policy to game. A kept example is one where
the repository's own test suite went green.

Why it improves on the SFT checkpoint even though the data comes from the same
model: sampling at temperature 0.8 explores k different solutions and we keep
only the successes, so the round-2 distribution is conditioned on correctness.
The model is trained on its own best behaviour rather than its average.

Two properties to preserve if you modify this:

* **Deduplicate kept patches per instance.** Eight identical solutions to one
  easy bug would otherwise dominate the gradient and teach the model that easy
  bugs are all that exist.
* **Cap kept samples per instance** (`--max-keep`), for the same reason.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from autofix.config import get_settings
from autofix.data.build import write_jsonl
from autofix.logging_conf import configure_logging, get_logger
from autofix.models import Candidate, EditingExample, Instance
from autofix.prompting import estimate_tokens
from autofix.rejection.workspace import InstanceWorkspace
from autofix.serving.client import LocalLlmClient
from autofix.serving.pipeline import InferencePipeline

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autofix-sample",
        description="Execution-verified rejection sampling over training instances.",
    )
    p.add_argument("--round", type=int, default=1, help="sampling round number")
    p.add_argument("--instances", type=int, default=500,
                   help="how many training instances to sample over")
    p.add_argument("--k", type=int, default=None,
                   help="samples per instance (default: SAMPLES_PER_ISSUE)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-keep", type=int, default=2,
                   help="max verified patches kept per instance")
    p.add_argument("--reranker-model", default="reranker")
    p.add_argument("--editor-model", default="editor")
    p.add_argument("--workers", type=int, default=None,
                   help="instances verified in parallel (default: MAX_VERIFY_CONCURRENCY)")
    p.add_argument("--resume", action="store_true",
                   help="skip instances already present in the output shard")
    return p


async def sample_instance(
    settings, pipeline: InferencePipeline, instance: Instance, k: int,
    temperature: float,
) -> tuple[list[Candidate], str | None]:
    """Check out the instance's repo, sample k patches, verify each."""
    async with InstanceWorkspace(settings, instance) as workspace:
        if workspace.error:
            return [], workspace.error
        result = await pipeline.run(
            instance, workspace.repo_dir, n_samples=k, temperature=temperature
        )
        return result.candidates, result.error


def candidates_to_examples(
    instance: Instance, candidates: list[Candidate], max_keep: int
) -> list[EditingExample]:
    """Keep verified patches only, de-duplicated, capped per instance."""
    kept: list[EditingExample] = []
    seen_diffs: set[str] = set()

    for candidate in candidates:
        if not candidate.resolved:
            continue
        normalised = _normalise(candidate.diff)
        if normalised in seen_diffs:
            continue
        seen_diffs.add(normalised)

        contents = _context_from_diff(candidate.diff)
        if not contents:
            continue
        kept.append(
            EditingExample(
                instance_id=f"{instance.instance_id}#rs{candidate.sample_index}",
                repo=instance.repo,
                problem_statement=instance.problem_statement,
                file_contents=contents,
                patch=candidate.diff,
                reasoning=_reasoning_of(candidate.raw_output),
                token_estimate=estimate_tokens(
                    instance.problem_statement + "".join(contents.values())
                ),
            )
        )
        if len(kept) >= max_keep:
            break
    return kept


def _normalise(diff: str) -> str:
    """Ignore hunk-offset and whitespace noise when de-duplicating."""
    lines = [
        ln.rstrip() for ln in diff.splitlines()
        if ln[:1] in ("+", "-") and not ln.startswith(("+++", "---"))
    ]
    return "\n".join(lines)


def _context_from_diff(diff: str) -> dict[str, str]:
    from autofix.data.build import reconstruct_context

    return reconstruct_context(diff)


def _reasoning_of(raw: str) -> str:
    from autofix.agent.responses import extract_block

    return extract_block(raw, "reasoning") or ""


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    settings.ensure_dirs()

    k = args.k or settings.samples_per_issue
    temperature = (
        args.temperature if args.temperature is not None else settings.sampling_temperature
    )
    workers = args.workers or settings.max_verify_concurrency

    out_dir = settings.data_root / f"rejection_round{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_path = out_dir / "train.jsonl"
    log_path = out_dir / "attempts.jsonl"

    instances = _load_training_instances(settings, args.instances)
    if args.resume and log_path.exists():
        done = {json.loads(ln)["instance_id"] for ln in log_path.read_text().splitlines() if ln}
        instances = [i for i in instances if i.instance_id not in done]
        print(f"Resuming: {len(done)} already attempted, {len(instances)} remaining.")

    print(f"\nRejection sampling round {args.round}")
    print(f"  instances : {len(instances)}")
    print(f"  k         : {k} at temperature {temperature}")
    print(f"  workers   : {workers}")
    print(f"  output    : {out_dir}\n")

    async with LocalLlmClient(settings.vllm_base_url, settings.vllm_api_key) as client:
        if not await client.health():
            raise SystemExit(
                f"No vLLM server at {settings.vllm_base_url}. "
                "Start it with scripts/serve_vllm.sh first."
            )

        pipeline = InferencePipeline(
            settings, client, args.reranker_model, args.editor_model
        )
        semaphore = asyncio.Semaphore(workers)
        stats: Counter = Counter()
        all_kept: list[EditingExample] = []
        started = time.monotonic()

        async def worker(index: int, instance: Instance) -> None:
            async with semaphore:
                try:
                    candidates, error = await sample_instance(
                        settings, pipeline, instance, k, temperature
                    )
                except Exception as exc:  # noqa: BLE001 - one bad repo must not stop the run
                    log.warning("sample.failed", instance=instance.instance_id,
                                error=str(exc)[:200])
                    candidates, error = [], str(exc)[:200]

                resolved = sum(c.resolved for c in candidates)
                stats["instances"] += 1
                stats["samples"] += len(candidates)
                stats["resolved"] += resolved
                if resolved:
                    stats["instances_with_fix"] += 1
                if error:
                    stats["errors"] += 1
                for candidate in candidates:
                    if candidate.rejection:
                        stats[f"reject:{candidate.rejection.split(':')[0]}"] += 1

                kept = candidates_to_examples(instance, candidates, args.max_keep)
                all_kept.extend(kept)

                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "instance_id": instance.instance_id,
                        "samples": len(candidates),
                        "resolved": resolved,
                        "kept": len(kept),
                        "error": error,
                    }) + "\n")

                rate = stats["resolved"] / max(stats["samples"], 1)
                print(f"  [{index + 1}/{len(instances)}] {instance.instance_id[:44]:<46} "
                      f"{resolved}/{len(candidates)} solved   "
                      f"pass@{k}={stats['instances_with_fix']}/{stats['instances']}  "
                      f"sample-rate={rate:.1%}")

        await asyncio.gather(*(worker(i, inst) for i, inst in enumerate(instances)))

    write_jsonl(kept_path, all_kept)
    _write_summary(out_dir, stats, all_kept, k, temperature,
                   round(time.monotonic() - started, 1))
    return 0


def _load_training_instances(settings, limit: int) -> list[Instance]:
    """Instances to sample over.

    Requires FAIL_TO_PASS: without an explicit list of tests the fix must turn
    green, "resolved" degrades to "the whole suite is green", which many repos
    never are. Those instances are fine for SFT but useless as an RL signal.
    """
    from autofix.data.sources import SOURCES, load_source

    spec = next(s for s in SOURCES if s.name == "swegym")
    instances = load_source(spec, limit * 3)
    usable = [i for i in instances if i.fail_to_pass and i.base_commit]
    log.info("sample.instances_selected", loaded=len(instances), usable=len(usable))
    return usable[:limit]


def _write_summary(out_dir: Path, stats: Counter, kept: list, k: int,
                   temperature: float, seconds: float) -> None:
    summary = {
        "samples_per_instance": k,
        "temperature": temperature,
        "wall_seconds": seconds,
        "instances_attempted": stats["instances"],
        "instances_with_at_least_one_fix": stats["instances_with_fix"],
        f"pass_at_{k}": round(
            stats["instances_with_fix"] / max(stats["instances"], 1), 4
        ),
        "sample_level_solve_rate": round(
            stats["resolved"] / max(stats["samples"], 1), 4
        ),
        "examples_kept": len(kept),
        "rejection_reasons": {
            key.split(":", 1)[1]: value
            for key, value in stats.items() if key.startswith("reject:")
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nRound complete. Retrain with the round data merged into {out_dir}")


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()

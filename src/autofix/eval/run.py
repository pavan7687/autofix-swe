"""Evaluation on SWE-bench Lite.

    autofix-eval --split test --limit 300 --k 1
    autofix-eval --editor-model editor-base --tag baseline   # untrained control

Every configuration in the ablation table is produced by this one command with
different `--editor-model` / `--reranker-model` / `--tag`, so the numbers are
comparable by construction rather than by hope.

The **baseline is the untrained base model**, served under its own name. That
comparison is what demonstrates the fine-tuning did the work, and it is the
control an interviewer will ask for first.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from autofix.config import get_settings
from autofix.data.sources import files_in_patch
from autofix.eval.metrics import EvalMetrics, aggregate_pass_at_k, record_localisation
from autofix.logging_conf import configure_logging, get_logger
from autofix.models import Instance
from autofix.rejection.workspace import InstanceWorkspace
from autofix.serving.client import LocalLlmClient
from autofix.serving.pipeline import InferencePipeline

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autofix-eval", description="Evaluate on SWE-bench Lite."
    )
    p.add_argument("--benchmark", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--k", type=int, default=1, help="samples per instance")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--reranker-model", default="reranker")
    p.add_argument("--editor-model", default="editor")
    p.add_argument("--tag", default="run", help="label for this configuration")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    return p


def load_benchmark(name: str, split: str, limit: int) -> list[Instance]:
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    instances: list[Instance] = []
    for row in ds:
        patch = row.get("patch", "") or ""
        instances.append(
            Instance(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                problem_statement=row.get("problem_statement", ""),
                patch=patch,
                test_patch=row.get("test_patch", "") or "",
                fail_to_pass=_as_list(row.get("FAIL_TO_PASS")),
                pass_to_pass=_as_list(row.get("PASS_TO_PASS")),
                environment_setup_commit=row.get("environment_setup_commit", "") or "",
                source=name,
                gold_files=files_in_patch(patch),
            )
        )
        if len(instances) >= limit:
            break
    return instances


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
            return [str(v) for v in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    settings.ensure_dirs()
    workers = args.workers or settings.max_verify_concurrency

    out_dir = settings.run_root / f"eval-{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"

    instances = load_benchmark(args.benchmark, args.split, args.limit)
    if args.resume and predictions_path.exists():
        done = {
            json.loads(ln)["instance_id"]
            for ln in predictions_path.read_text().splitlines() if ln
        }
        instances = [i for i in instances if i.instance_id not in done]
        print(f"Resuming: {len(done)} done, {len(instances)} remaining.")

    print(f"\nEvaluating [{args.tag}]")
    print(f"  benchmark : {args.benchmark} ({args.split})")
    print(f"  instances : {len(instances)}")
    print(f"  models    : reranker={args.reranker_model} editor={args.editor_model}")
    print(f"  k         : {args.k} at temperature {args.temperature}\n")

    metrics = EvalMetrics()
    per_instance_pass: list[tuple[int, int]] = []
    started = time.monotonic()

    async with LocalLlmClient(settings.vllm_base_url, settings.vllm_api_key) as client:
        if not await client.health():
            raise SystemExit(f"No vLLM server at {settings.vllm_base_url}.")

        pipeline = InferencePipeline(
            settings, client, args.reranker_model, args.editor_model
        )
        semaphore = asyncio.Semaphore(workers)

        async def worker(idx: int, instance: Instance) -> None:
            async with semaphore:
                async with InstanceWorkspace(settings, instance) as workspace:
                    if workspace.error:
                        metrics.total += 1
                        metrics.note_error("checkout_failed")
                        return
                    try:
                        result = await pipeline.run(
                            instance, workspace.repo_dir,
                            n_samples=args.k, temperature=args.temperature,
                        )
                    except Exception as exc:  # noqa: BLE001
                        metrics.total += 1
                        metrics.note_error(f"exception:{type(exc).__name__}")
                        log.warning("eval.instance_failed",
                                    instance=instance.instance_id, error=str(exc)[:200])
                        return

                metrics.total += 1
                record_localisation(
                    metrics, result.predicted_files, result.bm25_candidates,
                    instance.gold_files,
                )
                n_passed = sum(c.resolved for c in result.candidates)
                per_instance_pass.append((max(len(result.candidates), 1), n_passed))

                for candidate in result.candidates:
                    metrics.samples += 1
                    metrics.parsed += candidate.parsed
                    metrics.applied += candidate.applied
                    metrics.sample_resolved += candidate.resolved
                if result.resolved:
                    metrics.resolved += 1
                if result.error:
                    metrics.note_error(result.error[:60])

                with predictions_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "instance_id": instance.instance_id,
                        "model_name_or_path": args.tag,
                        "model_patch": result.diff,
                        "predicted_files": result.predicted_files,
                        "gold_files": instance.gold_files,
                        "resolved": result.resolved,
                    }) + "\n")

                mark = "PASS" if result.resolved else "fail"
                print(f"  [{idx + 1}/{len(instances)}] {instance.instance_id[:44]:<46} "
                      f"{mark}   running={metrics.resolve_rate:.1%}")

        await asyncio.gather(*(worker(i, inst) for i, inst in enumerate(instances)))

    report = metrics.as_dict()
    report["tag"] = args.tag
    report["config"] = {
        "benchmark": args.benchmark, "split": args.split, "k": args.k,
        "temperature": args.temperature, "reranker": args.reranker_model,
        "editor": args.editor_model,
    }
    report["pass_at_k"] = {
        f"pass@{k}": round(aggregate_pass_at_k(per_instance_pass, k), 4)
        for k in (1, args.k) if k <= args.k
    }
    report["wall_seconds"] = round(time.monotonic() - started, 1)

    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, indent=2))
    print(f"\nPredictions: {predictions_path}")
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()

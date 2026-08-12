"""`autofix-fix` — run the trained models against a real repository.

    autofix-fix --repo-path ./flask --problem "Blueprint prefix is dropped when ..."
    autofix-fix --repo-path ./flask --problem-file bug.txt --k 4

This is the demo surface: it takes a checkout and a bug description, runs the
full trained pipeline (BM25 -> reranker -> editor -> sandbox), and writes a
verified patch. It reports failure honestly when the sandbox cannot confirm a
fix, which is the common case and the point.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from autofix.config import get_settings
from autofix.logging_conf import configure_logging, get_logger
from autofix.models import Instance
from autofix.serving.client import LocalLlmClient
from autofix.serving.pipeline import InferencePipeline

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autofix-fix",
        description="Attempt a fix on a local checkout using the trained models.",
    )
    p.add_argument("--repo-path", type=Path, required=True,
                   help="path to a local git checkout")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", help="bug description as a string")
    group.add_argument("--problem-file", type=Path, help="file containing the description")
    p.add_argument("--k", type=int, default=1, help="candidate patches to sample")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--reranker-model", default="reranker")
    p.add_argument("--editor-model", default="editor")
    p.add_argument("--out", type=Path, default=Path("./autofix-output"))
    p.add_argument("--localise-only", action="store_true",
                   help="run retrieval only and print the ranked files; no generation")
    return p


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    repo_dir = args.repo_path.expanduser().resolve()
    if not (repo_dir / ".git").is_dir():
        raise SystemExit(f"{repo_dir} is not a git checkout.")

    problem = (
        args.problem_file.read_text(encoding="utf-8")
        if args.problem_file else args.problem
    )
    if not problem.strip():
        raise SystemExit("Empty problem description.")

    instance = Instance(
        instance_id="local", repo=repo_dir.name, base_commit="local",
        problem_statement=problem, source="cli",
    )

    async with LocalLlmClient(settings.vllm_base_url, settings.vllm_api_key) as client:
        if not await client.health():
            raise SystemExit(
                f"No vLLM server at {settings.vllm_base_url}. "
                "Start it with scripts/serve_vllm.sh."
            )
        pipeline = InferencePipeline(
            settings, client, args.reranker_model, args.editor_model
        )

        if args.localise_only:
            from autofix.agent.retrieval import RepoIndex

            index = RepoIndex(repo_dir).build()
            predicted, bm25 = await pipeline.localise(index, problem)
            print(f"\nBM25 surfaced {len(bm25)} candidates. Reranker selected:")
            for i, path in enumerate(predicted, 1):
                print(f"  {i}. {path}")
            return 0

        print(f"\nRepository : {repo_dir}")
        print(f"Sampling   : k={args.k} at temperature {args.temperature}\n")
        result = await pipeline.run(
            instance, repo_dir, n_samples=args.k, temperature=args.temperature
        )

    return _report(result, args.out)


def _report(result, out_dir: Path) -> int:
    print("\nLocalisation:")
    for i, path in enumerate(result.predicted_files, 1):
        print(f"  {i}. {path}")

    if result.error:
        print(f"\n[FAIL] {result.error}")
        return 2

    print(f"\nCandidates: {len(result.candidates)}")
    for candidate in result.candidates:
        status = "RESOLVED" if candidate.resolved else (candidate.rejection or "failed")
        print(f"  sample {candidate.sample_index}: {status[:80]}")

    if not result.resolved:
        print("\n[FAIL] No candidate passed the repository's tests. Nothing written.")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_path = out_dir / "fix.patch"
    patch_path.write_text(result.diff, encoding="utf-8")
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "resolved": True,
                "predicted_files": result.predicted_files,
                "samples": len(result.candidates),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[PASS] Verified fix written to {patch_path}")
    print(f"       Apply with: git apply {patch_path}")
    return 0


def main() -> None:
    args = build_parser().parse_args()
    try:
        sys.exit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()

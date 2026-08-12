"""The inference pipeline: bug report -> verified patch, using trained models.

This is the runtime counterpart of the training stack, and the thing the CLI
drives:

    BM25 (retrieval.py)  ->  reranker adapter  ->  editor adapter  ->  sandbox

Two properties are inherited from the training design and are worth stating:

* Prompts come from `autofix.prompting`, the same module the training data was
  built with, so there is no train/inference skew.
* Nothing is reported as a fix unless the sandbox says the tests pass. The
  system can return "no fix found", and does so often; that is correct
  behaviour, not a failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autofix.agent.retrieval import RepoIndex
from autofix.config import Settings
from autofix.logging_conf import get_logger
from autofix.models import Candidate, Instance
from autofix.prompting import edit_inference_messages, rerank_inference_messages
from autofix.rejection.verify import CandidateVerifier
from autofix.sandbox.detect import Toolchain, detect
from autofix.sandbox.runner import DockerSandbox
from autofix.serving.client import LocalLlmClient

log = get_logger(__name__)

_MAX_FILE_CHARS = 24_000


@dataclass
class InferenceResult:
    instance_id: str
    resolved: bool = False
    predicted_files: list[str] = field(default_factory=list)
    bm25_candidates: list[str] = field(default_factory=list)
    gold_files: list[str] = field(default_factory=list)
    diff: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    error: str | None = None

    @property
    def localisation_hit(self) -> bool:
        return bool(set(self.predicted_files) & set(self.gold_files))


class InferencePipeline:
    def __init__(
        self,
        settings: Settings,
        client: LocalLlmClient,
        reranker_model: str,
        editor_model: str,
    ) -> None:
        self._s = settings
        self._client = client
        self._reranker = reranker_model
        self._editor = editor_model

    async def localise(self, index: RepoIndex, problem: str) -> tuple[list[str], list[str]]:
        """BM25 shortlist, then reranker. Returns (predicted, bm25_candidates)."""
        bm25 = [path for path, _ in index.rank_files(problem, self._s.retrieval_candidates)]
        if not bm25:
            return [], []

        generation = await self._client.generate(
            model=self._reranker,
            messages=rerank_inference_messages(problem, bm25),
            temperature=0.0,
            max_tokens=256,
        )
        predicted = self._parse_paths(generation.texts[0], index, bm25)
        log.info("infer.localised", bm25=len(bm25), predicted=len(predicted))
        return predicted, bm25

    def _parse_paths(self, text: str, index: RepoIndex, allowed: list[str]) -> list[str]:
        """Map model output back onto real repository paths.

        The model is asked for paths one per line, but will sometimes number
        them, quote them, or emit a truncated suffix. Anything that cannot be
        resolved to a file that BM25 actually surfaced is discarded — the
        reranker is not permitted to invent candidates.
        """
        allowed_set = set(allowed)
        out: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip().strip("`\"'").lstrip("0123456789.-) ").strip()
            if not line or " " in line.strip():
                line = line.split()[-1] if line.split() else ""
            if not line:
                continue
            if line in allowed_set:
                out.append(line)
                continue
            resolved = index.resolve(line)
            if resolved in allowed_set:
                out.append(resolved)
        # de-duplicate, preserve order
        seen: dict[str, None] = {}
        for path in out:
            seen.setdefault(path, None)
        return list(seen)[: self._s.retrieval_keep]

    async def propose(
        self, index: RepoIndex, problem: str, files: list[str], n: int,
        temperature: float,
    ) -> list[Candidate]:
        contents = {}
        for path in files:
            text = index.read(path)
            if text is None:
                continue
            contents[path] = text[:_MAX_FILE_CHARS]
        if not contents:
            return []

        generation = await self._client.generate(
            model=self._editor,
            messages=edit_inference_messages(problem, contents),
            n=n,
            temperature=temperature,
            top_p=self._s.sampling_top_p,
            max_tokens=2048,
        )
        return [
            Candidate(instance_id="", sample_index=i, raw_output=text)
            for i, text in enumerate(generation.texts)
        ]

    async def run(
        self, instance: Instance, repo_dir: Path, n_samples: int = 1,
        temperature: float = 0.0,
    ) -> InferenceResult:
        result = InferenceResult(
            instance_id=instance.instance_id, gold_files=instance.gold_files
        )
        index = RepoIndex(repo_dir).build()

        predicted, bm25 = await self.localise(index, instance.problem_statement)
        result.predicted_files = predicted
        result.bm25_candidates = bm25
        if not predicted:
            result.error = "retrieval returned no usable files"
            return result

        toolchain: Toolchain | None = detect(repo_dir)
        if toolchain is None:
            result.error = "unsupported toolchain; cannot verify"
            return result

        candidates = await self.propose(
            index, instance.problem_statement, predicted, n_samples, temperature
        )
        for candidate in candidates:
            candidate.instance_id = instance.instance_id
        if not candidates:
            result.error = "editor produced no output"
            return result

        async with DockerSandbox(self._s, toolchain, repo_dir) as sandbox:
            install = await sandbox.prepare()
            if not install.passed:
                result.error = "dependency installation failed in sandbox"
                return result

            verifier = CandidateVerifier(self._s, instance, repo_dir, sandbox, toolchain)
            await verifier.establish_baseline()

            for candidate in candidates:
                await verifier.verify(candidate)
                result.candidates.append(candidate)
                if candidate.resolved:
                    result.resolved = True
                    result.diff = candidate.diff
                    break  # first verified fix wins; no reason to keep spending

        return result

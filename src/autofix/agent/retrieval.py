"""Stage-1 lexical retrieval over a repository checkout.

This is the coarse half of a coarse-to-fine pipeline: BM25 narrows thousands of
files to ~50 candidates, then a trained reranker (`autofix.serving.reranker`)
picks the ~5 that actually contain the defect. The same split is used by
SWE-Fixer, and for the same reasons.

Why BM25 and not embeddings for stage 1:

1. **No index to maintain.** Embeddings need an ingestion pass per repository
   snapshot; we evaluate across hundreds of repos at hundreds of commits.
2. **Bug reports are lexically rich.** Issues quote exception classes, function
   names and paths verbatim, which is the regime where exact-token matching is
   unusually strong.
3. **Determinism.** Identical input gives identical candidates, so a retrieval
   ablation measures the reranker rather than index drift.

The measured ceiling this imposes is reported as BM25 recall@50 in the eval
table — the reranker can never recover a file BM25 did not surface, so that
number bounds the whole system.
"""
from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from autofix.logging_conf import get_logger
from autofix.models import CodeChunk

log = get_logger(__name__)

SOURCE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rb", ".rs", ".c",
    ".h", ".cc", ".cpp", ".hpp", ".cs", ".kt", ".php", ".scala", ".swift",
}
SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".tox", ".mypy_cache", ".pytest_cache", "vendor", "third_party", ".next",
    "target", "coverage", "site-packages", ".eggs",
}
_MAX_FILE_BYTES = 400_000
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_MAX_CHUNK_LINES = 160


def split_identifier(token: str) -> list[str]:
    """`getUserById` / `get_user_by_id` -> ['get','user','by','id']."""
    out: list[str] = []
    for part in re.split(r"[_\-.]", token):
        out.extend(re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", part) or [part])
    return [p.lower() for p in out if p]


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _IDENT.finditer(text):
        token = match.group(0)
        tokens.append(token.lower())
        pieces = split_identifier(token)
        if len(pieces) > 1:
            tokens.extend(pieces)
    return tokens


@dataclass(slots=True)
class IndexedFile:
    path: str
    text: str
    tokens: Counter[str]
    length: int
    is_test: bool


class RepoIndex:
    """In-memory BM25 index over one checkout."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, IndexedFile] = {}
        self._doc_freq: Counter[str] = Counter()
        self._avg_len = 1.0

    def build(self) -> RepoIndex:
        for path in self._walk():
            rel = path.relative_to(self.root).as_posix()
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "\x00" in text[:1024]:
                continue
            tokens = Counter(tokenize(text))
            self.files[rel] = IndexedFile(
                path=rel, text=text, tokens=tokens,
                length=sum(tokens.values()) or 1, is_test=looks_like_test(rel),
            )
            self._doc_freq.update(tokens.keys())

        if self.files:
            self._avg_len = sum(f.length for f in self.files.values()) / len(self.files)
        log.debug("index.built", files=len(self.files))
        return self

    def _walk(self):
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in SOURCE_SUFFIXES:
                yield path

    def _bm25(self, query_tokens: list[str], doc: IndexedFile,
              k1: float = 1.5, b: float = 0.75) -> float:
        n_docs = len(self.files) or 1
        score = 0.0
        for token in set(query_tokens):
            freq = doc.tokens.get(token, 0)
            if not freq:
                continue
            df = self._doc_freq.get(token, 0) or 1
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            norm = freq * (k1 + 1) / (freq + k1 * (1 - b + b * doc.length / self._avg_len))
            score += idf * norm
        return score

    def rank_files(
        self, query: str, limit: int = 50, include_tests: bool = False
    ) -> list[tuple[str, float]]:
        """Top-`limit` candidate files for a free-text bug report."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Paths quoted verbatim in the report are a much stronger signal than
        # token overlap. Boost rather than filter: reporters often name the
        # wrong file, and a hard filter would make that unrecoverable.
        quoted = set(re.findall(r"[\w/\\-]+\.(?:py|js|ts|go|java|rb|rs|c|cpp|h)", query))

        scored: list[tuple[str, float]] = []
        for rel, doc in self.files.items():
            if doc.is_test and not include_tests:
                continue
            score = self._bm25(query_tokens, doc)
            if any(rel.endswith(q) or q in rel for q in quoted):
                score += 25.0
            if score > 0:
                scored.append((rel, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def read(self, rel_path: str) -> str | None:
        doc = self.files.get(rel_path)
        return doc.text if doc else None

    def resolve(self, rel_path: str) -> str | None:
        """Map a possibly-partial path emitted by the model to a real file."""
        if rel_path in self.files:
            return rel_path
        matches = [p for p in self.files if p.endswith(rel_path.lstrip("./"))]
        return matches[0] if len(matches) == 1 else (matches[0] if matches else None)

    def chunks_for(self, query: str, paths: list[str], max_chars: int = 60_000
                   ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for rel in paths:
            text = self.read(rel)
            if text is None:
                continue
            chunks.extend(chunk_file(rel, text))

        query_tokens = set(tokenize(query))
        for chunk in chunks:
            overlap = query_tokens & set(tokenize(chunk.text))
            chunk.score = len(overlap)

        chunks.sort(key=lambda c: c.score, reverse=True)
        selected: list[CodeChunk] = []
        budget = max_chars
        for chunk in chunks:
            if len(chunk.text) > budget:
                continue
            budget -= len(chunk.text)
            selected.append(chunk)
        selected.sort(key=lambda c: (c.path, c.start_line))
        return selected


def looks_like_test(rel: str) -> bool:
    lower = rel.lower()
    padded = f"/{lower}"
    return (
        "/tests/" in padded or "/test/" in padded or "/spec/" in padded
        or "/__tests__/" in padded or "/test_" in padded
        or lower.endswith(("_test.py", "_test.go", ".test.js", ".test.ts",
                           ".spec.js", ".spec.ts", "_spec.rb"))
    )


# --- AST-aware chunking ---------------------------------------------------


def chunk_file(rel: str, text: str) -> list[CodeChunk]:
    """Split a file into semantically whole units.

    A fixed-size window routinely cuts a function in half, and half a function
    is worse than useless in a patch prompt. Python gets real `ast` nodes;
    other languages fall back to an overlapping window.
    """
    if rel.endswith(".py"):
        chunks = _chunk_python(rel, text)
        if chunks:
            return chunks
    return _chunk_by_window(rel, text)


def _chunk_python(rel: str, text: str) -> list[CodeChunk]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    chunks: list[CodeChunk] = []

    def emit(node: ast.AST, qualname: str) -> None:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start) or start
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, min(d.lineno for d in decorators))
        if end - start > _MAX_CHUNK_LINES:
            end = start + _MAX_CHUNK_LINES
        chunks.append(CodeChunk(rel, start, end, "\n".join(lines[start - 1 : end]),
                                symbol=qualname))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            emit(node, node.name)
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    emit(child, f"{node.name}.{child.name}")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            emit(node, node.name)

    head_end = min(len(lines), 40)
    if head_end:
        chunks.append(CodeChunk(rel, 1, head_end, "\n".join(lines[:head_end]),
                                symbol="<module>"))
    return chunks


def _chunk_by_window(rel: str, text: str, window: int = 90, overlap: int = 15
                     ) -> list[CodeChunk]:
    lines = text.splitlines()
    chunks: list[CodeChunk] = []
    step = max(window - overlap, 1)
    for start in range(0, len(lines), step):
        end = min(start + window, len(lines))
        body = "\n".join(lines[start:end])
        if body.strip():
            chunks.append(CodeChunk(rel, start + 1, end, body))
        if end == len(lines):
            break
    return chunks

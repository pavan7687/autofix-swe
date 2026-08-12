"""Prompt construction — shared by data building, training, sampling and eval.

This module exists as a single top-level file for one reason: **train/inference
skew is the most common silent failure in applied LLM work.** If the string
format used to build a training example differs by even a stray newline from
the one used at inference, the model is being asked a question it was never
trained on, and the loss curve will look fine while the eval collapses.

Every consumer therefore calls the same three functions here. There is no
second place in this codebase where a prompt is assembled.
"""
from __future__ import annotations

from autofix.models import EditingExample, RetrievalExample

# --- retrieval (reranker) -------------------------------------------------

RERANK_SYSTEM = (
    "You are a code retrieval model. Given a bug report and a list of candidate "
    "file paths from a repository, identify which files must be modified to fix "
    "the bug.\n"
    "Answer with the file paths only, one per line, most likely first. "
    "Output at most 5 paths and nothing else."
)


def rerank_user(problem_statement: str, candidates: list[str]) -> str:
    listing = "\n".join(f"{i + 1}. {path}" for i, path in enumerate(candidates))
    return (
        f"# Bug report\n{_clip(problem_statement, 6000)}\n\n"
        f"# Candidate files\n{listing}\n\n"
        "Which of these files must be modified?"
    )


def rerank_target(gold_files: list[str]) -> str:
    return "\n".join(gold_files[:5])


def rerank_messages(example: RetrievalExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": rerank_user(example.problem_statement, example.candidates)},
        {"role": "assistant", "content": rerank_target(example.gold_files)},
    ]


# --- editing --------------------------------------------------------------

EDIT_SYSTEM = (
    "You are an expert software engineer. Given a bug report and the relevant "
    "source files, produce a minimal patch that fixes the bug.\n\n"
    "Rules:\n"
    "1. Output a unified diff and nothing else.\n"
    "2. Change the fewest lines possible. Do not reformat or refactor.\n"
    "3. Context lines must match the given source exactly, including indentation.\n"
    "4. Never modify test files. Never add dependencies.\n\n"
    "Format your answer as:\n"
    "<reasoning>\nBrief root-cause analysis.\n</reasoning>\n"
    "<patch>\n```diff\ndiff --git a/path b/path\n--- a/path\n+++ b/path\n"
    "@@ -L,C +L,C @@\n context\n-removed\n+added\n```\n</patch>"
)


def edit_user(problem_statement: str, file_contents: dict[str, str]) -> str:
    blocks = []
    for path, content in file_contents.items():
        blocks.append(f"### {path}\n```\n{content}\n```")
    return (
        f"# Bug report\n{_clip(problem_statement, 8000)}\n\n"
        f"# Source files\n" + "\n\n".join(blocks) + "\n\nProduce the patch."
    )


def edit_target(patch: str, reasoning: str = "") -> str:
    thought = reasoning.strip() or "Applying the minimal change that resolves the reported failure."
    return (
        f"<reasoning>\n{thought}\n</reasoning>\n"
        f"<patch>\n```diff\n{patch.strip()}\n```\n</patch>"
    )


def edit_messages(example: EditingExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EDIT_SYSTEM},
        {"role": "user", "content": edit_user(example.problem_statement, example.file_contents)},
        {"role": "assistant", "content": edit_target(example.patch, example.reasoning)},
    ]


def edit_inference_messages(
    problem_statement: str, file_contents: dict[str, str]
) -> list[dict[str, str]]:
    """Identical to `edit_messages` minus the answer. Same builders, no skew."""
    return [
        {"role": "system", "content": EDIT_SYSTEM},
        {"role": "user", "content": edit_user(problem_statement, file_contents)},
    ]


def rerank_inference_messages(
    problem_statement: str, candidates: list[str]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": rerank_user(problem_statement, candidates)},
    ]


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def estimate_tokens(text: str) -> int:
    """Cheap length proxy for dataset filtering.

    Real tokenisation of a 100K-instance corpus is slow and needs the tokenizer
    loaded; for deciding whether an example is grossly too long, chars/3.5 is
    within a few percent for code and costs nothing.
    """
    return int(len(text) / 3.5)

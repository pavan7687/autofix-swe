"""Parsing model responses into structured payloads.

Split out from `llm.py` deliberately: these are pure string functions with no
SDK dependency, no network, and no configuration, so they can be exercised
directly by an external test suite without an API key installed. Response
parsing is where a model's formatting drift turns into a production bug, so it
is the part most worth testing in isolation.

The parsers are strict on purpose. `extract_diff` returning None because the
model wrapped prose in a ```diff fence is a *good* failure — it routes into the
repair loop with a clear reason, rather than handing `git apply` garbage.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_DIFF_STARTS = ("diff --git", "--- ", "Index:")


def extract_block(text: str, tag: str) -> str | None:
    """Pull the contents of <tag>...</tag> from a model response."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_fenced(text: str) -> str | None:
    """Pull the first fenced code block."""
    match = _FENCE.search(text)
    return match.group(1).strip() if match else None


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object, tolerating tags, fences and surrounding prose."""
    candidate = extract_block(text, "json") or extract_fenced(text) or text
    candidate = candidate.strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_diff(text: str) -> str | None:
    """Pull a unified diff, tolerating the model's fencing habits.

    Returns None unless the payload actually looks like a diff — a prose
    explanation in a ```diff fence must not reach `git apply`.
    """
    tagged = extract_block(text, "patch")
    block = (extract_fenced(tagged) or tagged) if tagged else extract_fenced(text)
    if not block:
        return None

    block = block.strip()
    if not block.startswith(_DIFF_STARTS):
        return None
    if "@@" not in block:
        return None
    return block if block.endswith("\n") else block + "\n"

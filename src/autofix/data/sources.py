"""Loading raw issue-resolution instances from public corpora.

Each loader normalises a differently-shaped dataset into the same `Instance`.
Sources are deliberately plural: no single corpus is large, clean and diverse
enough on its own, and mixing them is what produces a training set that
generalises past one project's coding style.

Gold-file extraction happens here rather than downstream, because it is the
label for the retrieval task and is derived from the patch itself — the files a
patch touches *are* the files that needed changing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from autofix.logging_conf import get_logger
from autofix.models import Instance

log = get_logger(__name__)

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)", re.MULTILINE)
_MINUS_HEADER = re.compile(r"^--- a/(?P<path>\S+)", re.MULTILINE)


def files_in_patch(patch: str) -> list[str]:
    """Paths a unified diff touches, in order, de-duplicated."""
    paths = [m.group("b") for m in _DIFF_HEADER.finditer(patch or "")]
    if not paths:
        paths = [m.group("path") for m in _MINUS_HEADER.finditer(patch or "")]
    seen: dict[str, None] = {}
    for path in paths:
        if path != "/dev/null":
            seen.setdefault(path, None)
    return list(seen)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """How to pull one HuggingFace dataset into our schema."""

    name: str
    hf_id: str
    splits: tuple[str, ...] = ("train",)
    note: str = ""


# Ordered roughly by signal quality. All are public and permissively licensed;
# verify each licence before redistributing anything derived from them.
SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("swebench_train", "princeton-nlp/SWE-bench", ("train",),
               "Original corpus. Same construction as the benchmark, so it is "
               "the closest match to the eval distribution."),
    SourceSpec("swegym", "SWE-Gym/SWE-Gym", ("train",),
               "Executable environments, useful later for rejection sampling."),
    SourceSpec("swefixer", "internlm/SWE-Fixer-Train-110K", ("train",),
               "110K filtered instances released with the SWE-Fixer paper."),
    SourceSpec("r2e_gym", "R2E-Gym/R2E-Gym-Subset", ("train",),
               "Synthetically grown tasks with verified tests."),
)

# Each corpus names the same fields differently, and a corpus that yields zero
# instances is almost always an alias miss rather than an empty dataset. The
# loader logs the available column names when it finds nothing, so a new source
# can be wired up without guessing.
_FIELD_ALIASES = {
    "instance_id": ("instance_id", "id", "docker_image", "problem_id"),
    "repo": ("repo", "repo_name", "repository"),
    "base_commit": ("base_commit", "commit", "base_sha", "parent_commit"),
    "problem_statement": (
        "problem_statement", "issue", "text", "problem", "prompt",
        "problem_description", "issue_text",
    ),
    "patch": ("patch", "gold_patch", "solution", "fix_patch", "model_patch",
              "golden_patch", "diff"),
    "test_patch": ("test_patch", "tests_patch", "test_diff"),
    "environment_setup_commit": ("environment_setup_commit", "env_commit"),
}


def _pick(row: dict, key: str) -> str:
    for alias in _FIELD_ALIASES.get(key, (key,)):
        value = row.get(alias)
        if isinstance(value, str) and value:
            return value
    return ""


def _as_list(row: dict, key: str) -> list[str]:
    import json

    value = row.get(key)
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
            return [str(v) for v in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def load_source(spec: SourceSpec, limit: int | None = None) -> list[Instance]:
    from datasets import load_dataset

    instances: list[Instance] = []
    for split in spec.splits:
        try:
            ds = load_dataset(spec.hf_id, split=split)
        except Exception as exc:  # noqa: BLE001
            log.warning("source.unavailable", source=spec.name, split=split,
                        error=str(exc)[:200])
            continue

        skipped_no_patch = 0
        for row in ds:
            patch = _pick(row, "patch")
            if not patch.strip():
                skipped_no_patch += 1
                continue
            inst = Instance(
                instance_id=_pick(row, "instance_id") or f"{spec.name}-{len(instances)}",
                repo=_pick(row, "repo"),
                base_commit=_pick(row, "base_commit"),
                problem_statement=_pick(row, "problem_statement"),
                patch=patch,
                test_patch=_pick(row, "test_patch"),
                fail_to_pass=_as_list(row, "FAIL_TO_PASS") or _as_list(row, "fail_to_pass"),
                pass_to_pass=_as_list(row, "PASS_TO_PASS") or _as_list(row, "pass_to_pass"),
                environment_setup_commit=_pick(row, "environment_setup_commit"),
                source=spec.name,
                gold_files=files_in_patch(patch),
            )
            instances.append(inst)
            if limit and len(instances) >= limit:
                break
        if limit and len(instances) >= limit:
            break

    if not instances:
        # Diagnose rather than silently contribute nothing.
        try:
            columns = list(load_dataset(spec.hf_id, split=spec.splits[0]).features)
        except Exception:  # noqa: BLE001
            columns = []
        log.warning(
            "source.empty", source=spec.name,
            note="no usable instances - likely a field-name mismatch",
            available_columns=columns[:25],
        )
    log.info("source.loaded", source=spec.name, instances=len(instances))
    return instances


def load_all(names: list[str] | None = None, limit_per_source: int | None = None
             ) -> list[Instance]:
    wanted = set(names) if names else {s.name for s in SOURCES}
    out: list[Instance] = []
    for spec in SOURCES:
        if spec.name in wanted:
            out.extend(load_source(spec, limit_per_source))
    return out


def deduplicate(instances: list[Instance]) -> list[Instance]:
    """Collapse the same task appearing in several corpora.

    Keyed on repo@commit rather than instance_id, because the corpora above
    overlap heavily and each assigns its own ids. Earlier sources win, so the
    SOURCES ordering is a quality preference.
    """
    seen: dict[str, Instance] = {}
    for inst in instances:
        seen.setdefault(inst.contamination_key, inst)
    log.info("dedupe.done", before=len(instances), after=len(seen))
    return list(seen.values())

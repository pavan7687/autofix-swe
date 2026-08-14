"""Choosing a model size and context length that actually fit the GPU.

The naive assumption is "bigger model = better". For repository-level bug
fixing that is false in an important way: the task needs the buggy code *in the
prompt*, and a 32B at 4K tokens will often be shown a truncated file, while a
14B at 8K sees the whole function. Truncation is a hard failure; a slightly
weaker model is a soft one.

The table below therefore optimises for **the largest model that still leaves
room for a useful context**, not the largest model outright.

Rough QLoRA (4-bit NF4) memory, per model, at bf16 activations with gradient
checkpointing and flash-attention-2:

    weights  ≈ params × 0.55 GB/B
    LoRA state (r=32) ≈ 0.5-1.5 GB
    activations ≈ 1.2 GB per 1K tokens at batch 1 for a 32B

Numbers are conservative; measure with `--dry-run` before a long job.
"""
from __future__ import annotations

from dataclasses import dataclass

from autofix.logging_conf import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SizePlan:
    model_id: str
    label: str
    max_seq_len: int
    per_device_batch: int
    load_in_4bit: bool
    gradient_checkpointing: bool
    estimated_gb: float
    rationale: str


_EDITOR_IDS = {
    "7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}

# (min VRAM GB, size label, seq len, batch, est. peak GB, rationale)
_TABLE: tuple[tuple[int, str, int, int, float, str], ...] = (
    (78, "32b", 16384, 1, 62.0,
     "80GB (A100-80 / H100): the 32B fits with a 16K window, which holds most "
     "single-file bug contexts without truncation."),
    (44, "32b", 8192, 1, 38.0,
     "48GB-class card (A40 / L40 / L40S / A6000). Note the threshold is 44, not "
     "46: an A40 advertises 46068 MiB, which is 44.99 GiB, and a naive 46 GB "
     "boundary silently demotes it to the 14B profile. Marketing 'GB' and "
     "actual GiB differ by ~7%.\n"
     "     The 32B fits at 8K with roughly 7GB of headroom - workable but tight. "
     "If you hit OOM, set EDITOR_SIZE_OVERRIDE=14b rather than shortening the "
     "context; truncating the buggy function is the worse failure. These cards "
     "also have about a third of an A100's memory bandwidth, so expect 2-3x the "
     "wall-clock per epoch."),
    (38, "14b", 8192, 1, 33.0,
     "40GB: a 14B at 8K beats a 32B at 4K here. Bug fixing is context-bound, "
     "and truncating the buggy function is a hard failure that a larger model "
     "cannot recover from."),
    (22, "7b", 8192, 1, 19.0,
     "24GB: 7B at 8K is the largest configuration that keeps a usable window."),
    (14, "7b", 4096, 1, 13.0,
     "16GB: 7B at 4K. Expect truncation on large files; results will be a "
     "lower bound."),
)


def detect_vram_gb() -> float | None:
    """Total VRAM of the first visible CUDA device, or None if unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return round(props.total_memory / (1024**3), 1)


# Without 4-bit, weights occupy ~2 bytes/param instead of ~0.55, so the same
# card supports a far smaller model. These are the bf16 tiers.
# (min VRAM GB, size label, seq len, batch, est. peak GB, rationale)
_TABLE_BF16: tuple[tuple[int, str, int, int, float, str], ...] = (
    (78, "32b", 8192, 1, 74.0,
     "80GB in bf16: a 32B fits without quantisation, though with little room "
     "to spare."),
    (44, "7b", 8192, 1, 25.0,
     "48GB-class card with NO 4-bit support (old glibc: bitsandbytes wheels "
     "link against GLIBC_2.34 and will not load). In bf16 a 32B needs ~64GB of "
     "weights alone and cannot fit.\n"
     "     7B at 8K is preferred over 14B at 4K: bug fixing is context-bound, "
     "and truncating the buggy function is a harder failure than the loss from "
     "fewer parameters."),
    (22, "7b", 4096, 1, 18.0,
     "24GB in bf16: 7B at 4K. Expect truncation on large files."),
)


def plan_editor(
    vram_gb: float | None = None,
    size_override: str | None = None,
    seq_len_override: int | None = None,
    quantization: str = "auto",
) -> SizePlan:
    """Pick the editor configuration for the detected (or stated) GPU.

    `quantization="none"` selects the bf16 tiers, which are much more
    restrictive: 4-bit is what makes a 32B viable on a 48GB card at all.
    """
    use_4bit = quantization != "none"
    table = _TABLE if use_4bit else _TABLE_BF16
    if vram_gb is None:
        vram_gb = detect_vram_gb()

    if vram_gb is None:
        log.warning("sizing.no_gpu_detected", note="defaulting to the 40GB profile")
        vram_gb = 40.0

    if size_override:
        row = next((r for r in table if r[1] == size_override), None)
        if row is None:
            row = (0, size_override, 8192, 1, 0.0, "explicit override")
        chosen = row
        rationale = (
            f"EDITOR_SIZE_OVERRIDE={size_override} set explicitly; "
            f"auto-sizing bypassed. Verify it fits before a long run."
        )
    else:
        chosen = next((r for r in table if vram_gb >= r[0]), table[-1])
        rationale = chosen[5]

    _, label, seq_len, batch, est_gb, _ = chosen
    if seq_len_override:
        seq_len = seq_len_override
        rationale += f" MAX_SEQ_LEN_OVERRIDE={seq_len_override} applied."

    plan = SizePlan(
        model_id=_EDITOR_IDS.get(label, label),
        label=label,
        max_seq_len=seq_len,
        per_device_batch=batch,
        load_in_4bit=use_4bit,
        gradient_checkpointing=True,
        estimated_gb=est_gb,
        rationale=rationale,
    )
    log.info("sizing.editor", vram_gb=vram_gb, size=label, seq_len=seq_len,
             estimated_gb=est_gb, quantization="4bit" if use_4bit else "bf16")
    return plan


def _reranker_rationale(vram_gb: float, batch: int, checkpointing: bool) -> str:
    note = (
        "gradient checkpointing ON (limited VRAM)"
        if checkpointing
        else "gradient checkpointing OFF - ample VRAM, so it would only cost speed"
    )
    return f"1.5B in bf16 on {vram_gb:.0f}GB: batch {batch}, {note}."


def plan_reranker(model_id: str, vram_gb: float | None = None) -> SizePlan:
    """Configuration for the 1.5B file reranker.

    Its input is a bug report plus ~50 file *paths*, not file contents, so 4K
    tokens is comfortable and quantisation is unnecessary.

    Two settings matter for throughput, and both were wrong initially:

    * **No gradient checkpointing.** Checkpointing trades ~30-40% of speed to
      save activation memory. A 1.5B in bf16 is ~3GB of weights; on a 45GB card
      there is nothing to save, so it was pure loss. Enable it only if a batch
      genuinely does not fit.
    * **Batch 16, not 4.** With ~40GB free, a batch of 4 leaves the GPU mostly
      idle and pays fixed per-step overhead 4x more often than needed.

    Measured effect: ~54s per optimiser step became a small fraction of that,
    turning a 49-hour run into single-digit hours.
    """
    if vram_gb is None:
        vram_gb = detect_vram_gb() or 40.0

    # These are STARTING points, not limits: auto_find_batch_size halves them
    # on OOM. Erring low costs one retry; erring high costs a crash after the
    # model has loaded. A first attempt at batch 16 without checkpointing OOMed
    # on a 45GB A40 - activations for a 1.5B at 4K scale faster than the naive
    # per-parameter estimate suggests.
    if vram_gb >= 40:
        batch, checkpointing, est = 8, False, 30.0
    elif vram_gb >= 20:
        batch, checkpointing, est = 4, False, 16.0
    else:
        batch, checkpointing, est = 2, True, 10.0

    return SizePlan(
        model_id=model_id,
        label="reranker",
        max_seq_len=4096,
        per_device_batch=batch,
        load_in_4bit=False,
        gradient_checkpointing=checkpointing,
        estimated_gb=est,
        rationale=_reranker_rationale(vram_gb, batch, checkpointing),
    )


def render_plan(plan: SizePlan, vram_gb: float | None) -> str:
    return (
        f"  GPU detected     : {vram_gb or 'unknown'} GB\n"
        f"  Model            : {plan.model_id}\n"
        f"  Max sequence len : {plan.max_seq_len:,} tokens\n"
        f"  Per-device batch : {plan.per_device_batch}\n"
        f"  4-bit quantised  : {plan.load_in_4bit}\n"
        f"  Est. peak memory : ~{plan.estimated_gb:.0f} GB\n"
        f"  Why              : {plan.rationale}"
    )

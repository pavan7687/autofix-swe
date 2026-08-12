"""QLoRA fine-tuning entrypoint for both models.

    autofix-train --task editing     # the patch generator
    autofix-train --task retrieval   # the file reranker
    autofix-train --task editing --dry-run   # sizing + token stats, no GPU work

Design choices worth defending:

* **QLoRA over full fine-tuning.** A 32B full fine-tune needs hundreds of GB of
  optimiser state. 4-bit base weights with rank-32 adapters trains <1% of
  parameters and fits one A100, at a small and well-documented quality cost.
* **Adapters saved, not merged.** Merging bakes the adapter into 60GB of
  weights per experiment. Keeping adapters separate makes an ablation a 200MB
  artifact and lets vLLM hot-swap them.
* **Resume by default.** Cluster jobs get preempted. Every run checkpoints and
  restarts from the latest checkpoint unless told otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autofix.config import get_settings
from autofix.logging_conf import configure_logging, get_logger
from autofix.training.dataset import PadCollator, load_task_dataset
from autofix.training.sizing import (
    detect_vram_gb,
    plan_editor,
    plan_reranker,
    render_plan,
)

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autofix-train", description="QLoRA fine-tuning for retrieval and editing."
    )
    p.add_argument("--task", required=True, choices=["editing", "retrieval"])
    p.add_argument("--run-name", default=None, help="defaults to <task>-<config fingerprint>")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=-1, help="cap steps; -1 = full epochs")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the sizing plan and token statistics, then exit")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return p


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    settings.ensure_dirs()

    vram = detect_vram_gb()
    plan = (
        plan_editor(vram, settings.editor_size_override, settings.max_seq_len_override)
        if args.task == "editing"
        else plan_reranker(settings.reranker_base, vram)
    )

    run_name = args.run_name or f"{args.task}-{settings.fingerprint()}"
    out_dir = settings.model_root / run_name

    print(f"\n=== {args.task} ===")
    print(render_plan(plan, vram))
    print(f"  Output           : {out_dir}")

    data_dir = settings.editing_dataset if args.task == "editing" else settings.retrieval_dataset
    if not (data_dir / "train.jsonl").exists():
        raise SystemExit(f"No training data at {data_dir}. Run `autofix-data` first.")

    # Heavy imports live here so --help and --dry-run work without CUDA.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(plan.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_task_dataset(data_dir, "train", args.task, tokenizer, plan.max_seq_len)
    val_ds = load_task_dataset(data_dir, "validation", args.task, tokenizer, plan.max_seq_len)

    stats = train_ds.stats()
    print(f"\n  Train examples   : {len(train_ds):,}")
    print(f"  Val examples     : {len(val_ds):,}")
    print(f"  Token length     : p50={stats.get('p50_tokens')} "
          f"p90={stats.get('p90_tokens')} max={stats.get('max_tokens')}")
    truncated = stats.get("at_truncation_limit", 0)
    if truncated:
        print(f"  ⚠ {truncated}/{stats['sampled']} sampled examples hit the "
              f"{plan.max_seq_len} token limit and will be truncated.")

    if args.dry_run:
        print("\nDry run complete. No GPU work performed.")
        return

    _train(args, settings, plan, run_name, out_dir, tokenizer, train_ds, val_ds)


def _train(args, settings, plan, run_name, out_dir, tokenizer, train_ds, val_ds) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if plan.load_in_4bit
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        plan.model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if plan.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=plan.gradient_checkpointing
        )

    lora = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # All attention and MLP projections. Attention-only adapters
        # underperform noticeably on code generation.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} / {total:,} ({trainable / total:.2%})")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        num_train_epochs=args.epochs or settings.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=plan.per_device_batch,
        gradient_accumulation_steps=settings.grad_accum,
        gradient_checkpointing=plan.gradient_checkpointing,
        learning_rate=args.lr or settings.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=settings.warmup_ratio,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if len(val_ds) else "no",
        eval_steps=200,
        per_device_eval_batch_size=plan.per_device_batch,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to=["wandb"] if args.wandb else [],
        seed=settings.seed,
        optim="paged_adamw_8bit" if plan.load_in_4bit else "adamw_torch",
        group_by_length=True,   # fewer pad tokens per batch, real speedup
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) else None,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    resume = None
    if not args.no_resume and out_dir.exists():
        checkpoints = sorted(out_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[1]))
        if checkpoints:
            resume = str(checkpoints[-1])
            print(f"  Resuming from    : {resume}")

    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    _write_manifest(out_dir, args, settings, plan, train_ds, val_ds)
    print(f"\nAdapter saved to {out_dir / 'adapter'}")


def _write_manifest(out_dir: Path, args, settings, plan, train_ds, val_ds) -> None:
    """Record exactly what produced this checkpoint. Future-you will need it."""
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "base_model": plan.model_id,
                "max_seq_len": plan.max_seq_len,
                "load_in_4bit": plan.load_in_4bit,
                "lora": {
                    "r": settings.lora_r,
                    "alpha": settings.lora_alpha,
                    "dropout": settings.lora_dropout,
                },
                "learning_rate": args.lr or settings.learning_rate,
                "epochs": args.epochs or settings.num_epochs,
                "grad_accum": settings.grad_accum,
                "seed": settings.seed,
                "train_examples": len(train_ds),
                "val_examples": len(val_ds),
                "config_fingerprint": settings.fingerprint(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

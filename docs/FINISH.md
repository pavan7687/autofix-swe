# Finish line

What is done, what remains, and what "complete" means. Updated 2026-08-14.

## Done

| Component | Evidence |
|---|---|
| Dataset pipeline | 109,173 instances → 105,038 editing + 108,701 retrieval examples |
| Decontamination | 3 filters, case-normalised, **canary self-test 5/5** |
| Split hygiene | repo-grouped, 3.1% validation, verified stable under skew |
| Sandbox (reward fn) | namespace isolation, 8 checks passed incl. network blocked |
| Cluster environment | CUDA 12.8 matched to driver, offline weights, SLURM configured |
| Training loop | gradient flows, LoRA attached (2.34% params), steps advancing |

That is roughly 70% of the engineering, and all of the parts that are hard to
get right.

## Remaining

### 1. Train both models  *(~1 day wall clock, ~1h of your attention)*

```bash
sbatch scripts/train_reranker.sbatch
sbatch scripts/train_editor_multigpu.sbatch
bash scripts/watch.sh          # check the ETA it projects
```

If the projection exceeds the 48h partition limit, cut scope in `.env`:

```dotenv
NUM_EPOCHS=1
MAX_TRAIN_EXAMPLES=40000
```

Both checkpoint every 200 steps and resume automatically.

### 2. Serving environment  *(~30 min, login node)*

```bash
bash scripts/setup_serving_env.sh
```

Separate conda env because vLLM pins torch against the training stack.

### 3. Baseline evaluation  *(~4h)*  ← **do this before the trained model**

```bash
conda activate autofix-serve && bash scripts/serve_vllm.sh &
autofix-eval --tag baseline --editor-model editor-base --reranker-model reranker-base --limit 50
```

A near-zero baseline is the expected result and the control every other number
is measured against. If it is *not* near zero, the harness is wrong, not the
model.

### 4. Trained evaluation  *(~6h)*

```bash
autofix-eval --tag sft --limit 300
python -m autofix.eval.table
```

**This produces `artifacts/runs/RESULTS.md` — the deliverable.**

### 5. Rejection sampling  *(optional, ~2 days)*

```bash
autofix-sample --round 1 --instances 300 --k 4
autofix-train --task editing --run-name editing-rs1
autofix-eval --tag sft-rs1
```

## Two definitions of complete

**v1 — a complete, defensible project.** Steps 1–4. You have two trained models,
an honest ablation table against an untrained baseline, and a reproducible
pipeline. This is a real result and enough for a resume and an interview.
**Target: 1 week.**

**v2 — the distinctive version.** Adds step 5. The self-training loop with
execution-verified rewards is what separates this from a fine-tuning exercise.
**Target: 2–3 weeks.**

Ship v1 first. Always have something complete to show, then extend.

## Honest risk register

| Risk | Mitigation | Severity |
|---|---|---|
| Editor OOM at 8K | `auto_find_batch_size` halves and retries; else `MAX_SEQ_LEN_OVERRIDE=4096` | low |
| Training exceeds 48h | `NUM_EPOCHS=1`, `MAX_TRAIN_EXAMPLES` | low |
| Eval slower than expected | evaluate on 100 instances instead of 300 | medium |
| Resolve rate near zero | Expected for a 7B. **Report it honestly** — the delta over baseline is the result, not the absolute number | medium |

## What to claim

Not "I built a bot that fixes bugs." Rather:

> Trained a two-stage retrieval-and-editing pipeline (1.5B reranker + 7B editor,
> LoRA) on 105K decontaminated GitHub bug-fix instances, with an
> execution-verified sandbox as the reward signal. Measured resolve rate on
> SWE-bench Lite against an untrained baseline, with retrieval and generation
> ablations isolating each component's contribution.

Every clause there is something you can defend line by line — including the
parts that were forced by the hardware.

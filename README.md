# autofix-swe

Training open-weight LLMs to resolve real GitHub issues, with **execution-verified**
rejection sampling.

Two models are fine-tuned with QLoRA on a single A100 — a file **reranker** and a
patch **editor** — and improved by a self-training loop in which a Docker sandbox
runs each repository's own test suite to decide which of the model's own outputs
are worth learning from. No external LLM API is used anywhere in this project.

```
                    bug report
                         │
        ┌────────────────▼────────────────┐
        │ STAGE 1   BM25 over the repo    │   untrained, lexical
        │           → top-50 files        │
        │              ↓                  │
        │  ★ RERANKER (1.5B, QLoRA)       │   trained
        │           → top-5 files         │
        ├─────────────────────────────────┤
        │ STAGE 2                         │
        │  ★ EDITOR (7B/14B/32B, QLoRA)   │   trained
        │           → unified diff        │
        ├─────────────────────────────────┤
        │ STAGE 3   Docker sandbox        │   the reward function
        │           run the repo's tests  │
        │           → pass / fail         │
        └────────────────┬────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      ship the patch      keep as round-2 training data
                            (rejection sampling)
```

## Why this is not prompt engineering

The sandbox in Stage 3 turns a subjective task into a **verifiable** one. A
candidate patch scores 1 if the repository's own tests go green and 0 otherwise.
There is no reward model to over-optimise and no LLM-as-judge to fool — the
signal is a process exit code. That is what makes the self-training loop sound,
and it is the same signal used by [SWE-RL](https://arxiv.org/pdf/2502.18449) and
NVIDIA's Nemotron SWE pipeline.

## Repository layout

```
src/autofix/
├── prompting.py            ★ every prompt, built once — no train/inference skew
├── models.py                 shared schema: Instance, Candidate, examples
├── config.py                 all hyperparameters + a config fingerprint
│
├── data/                   ── dataset construction
│   ├── sources.py            load & normalise 4 public corpora
│   ├── decontaminate.py    ★ 3-level held-out protection
│   └── build.py              → retrieval/ and editing/ JSONL
│
├── training/               ── QLoRA fine-tuning
│   ├── sizing.py           ★ VRAM auto-detect → model size + context length
│   ├── dataset.py          ★ completion-only loss masking
│   └── run.py                Trainer, adapters, resume, manifest
│
├── rejection/              ── self-training loop
│   ├── verify.py           ★ the reward function (parse→scope→apply→test)
│   ├── workspace.py          per-instance checkout at base_commit
│   └── run.py                sample k, keep only what passes
│
├── serving/                ── inference
│   ├── client.py             vLLM OpenAI-compatible client, n>1 sampling
│   └── pipeline.py           BM25 → reranker → editor → sandbox
│
├── eval/                   ── the results
│   ├── metrics.py          ★ resolve rate, acc@k, BM25 recall, unbiased pass@k
│   ├── run.py                SWE-bench Lite harness
│   └── table.py              → RESULTS.md ablation table
│
├── sandbox/                  Docker runner, toolchain detection, output parsing
├── agent/                    BM25 index, AST chunking, git apply/reset
├── guardrails/scope.py       reject malformed patches before spending a test run
└── cli/main.py               autofix-fix: run the trained models on a checkout
```

★ = the parts most worth reading in an interview.

## Pipeline

```bash
# 0. environment (on the cluster)
bash scripts/setup_env.sh

# 1. data — decontamination runs BEFORE anything is written to disk
autofix-data --limit-per-source 20000

# 2. train both models
sbatch scripts/train_reranker.sbatch
sbatch scripts/train_editor.sbatch

# 3. serve
bash scripts/serve_vllm.sh &        # editor,   port 8000
bash scripts/serve_reranker.sh &    # reranker, port 8001

# 4. baseline FIRST — the untrained control
autofix-eval --tag baseline --editor-model editor-base --reranker-model reranker-base
autofix-eval --tag sft

# 5. rejection sampling, then retrain and re-evaluate
autofix-sample --round 1 --instances 2000 --k 8
autofix-train --task editing --run-name editing-rs1
autofix-eval --tag sft-rs1

# 6. the deliverable
python -m autofix.eval.table        # → artifacts/runs/RESULTS.md
```

## Results

Fill this in from `artifacts/runs/RESULTS.md`. Report the baseline row even when
it is embarrassing — a resolve rate with no untrained control is not a result.

| Configuration | Resolve rate | acc@1 | acc@3 | BM25 recall | Apply rate |
|---|---:|---:|---:|---:|---:|
| `baseline` (untrained) | | | | | |
| `sft` | | | | | |
| `sft-rs1` | | | | | |

Published reference points on SWE-bench Verified: SWE-Fixer 30.2% (7B retriever +
72B editor), NVIDIA Nemotron 37.2% at 8B, Llama3-SWE-RL-70B 41.0%. A single-A100
QLoRA run should expect **10–20% on SWE-bench Lite**. The interesting number is
the *delta* over the untrained baseline.

## Design decisions worth defending

| Decision | Why |
|---|---|
| **QLoRA, not full fine-tuning** | A 32B full fine-tune needs hundreds of GB of optimiser state. 4-bit base + rank-32 adapters trains <1% of parameters on one A100. |
| **Adapters kept unmerged** | An ablation becomes a 200MB artifact instead of a 60GB checkpoint, and vLLM can hot-swap them. |
| **Completion-only loss masking** | Loss over the prompt would spend capacity learning to reproduce bug reports the model is always *given*. |
| **Repo-grouped train/val split** | Two commits from one repo share nearly all their code; a random split makes validation loss optimistic. |
| **BM25 for stage 1, not embeddings** | No index to maintain across hundreds of repos at hundreds of commits, and bug reports quote identifiers verbatim — the regime where lexical wins. |
| **AST-aware chunking** | A fixed window cuts functions in half, and half a function is useless in a patch prompt. |
| **Execution as reward** | Unhackable and reproducible. Network is disabled during tests so the signal is deterministic. |
| **14B@8K over 32B@4K on a 40GB card** | Bug fixing is context-bound. Truncating the buggy function is a hard failure a bigger model cannot recover from. |

## Honest limitations

- Single-repository context; no cross-repo or dependency-level reasoning.
- Python, Node and Go toolchains; anything else aborts at detection rather than guessing.
- Training-set file contexts are reconstructed from patch context lines, not full
  checkouts. Cheap and scalable, but the model sees less surrounding code at
  training time than at inference. Effect is measurable in the acc@k gap.
- Rejection sampling is bounded by sandbox throughput, not GPU throughput.

## Documentation

- **[docs/METHOD.md](docs/METHOD.md)** — the full method: data, training, the
  self-training loop, and the threats to validity.
- **[docs/RUNBOOK.md](docs/RUNBOOK.md)** — cluster setup, SLURM, troubleshooting.

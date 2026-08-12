# Runbook

## Cluster setup (IITB, SLURM)

### Partitions, QOS and limits (discovered)

| Partition | GPUs | Nodes | Max wall | QOS | Use for |
|---|---|---|---|---|---|
| `a40` | gpu:4 | 19 | **4 days** | `a40` | **editor training** — longest wall clock, most nodes |
| `l40` | gpu:8 | 6 | 2 days | `l40` | reranker training — L40 is faster than A40 |
| `dgx` | gpu:8 | 9 | 6 days | `dgx` | check the GPU model; may be A100/H100 |
| `interactive` | gpu:8/4 | 3 | 4 hours | `interactive` | **the only partition allowing `srun --pty`** |
| `debug` | gpu:4 | 1 | **30 min** | `debug` | smoke tests only |

**Account:** `25m0803` (personal, FairShare 0.43) — preferred over `cccp`
(shared, FairShare 0.17).

### Four scheduler rules, each with its own cryptic error

| Rule | Error if violated |
|---|---|
| `--pty` only on `interactive` | `Interactive jobs are only allowed on partition 'interactive'` |
| GRES is untyped — use `--gres=gpu:1` | `Invalid generic resource specification` |
| **QOS name must equal the partition name** | `Invalid qos specification` |
| Time must be under the partition's max | `Requested time limit is invalid` |

The third is the non-obvious one: every partition declares
`AllowQos=<its own name>`, so a job on `a40` needs `--qos=a40`. A single
exported `SBATCH_QOS` cannot work across partitions, so each script carries its
own value explicitly.

### Why the editor trains on `a40`, not `l40`

The L40 is the faster card, but `l40` caps jobs at **2 days** and a 32B QLoRA
run on a 48GB card is estimated at 30–40 hours — uncomfortably close to the
limit. `a40` allows 4 days across 19 nodes instead of 6, so it queues sooner and
cannot be killed mid-epoch. The reranker is small and fast, so it takes the
quicker `l40` card where the 2-day cap is irrelevant.

### Step 0 — capability check (do this first)

```bash
srun --partition=interactive --gres=gpu:1 --time=00:10:00 --pty bash scripts/check_cluster.sh
```

If the interactive partition is full, use the batch fallback instead:

```bash
sbatch scripts/check_cluster.sbatch
cat cluster-check-*.out
```

Report back three things before going further: the GPU model and VRAM, whether
any container runtime exists, and where scratch storage is. They determine model
size, whether resolve-rate evaluation is possible at all, and where artifacts go.

### Step 1 — environment

```bash
srun --partition=interactive --gres=gpu:1 --time=01:00:00 --pty bash
cd autofix-swe
bash scripts/setup_env.sh          # venv + torch/cu121 + deps + flash-attn
cp .env.example .env
```

Run it on a **compute node**: flash-attn compiles against the GPU toolchain and
will fail on the login node.

## Docker on a shared cluster

The sandbox needs a container runtime. Most HPC clusters do **not** grant Docker
socket access — ask your admin whether Podman or Apptainer/Singularity is
available instead. If only Apptainer is available, `sandbox/runner.py` is the
one module needing a backend swap; its interface (`prepare`, `run_tests`) is
already isolated behind a class for exactly this reason.

Without a container runtime you can still run stages 1–3 (data, training,
localisation metrics), but not resolve-rate evaluation or rejection sampling.
Confirm this early — it is the single biggest execution risk in the plan.

### Step 2 — smoke test before any long job

```bash
sbatch scripts/smoke_test.sbatch
tail -f artifacts/runs/smoke-*.out
```

Twenty optimisation steps on the debug partition. If the loss does not move,
the loss mask is broken and nothing downstream matters. Never queue a 24-hour
job before this passes.

## Order of operations

```bash
# 1. Data (CPU node is fine, needs network for HuggingFace)
autofix-data --limit-per-source 20000
cat artifacts/data/stats.json          # check drop reasons before training

# 2. Sanity-check sizing without burning a GPU hour
autofix-train --task editing --dry-run

# 3. Train
sbatch scripts/train_reranker.sbatch   # ~4-8h
sbatch scripts/train_editor.sbatch     # ~12-24h
squeue -u $USER

# 4. Serve, then evaluate the BASELINE first
bash scripts/serve_vllm.sh &
autofix-eval --tag baseline --editor-model editor-base --reranker-model reranker-base --limit 50
autofix-eval --tag sft --limit 300

# 5. Rejection sampling
autofix-sample --round 1 --instances 2000 --k 8 --resume

# 6. Retrain on merged data, re-evaluate, build the table
autofix-train --task editing --run-name editing-rs1
autofix-eval --tag sft-rs1
python -m autofix.eval.table
```

Run the baseline on `--limit 50` first. It is fast, and if the baseline is not
near-zero something is wrong with your harness rather than your model.

## Submission failures

Run `bash scripts/slurm_debug.sh` first — it distinguishes the two failure modes
that look identical from the outside.

| Error | Meaning | Fix |
|---|---|---|
| `Invalid qos specification` | QOS missing or not permitted | QOS must equal the partition name |
| `AssocGrpSubmitJobsLimit` | Scheduler believes you have jobs submitted | Check section 1 vs section 6 of the debug script. If counters disagree with `squeue`, it is a stale counter — wait, or try `--account=cccp` |
| `Interactive jobs are only allowed...` | `--pty` used off the interactive partition | Use `sbatch`, or `--partition=interactive` |
| Queued forever, `squeue -p X` empty | Nodes are DOWN or DRAINED, not free | `sinfo -R` shows the reason; pick another partition |
| `Requested time limit is invalid` | Over the partition cap | `debug` is 30 min, `l40` 2 days, `a40` 4 days |

**Both accounts work.** If `25m0803` is limit-blocked, `cccp` is a valid
fallback with lower fair-share priority:

```bash
sbatch --account=cccp scripts/check_cluster.sbatch
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDA out of memory` at step 1 | Lower `MAX_SEQ_LEN_OVERRIDE`, or set `EDITOR_SIZE_OVERRIDE=14b`. Check `--dry-run` output for the truncation warning. |
| OOM after hundreds of steps | `group_by_length` batches long examples together. Reduce `GRAD_ACCUM` or cap sequence length. |
| `flash_attn` import error | Must be installed *after* torch, with `--no-build-isolation`. |
| Job preempted | Expected. `--requeue` plus automatic checkpoint resume handles it; verify a `checkpoint-*` directory exists. |
| Loss goes to 0 immediately | Loss masking is broken — every label is `-100`. Check `MaskedSFTDataset.__getitem__`. |
| Eval resolve rate exactly 0 | Almost always the sandbox, not the model. Run `autofix-fix --localise-only` to confirm retrieval works, then check Docker. |
| `test_patch would not apply` | The instance's `base_commit` is wrong or the repo history was rewritten. Skipped automatically; check the rate in `attempts.jsonl`. |
| vLLM `max_lora_rank` error | Must be ≥ `LORA_R` in `.env` (default 32). |

## What to record for the writeup

After every run, keep:

- `artifacts/data/stats.json` — dataset size and drop reasons
- `artifacts/models/*/manifest.json` — exact config per checkpoint
- `artifacts/data/rejection_round*/summary.json` — pass@k and rejection reasons
- `artifacts/runs/eval-*/results.json` — all metrics per configuration
- `artifacts/runs/RESULTS.md` — the ablation table

The rejection-sampling `summary.json` is the most interesting artifact in the
project: it shows the *distribution of failure reasons*, which is what tells you
whether to invest in retrieval, diff formatting, or reasoning next.

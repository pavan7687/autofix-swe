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

## Long-running commands: use tmux

An SSH drop kills every foreground process on the login node. Data building
takes tens of minutes and downloads tens of GB, so run it inside `tmux`:

```bash
tmux new -s data          # start a named session
# ... run the command ...
# detach with Ctrl+B then D; the work continues
tmux attach -t data       # reattach later, from anywhere
```

**tmux sessions are per-login-node.** If you started one on `login2` you must
SSH back to `login2` specifically to reattach — `tmux ls` on `login1` will not
show it. Note which node you are on:

```bash
hostname
```

Batch jobs (`sbatch`) are immune to this by design; only interactive work needs
tmux.

## Cache location

HuggingFace caches datasets in `~/.cache/huggingface`, which runs to tens of GB.
On this cluster `/home` is a 1.3 PB Lustre volume with hundreds of TB free, so
the default location is fine — no relocation needed.

Check your own quota before assuming that, since filesystem free space and a
per-user quota are different things:

```bash
lfs quota -u $USER /home 2>/dev/null || quota -s
du -sh ~/.cache/huggingface
```

Only if a quota is tight, redirect it **before** the first build (moving it
afterwards means re-downloading everything):

```bash
export HF_HOME=/scratch/$USER/hf-cache
```

## Fastest path (start here)

One job answers every remaining setup question and builds the environment:

```bash
sbatch scripts/bootstrap.sbatch
tail -f bootstrap-*.out
```

Read the summary block at the end. It reports the usable container runtime and
the attention backend; if flash-attn could not be built, export
`AUTOFIX_ATTN=sdpa` before training.

## Training in parallel

The reranker and editor are independent models trained on independent datasets,
and they use different partitions with separate QOS limits, so submit both at
once:

```bash
sbatch scripts/train_reranker.sbatch        # l40, ~8h
sbatch scripts/train_editor_multigpu.sbatch # a40 x4 GPUs, ~26h
```

Per-user caps allow this comfortably (`a40`: 3 running, `l40`: 4 running).

**The editor is the critical path**, so parallelising the reranker saves only
~8h. The real lever is the multi-GPU editor script: DDP across the 4 GPUs of one
a40 node cuts ~90h to ~26h, which fits in a single allocation instead of three
preempted restarts.

Use `scripts/train_editor.sbatch` (single GPU) only if 4-GPU nodes are queued.

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
| `AssocGrpSubmitJobsLimit` | The QOS has no limits of its own and inherits a saturated root-level cap | **Use `a40` or `l40`, not `debug`.** See below — this is not about your account |
| `Interactive jobs are only allowed...` | `--pty` used off the interactive partition | Use `sbatch`, or `--partition=interactive` |
| Queued forever, `squeue -p X` empty | Nodes are DOWN or DRAINED, not free | `sinfo -R` shows the reason; pick another partition |
| `Requested time limit is invalid` | Over the partition cap | `debug` is 30 min, `l40` 2 days, `a40` 4 days |

### Do not use the `debug` partition on this cluster

`sacctmgr show qos` reveals that `debug` defines **no QOS limits at all**:

```
Name          GrpSubmit MaxSubmit MaxJobsPU MaxSubmitPU
a40                 100         6         3           6
l40                  70         5         4           5
dgx                  90         5         4           5
interactive           5         2         2           2
debug        (all blank)
```

With nothing set, it falls back to the root association, which
`scontrol show assoc_mgr` reports as `GrpSubmitJobs=20(93)` — a cap of 20
against 93 jobs already submitted cluster-wide. Every `debug` submission
therefore fails with `AssocGrpSubmitJobsLimit`, for every user, regardless of
account. Switching accounts does not help.

**Use `a40` for short jobs instead.** It has `GrpSubmit=100`, allows 3 running
and 6 submitted jobs per user, and usually has idle nodes.

### Per-user concurrency caps

`MaxJobsPU` limits how many jobs you may have *running* at once:

| QOS | Running | Submitted |
|---|---|---|
| `a40` | 3 | 6 |
| `l40` | 4 | 5 |
| `dgx` | 4 | 5 |
| `interactive` | 2 | 2 |

Enough for the editor, the reranker and a sampling job in parallel.

### Two DGX nodes are drained

`cn11-dgx` and `cn13-dgx` have been `drain` with `Kill task failed` since
2026-08-07. `cn11-dgx` is one of only three `interactive` nodes, which is part
of why interactive allocations are slow. The other seven `dgx` nodes are fine.

## Installing anything: login node only

Compute nodes on this cluster have **no network**. Every `pip install`,
`huggingface-cli download` or `conda create` must run on the login node; jobs
then read the finished environment from shared home.

The failure mode is dangerous rather than obvious: pip reports
`from versions: none` (which reads like "this version does not exist" rather
than "I cannot reach the index"), the install silently does nothing, and any
test afterwards happily validates whatever was already installed. A batch job
that installs and then tests in the same script will therefore produce a
confident, wrong answer.

Split it: install on the login node, verify with a separate `sbatch`.

## No 4-bit quantisation on this cluster

bitsandbytes cannot work here, and the reason is a genuine dependency deadlock
rather than a version that needs finding:

| bitsandbytes | Blocker |
|---|---|
| >= 0.44 | wheels link `GLIBC_2.34`; RHEL 8 provides glibc **2.28** |
| <= 0.43 | imports `triton.ops`, removed in triton 3.x (torch 2.11 ships triton 3.6) |

Downgrading triton would require downgrading torch below the CUDA 12.8 build the
driver needs. There is no intersection.

**Consequence:** the editor trains in bf16, so weights cost ~2 bytes/param
instead of ~0.55. A 32B needs ~64GB of weights alone and does not fit on a 45GB
A40; the largest comfortable model is **7B at 8K context** (~25GB).

```dotenv
QUANTIZATION=none
EDITOR_SIZE_OVERRIDE=7b
```

`EDITOR_SIZE_OVERRIDE=14b` is worth attempting (~28GB weights, ~40GB peak) but
is tight enough to OOM mid-run. Test it with `--max-steps 50` before committing
to a multi-day job.

This is worth stating plainly in the write-up: the model size was chosen by the
hardware and its OS, not by preference, and a 7B places the work in the same
class as SWE-Fixer's 7B retriever and NVIDIA's 8B results.

## Two failures that point at the wrong thing

**`RuntimeError: No executable batch size found, reached zero.`**
Almost never a training-batch problem. `auto_find_batch_size` catches an OOM
anywhere in the loop - including evaluation - and responds by halving the
*training* batch. If the real culprit is eval retaining logits
(batch x seq x vocab), shrinking the training batch cannot help, so it halves to
zero and reports a message that names the wrong subsystem. Fixed here with
`prediction_loss_only=True`.

**`element 0 of tensors does not require grad`**
Gradient checkpointing with a fully-frozen base (LoRA). The embedding output has
`requires_grad=False`, so the recomputed segment has no autograd connection.
Fixed with `enable_input_require_grads()` plus `use_reentrant=False`. Only
appears on the bf16 path, because `prepare_model_for_kbit_training` handles it
in the 4-bit path.

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

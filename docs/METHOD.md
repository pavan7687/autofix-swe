# Method

## 1. Problem

Given a natural-language bug report and a repository at a specific commit,
produce a patch such that a designated set of failing tests passes and no
previously-passing test breaks. Correctness is decided by execution, not by
similarity to a reference patch.

Formally, a candidate patch `p` for instance `i` is **resolved** iff, in a clean
container at `base_commit` with `test_patch` applied:

```
∀ t ∈ FAIL_TO_PASS(i) :  t passes after p
∀ t ∈ PASS_TO_PASS(i) :  t still passes after p
```

Exact-match against the gold patch is deliberately *not* used as the primary
metric. Many correct fixes differ textually from the reference; grading on text
similarity would penalise them and reward memorisation.

## 2. Data

### Sources

Four public corpora, normalised into one `Instance` schema:

| Source | Contribution |
|---|---|
| `princeton-nlp/SWE-bench` (train split) | closest to the eval distribution |
| `SWE-Gym/SWE-Gym` | executable environments, needed for rejection sampling |
| `internlm/SWE-Fixer-Train-110K` | 110K filtered instances |
| `R2E-Gym/R2E-Gym-Subset` | synthetically grown tasks with verified tests |

They overlap heavily, so instances are de-duplicated on `repo@base_commit`
(not on instance id — each corpus assigns its own). Earlier sources win, making
the source ordering an explicit quality preference.

### Decontamination

Three independent filters, applied before any training example is written:

1. **instance id** — catches the obvious overlap between the public training
   corpora and the benchmarks.
2. **repo@base_commit** — catches the same underlying task appearing under
   different ids in different corpora.
3. **repository-level** (`--strict`) — drops every instance from any repository
   that appears in the eval set at all. Costs training data, but it is the only
   filter that answers "did your model just memorise `django/django`?"

All comparisons are case-normalised. Both halves of the key are lowercased,
because the corpora are inconsistent about SHA and repo-name casing and a leak
surviving on a casing technicality would invalidate every number downstream.

Default is repo+commit. If final numbers are close to a claimed baseline,
report the `--strict` figures instead.

### Two derived datasets

**Retrieval** — `(bug report, ~50 candidate paths) → which paths are buggy`.
Gold labels are the files the reference patch touches. Candidate lists are
synthesised by mixing gold files with distractor paths sampled from the *same
repository*, rather than by running BM25 over tens of thousands of repository
snapshots. This preserves the discrimination the reranker must learn at a tiny
fraction of the cost. **The eval harness runs true BM25**, so reported retrieval
numbers are unaffected by this shortcut.

**Editing** — `(bug report, pre-patch file contents) → unified diff`. File
contents are reconstructed from the diff's own context and removed lines, which
reproduce the file as it looked *before* the fix. Instances whose patch touches
a test file are dropped: learning to edit tests is learning to cheat.

### Splits

Grouped by repository, never by example. Two instances from one repo at nearby
commits share almost all their code, so an example-level split leaks and the
validation loss becomes optimistic.

## 3. Training

QLoRA (4-bit NF4 base, bf16 compute) with rank-32 adapters on all attention and
MLP projections. Attention-only adapters measurably underperform on code
generation.

**Completion-only loss masking.** Prompt tokens are set to `-100` so loss is
computed on the assistant turn alone. Without this the model spends capacity
learning to reproduce bug reports and source code — text it is always *given* at
inference and never asked to produce.

**Sizing.** VRAM is detected and a configuration chosen that keeps a usable
context window rather than maximising parameters:

| VRAM | Model | Context | Reasoning |
|---|---|---|---|
| 80 GB | 32B | 16K | fits with a window that holds most single-file contexts |
| 40 GB | 14B | 8K | a 14B@8K beats a 32B@4K — truncating the buggy function is a hard failure |
| 24 GB | 7B | 8K | largest config keeping a usable window |

Override with `EDITOR_SIZE_OVERRIDE` / `MAX_SEQ_LEN_OVERRIDE`.

Adapters are saved unmerged, so each ablation is a ~200MB artifact and vLLM can
hot-swap between them.

## 4. Rejection sampling (self-training)

For each training instance: sample k=8 patches at temperature 0.8, verify each,
keep only those that pass, retrain on the survivors.

The filter chain is ordered by cost, because each stage rejects candidates far
more cheaply than the next:

```
parse (µs) → scope check (ms) → git apply (10ms) → test suite (30-300s)
```

An early checkpoint fails 40–60% of samples before the expensive stage, which is
the difference between a sampling run that finishes overnight and one that does
not.

**Why this improves on the SFT checkpoint despite using the same model:**
sampling at temperature explores k solutions and only successes are kept, so the
round-2 distribution is conditioned on correctness. The model is trained on its
own best behaviour rather than its average behaviour.

Two safeguards: verified patches are de-duplicated per instance (ignoring hunk
offsets and whitespace), and capped per instance. Without them, eight identical
solutions to one easy bug would dominate the gradient and teach the model that
easy bugs are all that exist.

## 5. Evaluation

Four metrics, each answering a different question:

- **resolve rate** — the headline; execution-derived.
- **localisation acc@k** — did retrieval surface a truly-buggy file in the top k?
  Isolates the reranker from the editor.
- **BM25 recall@N** — the ceiling on acc@k. The reranker cannot recover a file
  stage 1 never surfaced, so this bounds the entire system and must be reported
  alongside acc@k or acc@k looks better than it is.
- **apply rate** — separates "cannot fix bugs" from "cannot emit a valid diff",
  which have completely different remedies.

`pass@k` uses the unbiased estimator (Chen et al., 2021), not "any of k
succeeded", which is biased upward when n > k.

### Required ablations

| Tag | What it isolates |
|---|---|
| `baseline` | untrained base model — proves the fine-tuning did the work |
| `sft` | supervised fine-tuning alone |
| `sft-rs1` | + one rejection-sampling round |
| `bm25-only` | reranker removed — isolates its contribution |
| `oracle-files` | gold files handed to the editor — upper bound if retrieval were perfect |

`oracle-files` is the most informative diagnostic: if it is much higher than
`sft`, the system is retrieval-bound and effort belongs in the reranker.

## 5a. Execution backend

The reward function needs to run a repository's test suite in isolation. Two
backends implement the same interface and are selected automatically:

**Docker** — a container per run. Preferred, but shared HPC clusters almost
never grant the daemon socket. On the target cluster `docker` is installed and
`docker ps` is refused, so this backend is unavailable.

**Local namespaces** — `unshare` user/PID/network namespaces plus POSIX
resource limits and a per-repository virtualenv. Requires no daemon, no root
and no image.

| Property | Docker | Local namespaces |
|---|---|---|
| Network isolation | `network_mode=none` | `unshare --net` |
| PID isolation | pid namespace | `unshare --pid --fork` |
| Memory cap | cgroup | `RLIMIT_AS` |
| Process cap | `pids_limit` | `RLIMIT_NPROC` |
| Wall-clock cap | enforced by runner | enforced by runner |
| Env isolation | fresh env | allowlist only |
| **Fresh root filesystem** | **yes** | **no** |

The last row is the real difference and is not glossed over: under the local
backend a test suite can read the host filesystem, including the user's home
directory. It cannot reach the network, see other processes, or exhaust host
memory — but it is not equivalent to a fresh container image.

For this project that trade is acceptable. The corpora are well-known
open-source repositories whose suites are executed by thousands of CI systems
daily. It would **not** be acceptable for a bot accepting arbitrary
repositories from strangers, which is what this codebase originally was.

Network isolation matters here for *correctness* as much as security: a test
that quietly reaches the internet makes the reward non-reproducible, and a
noisy reward is worse than a strict one.

The backend in use is recorded in every eval report — a resolve rate measured
under a different backend is not the same measurement.

## 6. Threats to validity

| Threat | Mitigation | Residual risk |
|---|---|---|
| Benchmark contamination | 3-level filter, case-normalised, applied pre-write | Base model may have seen these repos during *pretraining* — unfixable, applies to all published work, must be stated |
| Test-suite gaming | Patches touching test paths are rejected by the scope guard | A patch could still special-case an input to satisfy a weak test |
| Train/inference prompt skew | One `prompting.py`, verified byte-identical | — |
| Optimistic validation | Repo-grouped splits | — |
| Non-deterministic reward | Network disabled during test execution (both backends) | Genuinely flaky tests remain; the baseline run identifies them |
| Weaker isolation without Docker | Namespace backend documented above; repos are well-known OSS | A hostile test suite could read the filesystem. Not a concern for these corpora |
| Reconstructed contexts ≠ real files | Stated openly; eval uses real checkouts | Train/test distribution gap, visible in the acc@k vs resolve-rate gap |

The base-model-pretraining row is the one to raise *yourself* in an interview.
Every paper in this area has it, few state it, and volunteering it demonstrates
you understand what your numbers can and cannot support.

## References

- SWE-Fixer — [arXiv:2501.05040](https://arxiv.org/abs/2501.05040)
- SWE-RL — [arXiv:2502.18449](https://arxiv.org/pdf/2502.18449)
- SWE-Synth — [arXiv:2504.14757](https://arxiv.org/pdf/2504.14757)
- SWE-Dev — [arXiv:2506.07636](https://arxiv.org/pdf/2506.07636)

# GPTMini hard-recall feature ablation

Restore the historically successful model and AdamW first, commit that baseline,
then add one feature at a time. Stop at the first feature that loses the
baseline's fast 99% validation crossing. A finite-budget miss is a speed
regression, not a proof that the model can never generalize.

## Continuation: ordinary AdamW, architectural features

The next cohort keeps ordinary AdamW throughout and resumes the architectural
sequence after the failed optimizer screen. Each fresh candidate adds one
feature to the last passing model. The same seed 1, historical data streams,
50-update observations and 3,000-update 99% exact-sequence screen apply.
The passing control remains the recorded 2,900-update AdamW run. A missed
screen stops this sequence; it does not establish permanent inability to learn.

The first candidate replaces only the ReLU-squared MLP with the existing
SwiGLU implementation. It keeps every non-MLP parameter byte-identical to
the recorded historical initialization. The new MLP matrices receive fresh
Normal(0, 0.02) initialization after constructing the historical model; there
is no initialization from trained weights. Its parameter count is 1,249,048,
compared with 1,249,816 for the control. Shape changes therefore do not shift
the random initialization of unrelated embedding and attention matrices.

**SwiGLU passed:** first 99% at 2,300 updates, with 508/512 exact sequences
(99.22%) and 4,092/4,096 correct queries. Its endpoint, independent
latest-value replay and full history are retained in `results/`.
The public `GPT` now matches this passing candidate exactly; `GPTMini`
retains the original ReLU-squared control. The optimizer remains ordinary
AdamW.

The next candidate, `scaled-init`, changes only the initial attention and MLP
output matrices: multiply the parent's same Gaussian draws by
`1/sqrt(2*n_layer)`. All other tensors, including the MLP input matrices,
remain byte-identical. This isolates depth-scaled projection initialization
from residual scaling during the forward pass. It requires a passed SwiGLU
parent and verifies the parent's initial-parameter digest before training.

**Scaled initialization missed the fast screen.** It never reached 99% by
3,000 updates: its best exact accuracy was 481/512 (93.95%) at 2,800, and
its endpoint was 453/512 (88.48%). At its parent's 2,300-update crossing it
scored 89.26%. Eager replay exactly reproduces 4,036/4,096 correct endpoint
queries and 453 exact sequences. This is a finite speed regression in the
tested order; it does not establish inability to grok later, and interactions
with later normalization were not tested. Public GPT retains unscaled
initialization. Full results and audits are in `results/architecture_summary.json`.

The user then requested the next test while leaving scaled initialization out.
`embedding-norm` therefore branches from the passed **unscaled SwiGLU** parent,
not from the failed `scaled-init` model. Every parameter is byte-identical to
that parent, including the tied embedding/readout object. Only the embedding
output gains parameter-free native RMSNorm, using native dtype epsilon and
native backward. Other norm sites, attention, branch scaling, initialization,
optimizer and data remain unchanged. The separated radial RMS backward is
a later feature and is not included in this candidate. The same seed 1 and
3,000-update screen apply; another miss stops the sequence again.

**Embedding normalization missed the fast screen:** 375/512 exact sequences
(73.24%) and 3,935/4,096 correct queries (96.07%) at 3,000 updates. Its best
observation is that endpoint; it never reaches 99%. At the parent's
2,300-update crossing it scores only 9/512 exact sequences (1.76%). Eager
checkpoint replay reproduces every prediction count, and the independent
latest-value oracle, source hashes, optimizer steps and complete sampler
state all verify. All parameter bytes at initialization match the passed
SwiGLU parent; the failed initialization change is absent.

**Current decision:** stop again and retain public GPTMini + SwiGLU with
ordinary AdamW, raw embeddings and unscaled initialization (`b91fdf8`).
Embedding normalization is clearly learning by the endpoint; the result is
a regression under the declared fast criterion on one fresh seed, not a
claim of eventual impossibility. Effects of later normalization or different
training rates remain untested. Six focused isolation/gradient tests pass;
the preceding public-model promotion passed 830 domain/application tests
with 40 skipped. Ruff passes for the changed Python files.

| Ordinary AdamW, same seed and data | First 99% exact | Exact endpoint |
| --- | --- | --- |
| Historical ReLU-squared | 2,900 | 99.02% |
| SwiGLU | 2,300 | 99.22% |
| SwiGLU + scaled projection initialization | Miss by 3,000 | 88.48% (best 93.95%) |
| SwiGLU + embedding RMS, unscaled initialization | Miss by 3,000 | 73.24% |

## Requested continuation: Peri-RMSNorm with raw embeddings

The next requested test skips the failed embedding normalization and again
starts from the passed unscaled SwiGLU parent. `peri-norm` keeps each existing
branch-input RMSNorm and adds a parameter-free native RMSNorm after each
attention and MLP output projection, before residual addition:

```text
x <- x + RMS(attention(RMS(x)))
x <- x + RMS(MLP(RMS(x)))
```

This tests Peri-RMSNorm in the transformer branches. Embeddings stay raw;
projection initialization and forward residual coefficients stay unscaled.
Existing norm sites and all parameter values remain byte-identical to the
parent; added norms use native dtype epsilon and native backward. Learned
temperatures, ordinary XSA and the rest of the passing architecture remain
unchanged. The separate RMS backward repair and first-position/prefix
stabilization are not included. Interactions with skipped or later features
are outside this single-feature screen.

Every added norm wraps an existing projection and retains its exact parameter
objects. Construction draws no random numbers, and the initial parameter
digest and RNG state match the parent. Eight focused tests verify isolation,
independent post-projection RMS equations and finite full-stack gradients.
The fixed ordinary-AdamW recipe, seed 1 and 3,000-update 99% exact-sequence
criterion apply again; another miss stops the sequence.

## Requested switch to repaired RMS backward

The active screen is now `peri-norm-repaired`, using the existing FP64
radial/tangent backward from `domain/model/rms_norm.py` at **all 25 RMS
sites**: twelve branch inputs, twelve branch outputs and the final norm.
Every existing epsilon is preserved: 1e-5 at branch inputs/final readout,
and native dtype epsilon at the new outputs. Forward outputs, all initial
parameter bytes and construction RNG state match native Peri-RMSNorm exactly.
The complete compiled validation metrics at initialization match too.
Ten focused tests include all-site radial derivatives and native/repaired
forward equality.

The native-backward `peri-norm` run was interrupted at the user's request
to switch immediately. Its last completed observation was update 2,050:
0/512 exact sequences and 7.28% token accuracy. This is **not a completed
failure** at the declared 3,000-update horizon. Its full observed history,
protocol and interruption record are retained in `results/`; no endpoint
weights or optimizer state were saved. It is not used as a passed parent.

The repaired case starts fresh from the passing SwiGLU initialization under
the same ordinary AdamW recipe. Raw embeddings, unscaled initialization and
forward residual coefficient 1 remain fixed. This is the explicitly requested
Peri-plus-backward-repair combination; no conclusions about the numerical
repair follow from the interrupted native run. The remaining stability,
attention and epsilon changes remain separate. The 3,000-update screen and
first-99% stopping rule are unchanged.

```bash
.venv/bin/python -m research.recall_ablation.run --stage swiglu --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage scaled-init --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage embedding-norm --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage peri-norm --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage peri-norm-repaired --seeds 1 --max-updates 3000
```

After the Peri-RMSNorm screen, the untested remaining sequence is
residual scaling, Q/K normalization, removal of learned temperatures,
centered XSA and its clipping threshold, fused SDPA, first-position and
repeated-prefix stabilization, native dtype epsilon, and the separated RMS
backward. Normalization and temperature removal remain separate changes.
Later features are implemented and trained only after their predecessor passes.
The original optimizer-only evidence below remains unchanged. Its stop decision
is historical; the requested continuations above explicitly skip failed features.

## Completed result: stop at Schedule-Free AdamW

The restored baseline is committed as `b7b8e89`. In the fresh CPU comparison,
changing only the optimizer loses the rapid hard-recall transition:

| Unchanged GPTMini, seed 1 | First 99% exact validation | Endpoint exact sequences | Endpoint query accuracy |
| --- | --- | --- | --- |
| AdamW | 2,900 updates | 507/512 = 99.02% | 4,091/4,096 = 99.88% |
| Schedule-Free AdamW | Not reached by 3,000 | 0/512 = 0% | 411/4,096 = 10.03% |

At the same 2,900 updates, Schedule-Free also has 0/512 exact sequences.
Both arms start from byte-identical parameters and use identical splits,
batch order, learning rates, warmup, betas and decay. Only the optimizer
changes. The baseline's independent latest-value oracle checks every query;
it answers 2,712/2,714 queries of overwritten keys correctly and predicts
no superseded values. Eager checkpoint replay reproduces the compiled
endpoint prediction counts. Optimizer steps, epoch tails, sampler RNG state,
first-crossing observations and source snapshot hashes are verified.

**Original decision at `5bec856`:** retain GPTMini + AdamW and stop adding features. SwiGLU,
normalization changes, residual scaling, temperature removal and the remaining
numerical changes have not been tested in this ablation. The result is one
fresh matched seed and a finite speed screen; it does not establish permanent
inability to learn or explain all effects of the newer architecture.

[`results/summary.json`](results/summary.json) records the decision, matching
conditions, test coverage and limitations. The same folder retains both full
50-update histories, complete result metadata and endpoint replay audits.
Raw checkpoints and exact executed source snapshots remain under
`runs/recall-ablation/`. The replay script is retained there as
`audit_replay.py`. The baseline started before its restoration commit was
created; its metadata therefore records parent commit `df1e9f7` plus verified
snapshots of the restored source. Training trajectories need not reproduce
historical GPU/PyTorch crossing counts bit for bit.

Validation: 828 domain/application tests passed, 40 skipped; changed Python
files pass Ruff. The whole-repository suite could not collect optional
corpus/leaderboard tests because dependencies including `pyarrow` are absent.

The frozen `GPTMini` implements the original baseline: PreNorm, native parameter-free
RMSNorm with epsilon 1e-5, unit Q/K with learned head temperatures, RoPE,
ordinary XSA, ReLU squared with hidden width 4d, tied embedding/readout,
and one Normal(0, 0.02) initialization per distinct matrix. Default
`train`/`DiffGPT.train` uses native AdamW, betas (0.9, 0.98), decay 0.1,
rate 0.0003 and the historical 50-update warmup starting at zero.
The public `GPT` additionally includes the passing SwiGLU change described above.

The former production architecture remains available as `NormalizedGPT`
only through its explicit module. Existing numerical and learned-division
regressions, and research runners whose checkpoints require that architecture,
continue to use it. No constructor switch changes the canonical model.

## Frozen task and decision

| Setting | Value |
| --- | --- |
| Model | width 128, depth 6, four heads; 1,249,816 parameters |
| Task | 256 keys and 256 values; 16 writes, eight overwrites, eight queries |
| Data | historical streams, data seed 1; 20,000 training and 512 validation |
| Length | uniform 48–64; right-padded to 64 for compiled execution |
| Vocabulary | historical 548 tokens, without padding |
| Batch order | shuffled epochs; generator seed = model seed + 10,000 |
| Training batch | 64, retaining the 32-example epoch tail |
| Evaluation | all 512 validation rows, every 50 updates |
| Pass | at least 99% exact sequences, all eight latest-value answers correct |
| Baseline safety budget | 4,800 updates; first crossing retained |
| First-feature screen | 3,000 updates, covering all historical 2,150–2,950 crossings |
| Test set | never constructed or read by the screen |
| Execution | CPU TorchInductor, FP32; four threads |

The quick fresh comparison uses seed 1, with the all-three-seed historical
AdamW evidence retained in `historical_baseline.json`. It establishes a matched
seed result, not a fresh three-seed replication. The initial parameters and
training/validation fingerprints must match before the candidate can run.
Every 50-update observation is retained; there is no selection by training loss
or restoration of a best checkpoint.

1. Restore GPTMini and verify its initialization, logits and gradients against
   an independent fixture generated by archived `gpt_mini.py`; validate native
   AdamW updates, inference, checkpointing and the existing regressions. Commit.
2. Reproduce a fresh GPTMini + AdamW crossing, then change only the optimizer
   to the repository's Schedule-Free AdamW. Apply exactly the same external
   warmup to both arms; disable Schedule-Free's internal warmup. If the
   candidate misses 99% within 3,000 updates, stop and retain GPTMini + AdamW.
   This cap was chosen before candidate training, to keep the experiment short;
   it must be at least the fresh baseline's crossing time.
3. Only if the optimizer passes, introduce the newer architectural changes
   separately: SwiGLU, initialization, embedding normalization, output
   normalization, residual scaling, Q/K normalization and temperature removal,
   centered XSA, repeated-prefix stabilization and the separated RMS backward.
   Later changes are not tested after the first failure.

## Running

From the repository root:

```bash
.venv/bin/python -m research.recall_ablation.run --stage baseline --seeds 1
.venv/bin/python -m research.recall_ablation.run --stage schedule-free --seeds 1 --max-updates 3000
```

The second command requires a passed matching baseline. A run refuses to
overwrite an existing seed directory. Results live under ignored
`runs/recall-ablation/<stage>/seed1/`: protocol and source snapshots,
`history.jsonl`, `result.json` and the complete `endpoint.pt`. Timing includes
compilation and evaluation. The 50-update speed check under
`runs/recall-speedcheck` is only a runtime measurement, not a failed learning
experiment.

## Historical evidence

`historical_baseline.json` preserves the lab's committed record of AdamW
crossings at 2,950 / 2,150 / 2,750 updates. `historical_schedule_free.json`
extracts the later unchanged-GPTMini controls: seed 0 crossed at 12,200;
seed 1 was still at 0% exact sequences after 100,000. It records the original
source hashes and the distinction between historical observations and this
fresh CPU comparison. Those long runs motivate testing the optimizer first;
they do not replace the matched fresh screen.

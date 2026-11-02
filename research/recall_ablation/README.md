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

```bash
.venv/bin/python -m research.recall_ablation.run --stage swiglu --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage scaled-init --seeds 1 --max-updates 3000
```

If it passes, the remaining intended sequence is depth-scaled projection
initialization, embedding normalization, branch-output normalization,
residual scaling, Q/K normalization, removal of learned temperatures,
centered XSA and its clipping threshold, fused SDPA, first-position and
repeated-prefix stabilization, native dtype epsilon, and the separated RMS
backward. Normalization and temperature removal remain separate changes.
Later features are implemented and trained only after their predecessor passes.
The original optimizer-only evidence below remains unchanged.

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

**Decision:** retain GPTMini + AdamW and stop adding features. SwiGLU,
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

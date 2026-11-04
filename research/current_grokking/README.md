# Current GPT grokking sweep

The task catalog is [`../grokking_task_inventory.json`](../grokking_task_inventory.json).
This experiment trains fresh instances of the public `diff_gpt.domain.model.gpt.GPT`
on 165 declared cases with ordinary `torch.optim.AdamW`. It does not change the
model to repair difficult tasks. One model seed (1) is the initial screen.
The user subsequently skipped the nine unfinished modular-composition variants;
three completed composition results and the interrupted run's checkpoint remain.
Other families retain their declared budgets.

| Family | Cases |
| --- | ---: |
| Arithmetic functions, primes, powered division | 54 |
| Synthetic algorithmic suite and random-LM control | 38 |
| Series, formulas, noise and shuffled-time controls | 22 |
| Basis depth, recall, parity, noisy parity and L1–L6 | 14 |
| Modular composition chains | 12 |
| Associative division/register operations | 8 |
| Temporal division and Fibonacci, including MASH-half | 7 |
| New full-GPT versions of supplied-feature probes | 6 |
| Differenced division equations | 2 |
| Fashion-MNIST GPT and Tiny Shakespeare GPT | 2 |

`protocol.py` declares sizes, budgets, metrics and targets. The executed settings,
source files, data fingerprints and initial parameter hash are retained for each
case under `runs/current-grokking/<case>/seed1/`. The frozen model reference names
commit `01f18f86787511279fe52fe41883495fce0e3105` and checks all 15 model-source
hashes before training. Historical experiment code supplies data and scorers;
its historical models are never instantiated by the training harness.

The optimizer uses betas `(0.9, 0.98)`, epsilon `1e-8`, weight decay `0.1`, and a
50-update linear warmup whose first update has learning rate zero. Native fused
AdamW and compiled CUDA graphs accelerate the GPU runs. Capture warmup restores
parameters, optimizer moments and RNG; it consumes no scientific updates. Each
checkpoint retains the shuffled-epoch sampler, both RNG states, optimizer moments
and final weights. Full losses and scores are observed every declared interval;
series use one epoch. There is no best-checkpoint restoration.

“Passed” means the declared validation target was observed at least once.
`final_target_met` records whether it survived the follow-up. A budget miss is
bounded evidence, not proof that eventual grokking is impossible. Operational
delayed generalization requires at least two observations with fitted training
data and near-baseline validation before reaching target. Raw first-fit and
first-target updates remain available. Success without this plateau is reported
separately. Continuous forecasting, language modeling and negative controls do
not share an exact-sequence grokking interpretation.

The new numeric GPT probes serialize supplied vectors and generate entire answer
vectors freely. Their primary score is decoded MSE against the unquantized
teacher; missing or invalid components receive an explicit squared-error penalty.
They test the teacher functions with all GPT parameters learned, rather than
reproducing query-only or gate-only optimization. The new image experiment uses
196 quantized 2×2 patch means and a class token, with a fixed 50k/10k split of
official training images and the official test set held out. It is a trained GPT
classifier, not the original frozen-random-feature double-descent experiment.
The original two-example optimization witnesses become disjoint order/binding
tasks. L1–L6 retain the current model's RoPE. Architecture and supplied-feature
parameterization comparisons from the catalog are not extra current-GPT models.

Synthetic generative tasks use free generation for validation and test. Novel
validation inputs/prompts are selected when the source provides or permits them;
the original validation score is retained too. Recurrence splits keep whole
orbits disjoint and use the public codec, including MASH-half. Temporal models
retain decoded and native-token scores separately. Continuous series retain CE,
point errors and known-floor skill where available. Real forecasting compares
free-running generated values with persistence and seasonal naive on equal
eligible origins; generated rows feed the next rolling context.

```bash
.venv/bin/python -m research.current_grokking.run --write-plan --seeds 1
.venv/bin/python -m research.current_grokking.run --all --seeds 1
.venv/bin/python -m research.current_grokking.audit
.venv/bin/python -m research.current_grokking.report
```

The dispatcher skips completed cells and explicit `skipped.json` markers, and
resumes retained checkpoints. JSON/CSV
summaries include pending and execution-error cells. Original data dependencies
are local: `lean_projects/transformer`, the arithmetic task registry under
`rust_projects/grokking`, the repository's series/text datasets, and the archived
checksum-verified Fashion-MNIST IDX files. Those external repositories are read
only. `tests/diff_gpt/application/test_current_grokking*.py` check independent
integer oracles, codec behavior, numeric generation scoring, proper series scores,
sampler/checkpoint integrity and CUDA capture neutrality.

After the initial arithmetic and new-probe/image runs, dispatch prioritizes the
hard-recall anchor and one representative from each remaining family. This
changes execution order only; the frozen case settings and budgets are unchanged.

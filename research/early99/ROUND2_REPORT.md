# Round two: short selection and pending long diagnostic

The short-suite selection is frozen. **Scaled Peri-RMS + embedding RMS + SSMax,
without Q/K RMS, at LR ×2** reaches first-observed 99% exact held-out accuracy
in **20.9798 geometric-mean epochs**, with **75/75 passes**. This is **38.7% fewer
epochs than LR ×2** and **63.3% fewer than the original baseline**. The reused
30-cell long diagnostic is still running; this report does not claim that the
round is complete.

The shared-model research candidate **keeps Q/K RMS**, as requested. Its best
completed normalized arm is scaled Peri-RMS + embedding RMS + SSMax at
**24.3271 epochs**, **29.0% fewer than LR ×2**. The no-Q/K result is a benchmark
ablation, not a recommendation to change normalization by task size.

## Complete short comparison

Every full-suite row below passes all 75 cells. The epoch ratio uses the same
25 tasks and seeds 0/1/2 as LR ×2. Lower ratios are earlier; the worst ratio
retains slow individual cells even when the aggregate improves.

| Configuration | GM first-99 epochs | Ratio to LR ×2 | Worst paired ratio to LR ×2 | Trainer seconds |
|---|---:|---:|---:|---:|
| Original baseline | 57.1653 | — | — | 2066.37 |
| LR ×2 incumbent | 34.2499 | 1.0000 | 1.00 | 1297.23 |
| LR ×3 | 35.3367 | 1.0317 | 20.63 | 1470.06 |
| LR ×2.5 | 33.2585 | 0.9711 | 5.79 | 1181.87 |
| Scaled Peri-RMS + embedding RMS | 28.3606 | 0.8280 | 6.00 | 1371.78 |
| + SSMax, Q/K RMS retained | **24.3271** | **0.7103** | 4.00 | 1242.35 |
| + SSMax, Q/K RMS removed | **20.9798** | **0.6125** | 14.00 | 1279.04 |
| + LeRoPE, ordinary softmax | 28.3558 | 0.8279 | 5.50 | 1525.97 |
| + SSMax + LeRoPE, Q/K RMS retained | 25.9959 | 0.7590 | 6.67 | 1574.27 |
| + SSMax + Log-Scaled LeRoPE, Q/K RMS retained | 28.3632 | 0.8281 | 15.33 | 1687.52 |

All model rows use LR ×2 unless another multiplier is stated. They retain
centered XSA, SwiGLU and tied embedding/readout weights. The scaled variants
use parameter-free RMS, including the embedding, with residual coefficients
`1/sqrt(2L)` applied **after** branch RMS. SSMax uses the declared parameter-free
`s=1` rule and visible causal key count. Log-Scaled LeRoPE maps the zero-based
global token position to `p*log(p+1)` while retaining SSMax's score scaling.

Original parameter counts remain **100,736–1,183,488**, depending on the fixed
task. LeRoPE adds `head_dim/2` shared frequency scalars, giving
**100,744–1,183,504**; all original matrix entries and RNG draws are preserved.
Trainer seconds include setup, compilation and scoring. Runtime is recorded
but does not determine promotion or selection, as requested.

The selected ablation is earlier in 49 cells, tied in 9 and later in 17 versus
LR ×2. Its worst slowdown is **14×** on `basis-depth-hard`, seed 1
(0.4 to 5.6 epochs). The exact case identifier and every paired value are in
the [75-cell comparison](results/round2/peri-rms-scaled-ssmax-noqk-lr2-vs-lr2-short.json).
The complete [short summary](results/round2/final-short-summary.json) links all
controls, task medians/worst cases, runtimes and parameter counts.

The user's full Log-Scaled LeRoPE combination passes every short cell, but
takes **16.59% more aggregate epochs than SSMax/QK without learned rotations**.
SSMax + ordinary LeRoPE takes **6.86% more**. These measured regressions remain
part of the result; norm preservation does not imply faster learning.

## Rejected screens and failures

The screen requires all 18 cells to pass and at least three of six task median
paired ratios to improve over LR ×2. Every rejected arm remains recorded;
none receives a filtered full-suite aggregate.

| Screened arm | Passes / 18 | Improved task medians / 6 |
|---|---:|---:|
| LR ×2, beta2 0.95 | 18 | 2 |
| LR ×3 with cosine decay to ×1.5 over 200 epochs | 18 | 1 |
| Unscaled branch-only Peri-RMS | 12 | 0 |
| Unscaled Peri-RMS + embedding RMS | 18 | 2 |
| Scaled Peri-RMS + causal centering, LR ×2 | 18 | 1 |
| Scaled Peri-RMS + causal centering, LR ×1 | 15 | 1 |

Branch-only Peri-RMS exhausts the original budgets in all three key-value
binding seeds and all three square-sum seeds. The causal-centering LR ×1
screen also exhausts the square-sum budget in all three seeds, retained in its
[full screen record](results/round2/screen-peri-rms-scaled-csa-lr1.json).
The [reporting audit](../early99_reporting/README.md) reconstructs all 14 screen
decisions and all 26 full-short paired reports directly from raw histories,
checking the JSON and CSVs without rewriting the frozen results.

## Mathematical guarantees and practical limits

The [checked guarantee records](results/round2/guarantees.json) and
[attention contracts](results/round2/attention-extension-guarantees.json)
state their hypotheses and link the Lean fixtures and compiler output.
They use real arithmetic, positive width and positive finite epsilon; they
do not certify floating-point behavior or convergence of the complete GPT.

- Parameter-free finite-epsilon RMS has Euclidean norm at most `sqrt(d)`.
  Normalized embeddings and two normalized branches per layer, each scaled by
  `1/sqrt(2L)`, give residual-state RMS at most **`1+sqrt(2L)`**.
  The frozen record gives `2*alpha/sqrt(epsilon)` for one scaled RMS operation.
  A [supplemental Lean proof](../normalization_guarantees/README.md) improves this
  to `abs(alpha)/sqrt(epsilon)` for the unchanged real function; it does not
  bound the raw branch map's sensitivity or alter the frozen records.
- Q/K RMS bounds SSMax scores independently of projection-weight norms:
  `|score| <= log(N)*|g|*d` for `N>=1` visible keys and fixed score gain `g`.
  **The selected no-Q/K ablation loses this score bound**, while retaining the
  scaled residual-state bound.
- SSMax has a conditional probability lower bound when every competing score
  is separated by a specified positive margin. Q/K normalization does not
  guarantee that a trained model maintains that margin.
- Learned rotary transformations preserve norm. Ordinary LeRoPE has the
  fixed-frequency relative-position identity; Log-Scaled LeRoPE has the
  [warped-offset identity](results/round2/log-angle-guarantees.json) and loses
  ordinary translation invariance. The benchmark's learned frequency range
  remains unbounded.

These bounds do not control unconstrained readout weights, the full network's
Jacobian, optimization success or length extrapolation. The mechanism mapping
uses reviewed equations and numerical tests, not a formal translation of PyTorch.
The final scientific implementation passed the recorded **86 CPU/CUDA checks**,
including causal/cached behavior, independent equations, initialization pairing,
issued optimizer updates, and capture/RNG neutrality; see
[validation](results/round2/validation.json).

Separate [component stress repairs](../transformer_stress_fixes/README.md)
pass all 65 declared full-stack numerical cases. The
[frequency-gradient repair](../transformer_gradient_stress/README.md) additionally
preserves tiny derivatives that native tanh backward rounded to zero. These
use separate source snapshots and are not part of the benchmark configurations
above. Residual learning repairs reach 10/12 on one teacher but regress on a
mixed teacher; the [complete follow-up](../transformer_stress_parity_init/README.md)
retains the unresolved tradeoff. No universal optimum is established.

## Long diagnostic and reproducibility

The [selection](results/round2/selection.json) was frozen before the selected
arm's long run, with SHA-256
`ec56a5245f2c9626eab67e1d809bc8c6b7889010adf2cf3e774698cb5db24c81`.
It receives exactly one 30-cell diagnostic at the original budgets, from
source `7cff292`. The long suite was already observed in round one; it is
reused regression evidence, not a fresh holdout. Candidate choice and tuning
do not use its outcomes. Its complete outcome and checkpoint audit remain
pending and will be added without reselection.

The benchmark keeps all 35 fixed tasks, data seed 1, model/shuffle seeds 0/1/2,
original data and dimensions, native AdamW betas `(0.9,0.98)`, epsilon `1e-8`,
weight decay 0.1 and warmup 50, except for separately declared ablations.
Epochs count actual examples, including short final batches and the zero-rate
first update; capture warmup is excluded. Scoring uses complete held-out exact
sequences and required free generation. A first 99% crossing counts; historical
60/300-second thresholds define membership, not new timeouts.

Raw histories, checkpoints and source archives remain under `runs/early99`.
Use the [repository reporting audit](../early99_reporting/README.md) to reproduce
the comparisons and verify the frozen artifacts. Final long reports, final
round documentation and the user's requested subsequent research remain open.

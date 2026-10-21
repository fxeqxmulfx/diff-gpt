# Plan: baseline accuracy and incremental improvements

Started: 2026-09-05. Baseline: v0.31.1 (commit 1097617), plus benchmark
instrumentation (`M4_SEED`, `M4_SAMPLERS`, `M4_SAVE`, `M4_HEAD`, `M4_MIX_K`,
`M4_ATTNRES_EMBED_ONCE`), `MedianSampler`, and `benchmarks/m4_eval.py`.
Issue report (issues 9, 10 and C below): `encoder_decoder_issues.md`, deleted after
commit 1be1bff; read it with `git show 1be1bff:encoder_decoder_issues.md`.
The original plan header was updated at 2026-09-05 09:33; the run journal
continues through 13:52 that day. English edition: 2026-10-19.

## Evaluation protocol

**Primary: M4 Hourly, matching the README.** 414 series, horizon 48,
per-series z-normalization, one global model, k=1, V=256, n_embd=64,
2 layers, 4 heads, block_size=138, batch size 32, 20000 iterations,
evaluation every 500 iterations, best checkpoint selected by validation loss.
Hardware: RTX 3050 Laptop, 4 GB; about five minutes per seed initially.

```bash
# One seed; run_m4.sh runs a seed list and writes runs/<step>.pid.
M4_N_SERIES=414 M4_MAX_ITERS=20000 M4_LABEL_SMOOTHING_SIGMA=2.0 \
M4_SEED=0 M4_SAMPLERS=argmax,median M4_SAVE=runs/step0_seed0.pt \
uv run --no-sync python -m benchmarks.m4_bench
# Four seeds of one configuration; additional M4_* variables are forwarded.
runs/run_m4.sh <step> 0 1 2 3
# To stop a run: kill $(cat runs/<step>.pid)
```

Primary point metrics are sMAPE and MASE with median readout (step 1),
with argmax reported alongside. Baseline standard deviations over four seeds
are 0.24 sMAPE and 0.13 MASE. Changes requiring retraining therefore use
four seeds (0–3); inference-only changes use paired comparisons on the same
saved weights. Means over 414 series are dominated by tails: five series
account for 90% of the MASE shift. Accepted steps also report median
per-series sMAPE/MASE, trimmed means excluding the highest 2% by MASE, and
M4 OWA = ½(sMAPE/18.383 + MASE/2.395), using the Hourly Naive2 references
from Makridakis et al. (2020). OWA resolves disagreements between sMAPE and
MASE. Distribution evaluation is independent of point readout: CRPS, MSIS,
coverage@80, and true-token NLL (`M4_EVAL_READOUTS=1`).

**Secondary: ETTh1, iTransformer protocol** (`benchmarks/ett_bench.py`):
7 channels, sequence length 96, prediction length 96, MSE/MAE. This is used
for multichannel changes; M4 Hourly is univariate. With block size 1344 and
n_embd=128, GPU utilization reaches 99%, throughput is about 5 iterations/s,
and a seed with MC readout takes about 13 minutes. ETTh1 changes initially
use two screening seeds; seeds 2 and 3 are added when the result is within noise.

## Rules

1. One change per step: a flagged implementation, the four-seed protocol,
   and a row in the results table.
2. Accept improvements larger than noise and consistent across seeds.
   Use OWA and the median series when metrics disagree; the user makes the
   final decision. Correctness fixes are accepted if they do not regress.
3. Accepted changes accumulate. Each next step starts from the accepted
   configuration; `runs/run_m4.sh` supplies its default flags.
4. Save weights in `runs/` for inference-only comparisons without retraining.
   Checkpoints stay local and are excluded from Git.
5. Stop only the PID recorded in `runs/<step>.pid`. Do not use `pkill -f`
   or search for parent processes.

## Steps

| # | Change | Location | Retrain | Benchmark | Status |
|---|---|---|---|---|---|
| 0 | Baseline accuracy | Existing code | Yes | M4 | Done; README sMAPE 17.49 reproduced with seed 0 |
| 1 | `MedianSampler` instead of argmax for point forecasts | `sampler/median.py` | No | M4 | Accepted; −0.35 sMAPE |
| 2 | P≡0 tokenizer (LGT): qₜ = Δᵏ round(s·xₜ) + V/2, integer decoder | `encoder_decoder.py` | Yes | M4 | Accepted as a correctness fix; subsequently made the package default |
| 3 | Logistic-mixture head; token probability is cell mass, σ=0 | `model/gpt.py` | Yes | M4 | **Accepted**, user-confirmed 2026-09-05; Bayes readout improves all point metrics and proper scores |
| 3b | Step 3 with σ=2 | Benchmark flags | Yes | M4 | Rejected |
| 4 | `optimizer.eval()` at the end of `train()` | `train.py` | Yes | M4 | Accepted without a benchmark run; no effect in this protocol |
| 5 | Commit the embedding once in AttnRes | `model/gpt.py`, `block.py` | Yes | M4 | Rejected; worse on every seed, doubled embedding weight is a useful bias |
| 6 | ETTh1 baseline accuracy | Existing code | Yes | ETTh1 | Done; README 0.9787 / 0.6030 reproduced; LGT plus mixture is within noise on ETT |
| 7 | Window offsets divisible by F; fix `target_columns` masking | `data_loader.py` | Yes | ETTh1 | **Accepted**; mask bug fix and prerequisite for step 8 |
| 8 | Channel embedding indexed by position mod F | `model/gpt.py` | Yes | ETTh1 | **Accepted**; val loss −0.32 nats, median-window MSE −22%, CRPS −8%, both seeds |
| 9 | Quantile-based robust domain and a closed loop on integer levels | `encoder_decoder.py` | Yes | M4, ETTh1 | Quantile domain **implemented**; Daily runs remain pending at the end of the journal. Integer-level closed loop not implemented |
| 10 | Bayes readout from N trajectories (`predict_point`), distribution scoring | `metrics/point.py`, `diff_gpt.py`, `m4_eval.py` | No | M4 | Seeds 0/1 of step 3 give 16.99 / 1.305 / OWA 0.735; accepted as the main mixture-head readout |
| 11 | Exact token integrals and seasonal lags as numeric features, inspired by transformer-vm | `model/gpt.py`, `model/kv_cache.py` | Yes | M4 | **Accepted; largest gain in the session**: OWA 0.628 versus 0.726, sMAPE 14.40 versus 16.91, val loss −0.082 nats on all four seeds |
| 12 | Levels instead of differences: k=0 at V=256 and V=1024 | Benchmark flags | Yes | M4 | **Rejected**; initial status recorded OWA 1.08 and 1.42 versus 0.726; final values appear below. Also limited by trends; see issue 10 |
| 13 | Ordinal loss (ranked probability score) instead of σ for softmax | `model/gpt.py` | Yes | M4 | Proposed; compatible with existing checkpoints |
| 14 | Level normalization instead of z-score; investigate sMAPE/MASE disagreement | `benchmarks/m4_bench.py` | Yes | M4 | Proposed; changes the protocol and loses direct README comparability |
| 15 | Hybrid head: mixture plus free per-bin correction with weight decay | `model/gpt.py` | Yes | M4 | Proposed; low priority after step 3b |
| 16 | Forward sampling seeds through `predict_samples` and `predict_point` | `diff_gpt.py` | No | — | **Done**; all sampling entry points accept `seed`, with three tests. Calibration deliberately shares a seed for paired comparisons |
| 17 | Mid-rise quantizer to use token zero: D/(V−1) instead of D/(V−2) | `encoder_decoder.py` | Yes | M4 | Proposed; about 0.4% potential gain, low priority |
| 18 | Calibrate free-run temperature on validation windows | `diff_gpt.py` | No | M4, ETTh1 | **Accepted on ETT** using CRPS; no gain on M4, whose rollout is 48 rather than 672 tokens |

After step 5 and the step 10 evaluation, the planned order was 6 → 7 → 8
(ETTh1), then structural steps 11 and 12, followed by 9 and 13 as time permits.

## Results

| # | Change | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean sMAPE / MASE | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | Baseline, argmax | 17.49 / 1.682 | 17.91 / 1.958 | 17.71 / 1.752 | 18.04 / 1.689 | 17.79 / 1.770 | README 17.49 is seed 0; seed std 0.24 sMAPE, 0.13 MASE |
| 1 | Median readout, same weights | 16.88 / 1.671 | 18.01 / 1.901 | 17.13 / 1.737 | 17.74 / 1.721 | 17.44 / 1.758 | Paired sMAPE changes −0.61, +0.10, −0.58, −0.30; mean −0.35 ± 0.17; accepted |
| 2 | LGT (P≡0), argmax | 17.63 / 1.773 | 17.33 / 1.820 | 17.52 / 1.878 | 17.30 / 1.878 | 17.45 / 1.837 | sMAPE −0.35, MASE +0.07 against std 0.13; accepted as a correctness fix |
| 2 | LGT (P≡0), median | 16.70 / 1.747 | 17.49 / 1.882 | 17.43 / 1.838 | 17.43 / 1.818 | 17.26 / 1.821 | sMAPE −0.18 versus step 1; seed std 0.37 versus 0.51 |
| 3 | Logistic mixture, σ=0, LGT, argmax | 19.38 / 1.673 | 19.61 / 1.570 | 18.79 / 1.603 | 19.14 / 1.581 | 19.23 / 1.607 | |
| 3 | Logistic mixture, σ=0, LGT, median | 18.03 / 1.635 | 17.93 / 1.537 | 17.70 / 1.540 | 18.42 / 1.549 | 18.02 / 1.565 | MASE −0.26 (−14%) on all seeds; sMAPE +0.76 on all seeds; OWA improves |
| 3b | Mixture with σ=2, LGT, median | 19.15 / 1.895 | 17.97 / 1.927 | 18.32 / 2.103 | 18.48 / 1.889 | 18.48 / 1.954 | Worse than steps 2/3 on both metrics on all seeds; rejected. Argmax 18.87 / 2.026 |
| 4 | `optimizer.eval()` after training | Not run | | | | | No effect in this protocol; after 2000 iterations x loss 0.118 versus y loss 0.128; accepted |
| 5 | Single embedding commit on step 3, argmax | 18.99 / 1.595 | 19.38 / 1.666 | 19.99 / 1.682 | 20.23 / 1.791 | 19.65 / 1.684 | |
| 5 | Single embedding commit on step 3, median | 18.05 / 1.532 | 18.10 / 1.617 | 18.51 / 1.643 | 18.79 / 1.735 | 18.36 / 1.632 | sMAPE +0.34 on all seeds; MASE +0.07 on three of four; rejected, flag stays off |
| 12 | k=0 levels, V=256, median | 15.51 / 2.976 | 16.13 / 3.278 | — | — | 15.82 / 3.127 | Best sMAPE at that point, worst MASE; OWA 1.083 versus 0.726 |
| 12 | k=0 levels, V=1024, median | 16.26 / 4.985 | 17.48 / 6.307 | — | — | 16.87 / 5.646 | Finer grid does not help; MASE nearly doubles; rejected |
| 11 | Exact features, lag 24, argmax | 15.56 / 1.192 | 15.54 / 1.166 | 14.90 / 1.122 | 15.75 / 1.189 | 15.44 / 1.167 | OWA 0.664 |
| 11 | Exact features, lag 24, median | 14.04 / 1.144 | 14.95 / 1.136 | 14.05 / 1.096 | 14.58 / 1.145 | **14.40 / 1.130** | **OWA 0.628**; val loss −0.082 nats on all four seeds; accepted |

Each result cell contains sMAPE / MASE.

### Robust metrics: median readout, four-seed means (`benchmarks/m4_eval.py`)

| # | Variant | sMAPE | MASE | Median sMAPE | Median MASE | Trimmed sMAPE | Trimmed MASE | OWA |
|---|---|---|---|---|---|---|---|---|
| 1 | mash, softmax σ=2 | 17.44 | 1.758 | 4.94 | 1.296 | 17.76 | 1.565 | 0.841 |
| 2 | lgt, softmax σ=2 | 17.26 | 1.821 | 5.05 | 1.304 | 17.58 | 1.629 | 0.850 |
| 3 | lgt, mixture σ=0 | 18.02 | 1.565 | 4.58 | 1.203 | 18.27 | 1.400 | 0.817 |

Step 3 improves on step 2 in OWA (−0.033), the median series (sMAPE −9%,
MASE −8%), and trimmed MASE (−14%). Mean sMAPE worsens on volatile series
with small absolute mean relative to standard deviation: +2.91 there,
versus −0.24 elsewhere. The step 10 hypothesis was that an sMAPE-optimal
point differs from the median on these distributions, and that a
token-median rollout is not the marginal optimum for the final metric.

### Bayes readout and distribution scores: step 10, 100 trajectories

`M4_EVAL_READOUTS=1`. Paired seed 0 / seed 1 results on the same saved
weights; OWA = ½(sMAPE/18.383 + MASE/2.395).

| Weights | Readout | sMAPE | MASE | OWA |
|---|---|---|---|---|
| Step 2, softmax σ=2 | Median-token path | 16.70 / 17.49 | 1.747 / 1.882 | 0.819 / 0.869 |
| Step 2, softmax σ=2 | MC median | 25.61 / 24.00 | 3.797 / 3.587 | 1.489 / 1.402 |
| Step 2, softmax σ=2 | MC OWA-Bayes | 22.95 / 21.85 | 3.780 / 3.584 | 1.413 / 1.343 |
| Step 3, mixture σ=0 | Median-token path | 18.03 / 17.93 | 1.635 / 1.537 | 0.832 / 0.809 |
| Step 3, mixture σ=0 | MC median | 17.27 / 17.18 | 1.324 / 1.259 | 0.746 / 0.730 |
| Step 3, mixture σ=0 | MC sMAPE-Bayes | 17.17 / 16.93 | 1.343 / 1.273 | 0.747 / 0.726 |
| Step 3, mixture σ=0 | **MC OWA-Bayes** | **17.11 / 16.87** | **1.340 / 1.270** | **0.745 / 0.724** |
| Step 3, seeds 2 / 3 | MC OWA-Bayes | 16.97 / 16.67 | 1.273 / 1.220 | 0.727 / 0.708 |
| Step 3, four-seed mean | MC OWA-Bayes | **16.91** | **1.276** | **0.726** |
| Step 5 on step 3 | MC OWA-Bayes | 18.43 | 1.835 | 0.884 |

| Weights | CRPS | MSIS α=0.05 | Coverage@80 | True-token NLL |
|---|---|---|---|---|
| Step 2, softmax σ=2 | 343.1 / 342.8 | 47.8 / 44.5 | 0.947 / 0.953 | 1.997 / 2.007 |
| Step 3, mixture σ=0 | **186.0 / 185.9** | **9.9 / 10.0** | **0.872 / 0.870** | **1.431 / 1.423** |
| Step 3, seeds 2 / 3 | 185.0 / 175.7 | 9.6 / 9.7 | 0.861 / 0.866 | 1.427 / 1.433 |
| Step 5, seed 0 | 213.5 | 12.5 | 0.807 | 1.527 |

The mixture head improves CRPS by 46%, MSIS by 78%, and NLL by 0.57 nats;
coverage moves closer to nominal. The README estimated M4 winner MSIS at
13–15, compared with about 10 here. With Bayes readout, step 3 also improves
point metrics over step 2's median-token path: two-seed sMAPE 16.99 versus
17.10, MASE 1.305 versus 1.815 (−28%), OWA 0.735 versus 0.844 (−13%).
The token-median path obscured distribution quality. MC readout on the
σ=2 softmax head is poorly calibrated, explaining the README's 24.18 sMAPE.
Step 5 is also worse under this evaluation.

Protocol decision: the mixture head uses MC OWA-Bayes (`predict_point`,
`metric="owa"`) as the main readout, with sMAPE-Bayes and median alongside.
Steps 3 and 10 form the accepted configuration at this stage. Four-seed
results: sMAPE 16.91, MASE 1.276, OWA 0.726, CRPS 183, MSIS 9.8. Against
step 2's median-token path (17.26 / 1.821 / 0.850), this is −0.35 sMAPE,
−30% MASE, and −15% OWA.

## Training speed: 2026-09-05, RTX 3050 Laptop, batch 32, T=138

GPU utilization of 20–22% alongside one fully used Python core initially
looked CPU-bound. Increasing batch size from 32 to 128 costs 3.5 times
more, indicating substantial GPU work with small, memory-bound kernels.
Steady-state mixture-head microbenchmark after warmup:

| Configuration | ms/iteration | Iterations/s |
|---|---|---|
| Eager, before head optimization | 27.9 | 36 |
| Compile, max-autotune-no-cudagraphs, before optimization | 8.2 | 123 |
| CUDA graphs, reduce-overhead | 8.1 | 123 |
| Compile, target-cell-only mixture loss | **6.7** | **149** |
| Eager, target-cell-only mixture loss | 12.8 | 78 |
| Compile, batch 128 / 256 | 22.9 / 43.0 | |

Training now computes only the target cell's mass: K sigmoid evaluations
per position instead of V·K (`_mixture_target_nll`), with equivalence to
full cross-entropy checked by a test. Head cost becomes independent of V,
making larger vocabularies in step 12 inexpensive at the head level.
`_estimate_loss` uses `no_grad` instead of `inference_mode` to allow
CUDA-graph recording. `M4_COMPILE_MODE` selects compilation; modes with
CUDA graphs compile a separate training-only wrapper.

CUDA graphs gave no additional speedup. Four parallel seeds shared the
GPU: each took 59 s versus 16 s alone; only data loading and compilation
overlap helped. The loader costs 1.3 ms of the 6.7 ms step. Larger batches
with fewer iterations would change the optimization protocol and need
a new baseline. A full seed now takes about 3.5 minutes instead of five.

### ETTh1, prediction length 96, MSE / MAE: steps 6–8

| # | Configuration | Readout | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---|---|---|---|---|---|---|
| 6a | mash, softmax σ=0, V=64 (README) | argmax | 0.9787 / 0.6030 | 0.9535 / 0.6013 | — | — | 0.966 / 0.602 |
| 6a | | median | 0.9894 / 0.6119 | 0.9694 / 0.6080 | — | — | 0.979 / 0.610 |
| 6b | lgt, mixture σ=0, V=64 | argmax | 0.9803 / 0.6063 | 1.7209 / 0.7646 | 0.9734 / 0.6049 | 0.9490 / 0.6036 | 1.156 / 0.645 |
| 6b | | median | 0.9829 / 0.6086 | 1.8243 / 0.8033 | 0.9819 / 0.6113 | 0.9647 / 0.6252 | 1.188 / 0.662 |
| 6b | | mc-median | 1.574 / 0.847 | 1.709 / 0.883 | 2.180 / 0.966 | 1.201 / 0.720 | 1.666 / 0.854 |
| 7 | Aligned windows, unaligned split | argmax | 0.9596 / 0.6014 | 0.9686 / 0.6073 | — | — | 0.964 / 0.604 |
| 8 | Channel embedding, unaligned split | argmax | 7.269 / 1.297 | Stopped | — | — | Invalid |
| 7b | Aligned windows and split | argmax | 1.1578 / 0.6752 | 1.0041 / 0.6199 | — | — | 1.081 / 0.648 |
| 7b | | median | 1.0352 / 0.6512 | 1.0294 / 0.6280 | — | — | 1.032 / 0.640 |
| 8b | Channel embedding, aligned split | argmax | 1.0954 / 0.6924 | **0.9268 / 0.5926** | — | — | 1.011 / 0.643 |
| 8b | | median | 1.3414 / 0.7234 | 0.9636 / 0.6054 | — | — | 1.152 / 0.664 |

**Validation loss, the one-step CE actually optimized:** step 7b gives
2.510 / 2.514; step 8b gives **2.183 / 2.196**. Channel embeddings improve
the model by about 0.32 nats on both seeds. Mean test MSE can obscure this
because a few divergent windows dominate. Step 8b seed 1 reaches
0.9268 / 0.5926, better than the README's 0.9787 / 0.6030.
Robust, temperature-calibrated comparisons are in `runs/ett_compare.log`.

The README ETTh1 result (V=64, plain CE), MSE 0.9787 / MAE 0.6030,
was reproduced exactly by step 6a, seed 0.

ETT is much noisier than M4: step 6b seed 1 gives MSE 1.72, while the
other three give 0.95–0.98. Four-seed means can therefore be misleading,
and two seeds alone are insufficient for a verdict based on mean MSE.
The median over seeds is 0.977 for LGT plus mixture versus 0.966 for
softmax plus MASH, within noise. The working interpretation is that ETT
is limited by data (about two tokens per parameter), while MC readout
suffers from overdispersed free-running trajectories; see step 18.

**A few windows dominate ETT test MSE.** Per-window MSE on 29 windows,
using argmax (`scratchpad/ett_windows.py`):

| Checkpoint | Mean | Median | p90 | Maximum | Contribution of worst three windows |
|---|---|---|---|---|---|
| ett7 seed 0 | 0.960 | 0.946 | 1.69 | 1.93 | 20% |
| ett7b seed 0, corrected split | 1.158 | 1.046 | 1.97 | 3.87 | 28% |
| ett6b seed 1, outlier | 1.717 | 1.089 | 2.18 | **17.8** | 50% |
| ett6b seed 3 | 0.949 | 0.897 | 1.44 | 1.87 | 19% |

Median-window MSE stays within 0.90–1.09 while mean MSE ranges from 0.95
to 1.72. A 672-token autoregressive rollout occasionally diverges, and one
of 29 windows can supply half the total error. Validation loss remains
2.48–2.54, implicating free-running inference rather than training.
ETT decisions therefore use median-window MSE and p90, with the mean
reported alongside. This is another instance of tail-dominated averages
and the free-run overdispersion addressed by step 18.

### Robust comparison of steps 7b and 8b with calibrated temperature

`runs/ett_compare.log`: temperature selected by CRPS on six validation
windows, evaluated on 29 test windows, N=16.

| Checkpoint | T* | Argmax MSE | MC MSE | **Median-window MSE** | MC MAE | **CRPS** | Coverage@80 |
|---|---|---|---|---|---|---|---|
| 7b seed 0 | 0.50 | 1.158 | 1.013 | 0.949 | 0.636 | 0.517 | 0.569 |
| 7b seed 1 | 0.30 | 1.004 | 1.011 | 1.002 | 0.617 | 0.530 | 0.336 |
| 8b seed 0 | 0.30 | 1.095 | 0.955 | 0.785 | 0.639 | 0.502 | 0.519 |
| 8b seed 1 | 0.50 | 0.927 | 0.932 | **0.742** | 0.613 | **0.461** | 0.592 |
| 7b mean | | 1.081 | 1.012 | 0.976 | 0.627 | 0.524 | |
| 8b mean | | 1.011 | **0.944** | **0.764** | 0.626 | **0.482** | |

Step 8 was accepted: median-window MSE −22%, CRPS −8%, MC MSE −7%,
validation loss −0.32 nats. Each 8b seed beats each 7b seed on median-window
MSE and CRPS. Calibrated MC readout also beats argmax (0.944 versus 1.011),
supporting step 18; at T=1 it had been about twice as bad. Step 7 was
accepted as both a mask correctness fix and a prerequisite for step 8.

**Split-alignment bug found and fixed in step 7.** `align_windows` initially
aligned window starts but left the train/validation split at an arbitrary
token. For ETTh1, token 60474 belongs to channel 1, so validation windows
started at phase 1 while training windows started at phase 0. Without
channel embeddings, validation loss was still 2.487–2.513. With channel
embeddings, it rose to 2.798, and `save_best` used the corrupted metric,
leading to step 8's MSE 7.27. The split is now rounded down to a multiple
of F when alignment is enabled, checked by
`test_aligned_split_point_keeps_val_windows_on_column_zero`. Steps 7 and 8
were rerun as 7b and 8b.

### Step 18: free-run temperature calibration on ETTh1

Step 6b seed 0 weights; four validation windows for calibration,
29 test windows for evaluation, N=12.

| Validation criterion | Selected T | Test CRPS | Test MSE, Bayes MSE readout | Median-window MSE | Coverage@80 |
|---|---|---|---|---|---|
| CRPS, proper score | **0.5** | **0.501** | **0.970** | 0.964 | 0.537 |
| Coverage closest to 0.8 | 1.0 | 0.874 | 2.305 | 2.008 | 0.879 |

Full test grid: T=0.3 gives CRPS 0.542 / MSE 0.959; T=0.5 gives
0.501 / 0.970; T=0.7 gives 0.579 / 1.220; T=1.0 gives 0.874 / 2.305.
Argmax MSE is 0.980.

Calibration should use a proper scoring rule. In this grid, coverage
increases with temperature and reaches the nominal level only near T=1,
where the forecast has already spread too far: MSE is 2.4 times worse
and CRPS nearly doubles. Coverage alone can be satisfied by a sufficiently
wide interval. CRPS selects T=0.5, where MC MSE matches argmax
(0.97 versus 0.98) and CRPS improves by 43%.

Implementation: `DiffGPT.calibrate_temperature(windows, H, criterion=...)`,
supporting `"crps"` and `"coverage"`, with four tests. The same calibration
was queued on M4, whose nominal 80% intervals had coverage around 0.87.

### Step 18 on M4 Hourly: no gain

Step 3 seed 0 weights, 414 series, N=100. Calibration uses 30 windows
from the end of the training portion; the test horizon is untouched.

| T | sMAPE | MASE | OWA | CRPS | MSIS | Coverage@80 |
|---|---|---|---|---|---|---|
| 0.50 | 17.21 | 1.427 | 0.766 | 207.6 | 10.58 | 0.669 |
| 0.70, selected by validation CRPS | 16.98 | 1.372 | 0.748 | 198.0 | 9.40 | 0.773 |
| 0.85 | 16.99 | 1.357 | 0.745 | 189.1 | 9.36 | 0.826 |
| 1.00, current setting | 17.11 | 1.340 | 0.745 | **186.0** | 9.94 | 0.872 |

The OWA change, 0.748 versus 0.745, is within noise, and T=1 has better
test CRPS (186 versus 198). The validation choice does not transfer here.
The working explanation is rollout length: M4 Hourly uses 48 tokens,
whereas ETT uses 672, allowing much more error accumulation. Coverage
at T=1 is only 0.87 on M4 versus about 0.95 on ETT.

Step 18 is accepted for the long multivariate ETT rollouts; M4 keeps T=1.
As a consistency check, the T=1 row reproduces the earlier step 3 result
(17.11 / 1.340 / OWA 0.745) to three decimal places.

### Probe motivating step 11: what does the model retain exactly?

Linear probe on the last hidden state, step 3 seed 0 weights,
13440 positions, 70/30 split. The mixture head is linear in that state,
so linear recoverability measures information directly available to it.

| Quantity, a deterministic function of token history | Held-out R² |
|---|---|
| Current level, the running token sum computed by decoder cumsum | 0.690 |
| Level change over one day, lag 24 | 0.218 |
| Token exactly one day earlier, a LookUp operation | 0.400 |
| Next token, the model's own prediction target | 0.646 |

The model spends capacity approximating quantities that can be computed
exactly. A running sum and a seasonal lookup are deterministic functions
of the available prefix. This motivates step 11: supply exact CumSum and
LookUp primitives as numeric features, inspired by transformer-vm.
Probe script: `scratchpad/probe_level.py`.

### M4 Hourly summary

414 series, four training seeds, two seeds for the final readout evaluation.

| Configuration | sMAPE | MASE | OWA | CRPS | MSIS | Coverage@80 |
|---|---|---|---|---|---|---|
| README: MASH, softmax σ=2, argmax | 17.49 | 1.659 | 0.822 | — | — | — |
| Median readout and LGT | 17.26 | 1.821 | 0.850 | 343 | 45.7 | 0.95 |
| Mixture head and OWA-Bayes readout | 16.91 | 1.276 | 0.726 | 186 | 9.9 | 0.87 |
| **Exact features, step 11, and Bayes readout** | **13.11** | **1.018** | **0.569** | 190 | **7.2** | **0.77** |

OWA improves by 31% and sMAPE by 25% against the original configuration.
Coverage moves from 0.87 to 0.77 against nominal 0.80; MSIS improves from
9.9 to 7.2. The README estimated probabilistic M4 winners at MSIS 13–15.
Bayes readout on the feature-enhanced model gives about another 0.05 OWA
improvement over the median-token path (0.569 versus 0.633).

### M4 Daily: approximately the naive forecast

3040 series of length at least 400, context 78, horizon 14, feature lag 7,
using the accepted configuration. The recorded Daily Naive2 reference
constants are sMAPE 3.045 and MASE 3.278; the original journal notes that
these were checked from memory rather than independently sourced.

| Forecast | sMAPE | MASE | OWA |
|---|---|---|---|
| Our Daily forecast | 2.99 | 3.35 | **1.00** |
| Hourly, for comparison | 13.11 | 1.018 | **0.569** |

The Daily result is close to Naive2. The working interpretation is that
these series are closer to weakly seasonal random walks. For a random walk,
the last observed value is optimal for the relevant point losses under the
usual symmetric-increment assumptions, and an increment predictor should
return zero. Additional prediction noise can hurt. Strong seasonality
appears to be a favorable setting for the method; this limitation should
be discussed in the paper.

`runs/daily_seed0.log` predates the seasonality correction and reports
MASE 1.201 instead of about 3.35.

### Why Daily trails the naive forecast: the grid is too coarse

The Daily model's forecast is flat: its range over 14 steps is exactly
0.0000. The remaining roughly 2% gap is associated with tokenization:

| M4 Daily domain | D | Δ/2 in typical-step units | Out of domain |
|---|---|---|---|
| Maximum, current setting | 19.513 | **3.80** | 0% |
| q99.99 | 1.217 | 0.24 | 0.01% |
| q99.9 | 0.413 | 0.08 | 0.1% |

The median increment is 0.0202, while the collection-wide maximum is 19.5.
One pathological series sets a coarse grid for the remaining 3039.
Quantization error alone can be 3.8 typical steps, making ordinary moves
unrepresentable. The same issue was much less visible on Hourly; the
maximum/q99.9 ratio is 1.7 there versus 47 on Daily.

Implemented: `get_domain_of_definition(..., quantile=0.9999)`,
`M4_DOMAIN_QUANTILE`, and two tests. The journal first described q99.99
and q99.9 runs as launched; its final 13:52 entry clarifies that they
were not run and remain pending.

### M4 Daily scaling curve: flat

Train on subsets, always evaluate on the same 3040 series;
median readout, one seed.

| Training series | sMAPE | MASE |
|---|---|---|
| 100 | 2.99 | 3.351 |
| 400 | 2.99 | 3.349 |
| 1500 | 2.99 | 3.349 |
| 3040 | 2.99 | 3.349 |

A roughly thirtyfold increase in training data changes neither metric
materially. The original interpretation was that this setup had reached
its effective error floor, written L = L∞ + A·N^−α + B·D^−β, rather than
being limited by training data. The flat forecast and coarse grid are
consistent with that interpretation, though this single-seed experiment
alone does not establish a general scaling law.

The proposed next scaling dataset is M4 Monthly: the plan records 48000
series, 10.4M points, and strong annual seasonality, offering both more
data and predictable structure.

## Rejected ideas

| Idea | Reason |
|---|---|
| Better linear predictor in place of Δᵏ, such as AR or seasonal prediction | Maximum residual on ETTh1/M4 improves only 1.04–1.15 times; outliers determine the maximum |
| Nonuniform, companding quantizer | Can improve MSE but not the sup-norm bound |
| Dithering | Makes noise signal-independent at a recorded cost of +0.72 bits/token |
| Smoothing σ on top of the mixture head, step 3b | Worse on both metrics on every seed |
| More tokens per sample | Doubles sequence length when context is already a bottleneck |
| Direct EFM instead of MASH cascade | Same headroom, bound, and even tokens in the comparison; no measured gain |
| Half-step anchor c(B)A = 1/(2s), as at the order 0, as the default | One more level, D/(V − ‖c‖₁ + 1): only 0.4% at V = 256. Flat stretches encode as periodic patterns of tokens (32, 31, 32, 31 at Δ with V = 64), and constants and integer grids no longer decode exactly. Opt-in as `tokenizer="mash-half"` since 2026-10-21; not run on a benchmark: likely worse |

## Accepted decisions

1. Step 3 accepted on 2026-09-05. LGT, mixture head with σ=0, and Bayes
   readout for the target metric form the accepted configuration;
   `run_m4.sh` supplies these defaults.
2. LGT became the package default (`DEFAULT_TOKENIZER="lgt"`). MASH is
   available via `tokenizer="mash"` or `DIFF_GPT_TOKENIZER=mash`.
   The README was updated. The main paper under `papers/` still needs
   its Theorem 1 reformulated to cover the updated tokenizer.

## Run journal

- 2026-09-05 07:48: instrumented `m4_bench.py`, added `MedianSampler`
  with four tests, and started steps 0/1 together for seeds 0/1.
  Logs: `runs/step0_seed{0,1}.log`; weights: `runs/step0_seed{0,1}.pt`.
- 2026-09-05 07:50: the first run was lost after an incorrect process-stop
  action rebooted the machine. Restarted through `runs/run_step0.sh`;
  PID in `runs/step0.pid`. Stop only with `kill $(cat runs/step0.pid)`.
- 2026-09-05 07:56: seed 0 finished: training 197 s, about 45 iterations/s,
  val_loss 2.3502; argmax 17.49 / 1.682, median 16.88 / 1.671.
- 2026-09-05 08:00: seed 1 finished: training 162 s, val_loss 2.3580;
  argmax 17.91 / 1.958, median 18.01 / 1.901. Seed variation was larger
  than expected, especially in MASE because of small naive-error denominators.
  Started seeds 2/3 with `runs/run_m4.sh step0 2 3`. Clarified the rule:
  paired saved-weight comparisons for inference-only changes, at least
  four seeds for retraining changes.
- 2026-09-05 08:09: seeds 2/3 finished. Step 1 accepted: median wins on
  three of four seeds, mean sMAPE −0.35 with paired error 0.17; mean MASE
  change −0.012. Later steps report both readouts and use median for decisions.
- 2026-09-05 08:14: implemented step 2, `encode_lgt`/`decode_lgt`, selected
  through `tokenizer=` or `DIFF_GPT_TOKENIZER`, then still defaulting to
  `mash`. Added 14 tests in `test_lgt.py`; integration tests passed in
  both modes. Started `DIFF_GPT_TOKENIZER=lgt runs/run_m4.sh step2 0 1 2 3`.
- 2026-09-05 08:29: step 2 completed and accepted. All 40 encoder tests
  passed. `runs/run_m4.sh` now defaults to LGT for subsequent steps.
  Making LGT the package default and updating the README remained a user decision.
- 2026-09-05 08:35: implemented step 3 with
  `GPT(head="mixture", mixture_components=K)`. `_mixture_log_probs`
  returns (B,T,V) log-probabilities, preserving the loss, Engine, and
  sampler interfaces. The mixture branch has no tied `lm_head`.
  Five new tests, 56 total passing. Started
  `M4_HEAD=mixture M4_LABEL_SMOOTHING_SIGMA=0 runs/run_m4.sh step3 0 1 2 3`
  with LGT.
- 2026-09-05 08:50: step 3 completed with conflicting point metrics:
  MASE improved by 14%, sMAPE worsened by 4%. Started step 3b with the
  same head and σ=2 (`M4_HEAD=mixture M4_LABEL_SMOOTHING_SIGMA=2.0`).
  Started per-series step 2/3 diagnostics on seed 0 weights.
- 2026-09-05 08:58: `benchmarks/m4_eval.py` found median per-series
  ΔsMAPE 0.00; half the series improved. Mean sMAPE deterioration was
  concentrated in series with small absolute mean/std: +2.91 versus
  −0.24 elsewhere. Five series supplied 90% of the MASE gain
  (−0.101 of −0.112). Median per-series sMAPE improved 5.12 → 4.53.
  Added per-series medians and means trimmed by the worst 2% of MASE,
  evaluated on saved weights for all steps (`runs/robust_*.log`).
- 2026-09-05 09:12: implemented step 4 in `train.py`, also replacing
  `.seconds` with `total_seconds()`, with `test_train_optimizer_eval.py`.
  The probabilistic median-band test needed more training because the
  no-validation path now returns x weights: increased 500 to 2000 iterations.
  Implemented step 5 behind `GPT(attn_res_commit_embed_once=True)` and
  `M4_ATTNRES_EMBED_ONCE=1`, with `test_attn_res_embed_once.py`.
  Its run would follow step 3b, on top of step 3.
- 2026-09-05 09:13: step 3b completed and was rejected; smoothing the
  mixture head worsened both metrics. Started step 5 on mixture, σ=0,
  LGT. The fallback plan was to rerun it on softmax σ=2 if the user
  declined step 3.
- 2026-09-05 09:25: implemented `metrics/point.py::bayes_point` and
  `DiffGPT.predict_point`, with six tests: median for MAE/MASE, mean for
  MSE, expected-sMAPE minimization, and OWA minimization. Numeric probes
  found the sMAPE optimum at or slightly above the median for the tested
  unimodal cases (+0.8% for normal(10,3), +3.9% for normal(2,1)), and near
  a mode for tested bimodal cases. Added `M4_EVAL_READOUTS=1` to compare
  median-token paths, MC median, MC sMAPE-Bayes, MC OWA-Bayes, CRPS, MSIS,
  coverage@80, and true-token NLL. Queued saved-weight evaluation of steps
  2/3/5 after step 5 (`runs/readouts.log`). The Percepta-Core transformer-vm
  idea suggested exact CumSum/LookUp features and fixed lag heads as a
  separate retraining step.
- 2026-09-05 09:30: updated step statuses, added proposals 11–17,
  recorded rejected ideas and open decisions, and moved step 3b into results.
- 2026-09-05 09:33: step 5 completed and was rejected: median
  18.36 / 1.632 versus step 3's 18.02 / 1.565. Bayes-readout evaluation
  started automatically on the GPU.
- 2026-09-05 09:46: readout evaluation finished (`runs/readouts.log`).
  Mixture plus MC OWA-Bayes beat softmax σ=2 on sMAPE, MASE, OWA, CRPS,
  MSIS, coverage, and NLL. GPU utilization was 20–22%, one Python core
  fully busy, loader 1.3 ms of a 22 ms iteration. Started
  `runs/probe_speed.sh` with four parallel seeds and CUDA-graph compilation
  modes (`M4_COMPILE_MODE`), followed by step 3 seeds 2/3 readout evaluation.
- 2026-09-05 09:58: step 3 seeds 2/3 readouts finished; recorded the
  four-seed result. The 1500-iteration speed probe (`runs/probe_speed.out`)
  took 16 s for one training run and 59 s per seed for four parallel runs.
  GPU time was shared rather than training being accelerated; overlap in
  loading/compilation reduced total wall time to 125 s versus 256 s.
  CUDA-graph modes `reduce-overhead` and `max-autotune` failed initially.
- 2026-09-05 10:10: speed measurements completed; see the training-speed
  section. Added target-cell-only mixture NLL with a full-CE equivalence
  test, `no_grad` in `_estimate_loss`, and `M4_COMPILE_MODE`.
- 2026-09-05 10:21: user confirmed step 3 and LGT as the default.
  Set `DEFAULT_TOKENIZER="lgt"`; updated the README for LGT, mixture,
  `predict_point`, and M4 results; `run_m4.sh` retains the accepted flags.
  Instrumented `ett_bench.py` with `ETT_SEED`, `ETT_HEAD`, `ETT_MIX_K`,
  `ETT_VOCAB_SIZE`, `ETT_READOUTS=argmax,median,mc`, `ETT_NUM_SAMPLES`,
  `ETT_SAVE`, and compilation. Started step 6 at prediction length 96:
  6a, README settings (mash, softmax, V=64, argmax), seeds 0/1;
  6b, accepted settings (lgt, mixture σ=0, V=64, MC N=50), seeds 0–3.
  Started the full test suite with the new default in the background.
- 2026-09-05 10:24: implemented step 7 with per-row masking phases,
  fixing issue C, `DiffDataFrameDataLoader(align_windows=True)`, and
  `ETT_ALIGN_WINDOWS=1`. Rewrote the old mask test, which encoded the bug.
  Added step 8 via `GPT(n_channels=F)`, channel embeddings indexed by
  position mod F, aligned with cache and generation positions, exposed as
  `ETT_N_CHANNELS=7`, with `test_channel_embedding.py`. Both awaited 6b.
- 2026-09-05 10:27: ETT measured 5 iterations/s at 99% GPU utilization,
  about 13 minutes per seed. Reduced steps 7/8 to two screening seeds;
  chain PID in `runs/ett78_chain.pid`.
- 2026-09-05 10:31: the default change exposed two LGT defects, recorded
  in issue 9: mandatory 2ᵏ headroom at k≥2 (V=4, k=2 returned x/2 after
  clipping), and loss of constant-column levels. Fixed both and pinned
  Sigma-Delta tests to `tokenizer="mash"`. M4/ETT settings, V=256/64,
  k=1, D>0, are unaffected. ETT 6a seed 0 reproduced the README exactly:
  argmax 0.9787 / 0.6030, median 0.9894 / 0.6119.
- 2026-09-05 10:45: ETT 6a seed 1: argmax 0.9535 / 0.6013, median
  0.9694 / 0.6080. ETT 6b seed 0: argmax 0.9803 / 0.6063, median
  0.9829 / 0.6086; MC mean 1.497 / 0.819 and MC median 1.574 / 0.847.
  Hypothesis: V=64 open edge bins produce ±31-bin increments that
  destabilize 672-token trajectories. Proposed checks were edge-token
  frequency, CRPS/coverage, and a V=256 run, since mixture training-head
  cost no longer scales with vocabulary size.
- 2026-09-05 10:47: MC diagnostic on 6b seed 0, five windows, N=20:
  edge tokens were only 0.07%, rejecting that hypothesis. Trajectory
  spread was 2.70 z, coverage@80 0.951, CRPS 0.95 versus argmax RMSE 0.88.
  The T=1 free run was overdispersed across 672 steps and seven coupled
  channels. Validation CE was about 2.5 nats at V=64; sampled noise in
  one channel changes the context of the others. Initial ETT decisions
  therefore used argmax/median paths.
- 2026-09-05 10:48: temperature probe on the same five windows, N=20,
  reporting MC-median MSE / coverage@80 / CRPS:
  T=1.0 → 2.95 / 0.95 / 0.95;
  T=0.7 → 1.16 / 0.87 / 0.60;
  T=0.5 → 0.77 / 0.70 / 0.46, against argmax 0.78;
  top-p 0.7 → 0.88 / 0.67 / 0.50. Added inference-only step 18,
  free-run temperature calibration on validation data for both tasks.
- 2026-09-05 11:43: step 6 completed. Steps 7/8 were restarted as 7b/8b
  after fixing train/validation split alignment, diagnosed from step 8's
  abnormal validation loss 2.798. Stopped the invalid run using its PID file.
- 2026-09-05 11:54: full suite passed, 231 tests. Per-window ETT analysis
  showed stable medians and means dominated by one to three divergent
  windows out of 29. Changed ETT decisions to median-window MSE and p90.
- 2026-09-05 12:04: step 7b completed: argmax/median MSE 1.158 / 1.035
  for seed 0 and 1.004 / 1.029 for seed 1. Implemented and checked step 18
  on ETT, comparing CRPS and coverage calibration. Queued M4 calibration
  on step 3 seed 0 after 8b (`runs/m4_calib.log`).
- 2026-09-05 12:06: step 8b seed 0 reached val loss 2.183 versus
  2.487–2.514 for 7b, a 0.33-nat improvement. Argmax test MSE 1.095 was
  within the tail-driven noise. Queued robust 7b/8b comparison with
  median-window MSE, CRPS, and calibrated MC readout after M4 calibration
  (`runs/ett_compare.log`).
- 2026-09-05 12:12: step 8b completed. Validation loss 2.183 / 2.196
  versus 7b's 2.510 / 2.514, improving about 0.32 nats on both seeds.
  Mean argmax MSE 1.011 versus 1.081; seed 1 reached 0.9268 / 0.5926,
  better than the README. Recommended accepting step 8 subject to the
  robust comparison.
- 2026-09-05 12:13: M4 calibration failed because the script supplied
  truth shaped (48,) instead of (48,1). `crps_empirical` requires matching
  trailing dimensions. Fixed the script and requeued it after the 7b/8b
  comparison; no library-code change was needed.
- 2026-09-05 12:22: accepted steps 7, 8, and 18 after robust comparison.
  The multivariate configuration became LGT, mixture head, aligned windows,
  channel embeddings, and MC readout at validation-CRPS-calibrated temperature.
- 2026-09-05 12:27: M4 calibration completed without gain; keep T=1.
  Calibration helped the longer rollout; M4 coverage was 0.87 while ETT's
  had been 0.95.
- 2026-09-05 12:31: full suite passed, 234 tests. Started step 12 with
  two screening seeds: k=0 levels, no decoder integration, V=256/1024,
  LGT plus mixture. The optimized head made this cheap. The experiment
  tests the paper's claim that differencing supplies the inductive bias.
- 2026-09-05 12:34: user pointed out that k=0 cannot continue a trend.
  Verified and recorded issue 10: synthetic continuation error 3.96
  versus 0.0006 for k=1; out-of-domain levels at 1.86% of M4 Hourly points
  (about 25% of series) and 30.2% of exchange_rate points. Practical rule:
  choose k at least as large as the polynomial trend degree. Reframed
  step 12 as an option for stationary-level series; continued the runs
  to obtain measurements.
- 2026-09-05 12:36: step 12, k=0, V=256, seed 0: argmax
  16.77 / 3.101, median **15.51** / 2.976. sMAPE was the best so far,
  MASE the worst, against accepted 16.91 / 1.276. Holding the mean level
  gives about MASE 3.1 and sMAPE 19% here. Median statistics over 414 series:
  seasonal-naive error 19.6, deviation from level 45.1, level 83.4.
  Order zero largely reproduced mean reversion and lost the daily cycle.
  OWA 1.04 versus 0.726 illustrates why sMAPE alone would mislead.
- 2026-09-05 12:43: the linear probe supported step 11. At k=0,
  V=1024, val_loss was 3.13 versus 1.53 for V=256, as expected from a
  finer grid and higher token entropy. Compare forecast metrics rather
  than raw CE across different vocabulary sizes.
- 2026-09-05 12:47: implemented step 11. Features are deterministic
  functions of token history, computed inside `GPT.forward`; the KV cache
  retains cross-step history via `KVCache.record_tokens`. Engine,
  loaders, and training did not need interface changes. Zero-initialized
  `feature_proj` starts as a no-op. Features: token integral, level change
  over a lag, and the lagged token. Nine tests cover incremental/full
  forward equivalence and interaction with channel embeddings. Started
  `M4_FEATURE_LAGS=24` on four seeds.
- 2026-09-05 12:49: step 12 completed and was rejected. V=256:
  15.82 / 3.127, OWA 1.083; V=1024: 16.87 / 5.646, OWA 1.638;
  accepted configuration: 16.91 / 1.276, OWA 0.726. Increasing V worsened
  results despite a finer grid. The experiment supports differencing
  as a useful inductive bias on this task. Step 11 was running on four seeds.
- 2026-09-05 13:08: step 11 completed and was accepted. Four-seed
  median-readout results: sMAPE 14.40 ± 0.38, MASE 1.130 ± 0.020,
  OWA 0.628 versus 0.726 previously and README 0.822. Validation loss
  improved by 0.082 nats on every seed. Supplying exactly computable
  quantities helped as predicted by the probe. Started Bayes-readout
  evaluation (`runs/readouts11.log`).
- 2026-09-05 13:20: step 11 readout evaluation: MC OWA-Bayes
  13.09 / 1.026 and 13.13 / 1.009, OWA 0.570 / 0.568;
  CRPS 196 / 185, MSIS 7.17 / 7.32, coverage 0.787 / 0.748,
  NLL 1.349 / 1.438. Recorded the summary table. Started M4 Daily with
  3040 filtered series, context 78, lag 7: about 61 tokens per parameter
  versus 2.2, as an initial larger-data experiment.
- 2026-09-05 13:28: completed step 16, forwarding `seed` through
  `predict_samples`, `predict_point`, `predict_quantiles`, and
  `calibrate_temperature`. Calibration intentionally shares a seed across
  temperatures to reduce paired-comparison variance. Corrected Daily
  MASE reporting: M4 uses seasonality 1 for Daily, while the script used 7,
  understating MASE by a factor of 2.71. Hourly was unaffected. Comparing
  Daily with Hourly also confounded data volume with task difficulty.
  Started a within-Daily scaling curve: train on 100/400/1500/3040 series,
  evaluate on the same 3040-series test set.
- 2026-09-05 13:34: M4 Daily completed at OWA 1.00, approximately Naive2.
  Full suite: 243 tests passed. The within-Daily scaling curve was running.
- 2026-09-05 13:40: scaling results were essentially flat: 100 series
  gave 2.99 / 3.351, 3040 gave 2.99 / 3.349. Forecasts were exactly flat.
  The domain analysis associated the remaining roughly 2% gap to Naive2
  with a quantization half-bin 3.8 times the typical step. Implemented
  quantile domains and prepared q99.99/q99.9 runs; the entry initially
  described them as launched, corrected by the final entry below.
- 2026-09-05 13:52: stopped at the user's request. The scaling curve
  finished and was flat. **Not run; pending continuation:** Daily robust
  domains (`M4_DOMAIN_QUANTILE=0.9999` and `0.999`, implementation ready
  and tested), step 11 on ETT, ETT model-capacity tuning, and M4 Monthly
  as the proposed scaling experiment.

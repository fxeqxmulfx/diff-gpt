# TODO

Bugs and loose ends from the code review of 2026-10-19 (commit abcecd9),
most important first. Each item was confirmed by a script or by reading the
code unless it is marked *unconfirmed*. "Audit #n" refers to the encoder audit
in [encoder_decoder_issues.md](encoder_decoder_issues.md).

Tests for the fixes use real objects (no mocks, no monkeypatch) and must fail
on the unfixed code.

## P1: wrong forecasts, wrong training signal

- [x] **predict decodes from the start of the context, so one clipped context
  step shifts the whole forecast** (audit #1, "effect outside this module")
  - Where: `DiffGPT._generate_future_samples`
    ([diff_gpt.py:288](src/diff_gpt/diff_gpt.py#L288)) and
    `_predict_teacher_forced` ([diff_gpt.py:666](src/diff_gpt/diff_gpt.py#L666))
    decode context and generated tokens as one stream from `start`, the first
    `max_k` context rows. `predict`, `predict_samples`, `predict_quantiles`,
    `predict_point` and `calibrate_temperature` all go through them.
  - Repro: V=64, k=1, domain [1.0], context [0, .5, 1, 4, 4.5, 4.5, 4.5, 4.5],
    a sampler that always returns token V/2 (a zero step). The forecast is 2.516
    instead of 4.5. With k=2 and context [0, 0, 0, 3, 6, 9, 10, 11, 12] it is
    3.07, 3.03, 3.00, 2.97 instead of 13, 14, 15, 16. Teacher-forced: a +3 level
    shift of a covariate inside the context returns its future as
    [1.03, 1.32, 1.52, 1.64] instead of the supplied [3.37, 3.66, 3.87, 3.98].
  - Reach: M4 with `M4_DOMAIN_QUANTILE`. Daily at 0.999: 20 of 414 contexts
    are off by more than three quantization steps, by up to 2.1 series std.
    Hourly at 0.9999: 2 of 414, up to 0.33 std. With the default max domain,
    M4 and the four ETT sets stay within quantization.
  - Fix: decode only the generated tokens, anchored at the last `max_k` context
    rows. `encode` of those rows yields a `start` in the right form for both
    tokenizers and for scalar and per-column orders. Slice them as
    `context[len(context) - max_k:]`: `context[-0:]` is the whole context.
  - Test: a context with an out-of-domain jump and a zero-step sampler (a real
    `Sampler` subclass): the forecast stays at the last context value.

- [x] **LGT at k=1 overflows the vocabulary on rounding ties, and every
  overflow lowers the decoded level for good** (new)
  - Where: `_round_levels` rounds half to even
    ([encoder_decoder.py:185](src/diff_gpt/encoder_decoder.py#L185)), and
    `_compute_scale` gives k <= 1 the scale (V-2)/(2D)
    ([encoder_decoder.py:171](src/diff_gpt/encoder_decoder.py#L171)). When s·x is
    exactly n + 0.5, a step of exactly D gives ΔL = V/2, that is token V, which
    the clip turns into V-1.
  - Repro: x = [3, 5], D = 2, V = 256 gives s = 63.5, levels 190 and 318,
    ΔL = 128, token 256 clipped to 255. On an integer walk with steps -2..2,
    T = 5000, 261 of 4999 steps clip and the decode error reaches 4.12, which
    is 523·Δ/2; MASH on the same data stays at Δ/2. Decimal mode is identical.
    k = 0 and k >= 2 stay within Δ/2 on tie-heavy data.
  - Reach: the default tokenizer with k = 1, which every benchmark uses, on
    integer or grid-valued data where ties are exact (D = 2 or 4, values on a
    0.5 grid). It breaks the README's "tight Δ/2 reconstruction bound … at
    any order k".
  - Fix: round half up, L = floor(s·x + 1/2), in float and in Decimal. Then
    |ΔL| <= (V-2)/2 at k = 1 (127 at most on the walk above), and the k = 0
    and k >= 2 bounds still hold because |ζ| <= 1/2. Tokens change only at
    exact ties. Update the `_round_levels` docstring.
  - Test: a k = 1 round trip on an integer walk with D = 2, in float and in
    Decimal, must stay within Δ/2. `test_lgt.py` uses random floats only,
    which never tie.

- [x] **The val target mask scores a covariate when `align_windows=False`,
  the default** (the fix of audit C is incomplete)
  - Where: in `DiffDataFrameDataLoader` the phase `(local + 1) % F`
    ([data_loader.py:553](src/diff_gpt/data_loader.py#L553)) is relative to the
    slice, but only `align_windows=True` puts the start `n` of the val slice on
    a column boundary ([data_loader.py:498](src/diff_gpt/data_loader.py#L498)).
  - Repro: 3 columns, 335 rows, k=1, train_part=0.7 give n = 701 and
    701 % 3 = 2. Train keeps the target's tokens; val keeps the neighbouring
    column's (31 for target 0, 32 for target 1, 33 for target 2).
  - Reach: any `target_columns` with `train_part < 1`. Val loss, early stopping
    and `save_best` then follow a covariate, depending on n % F.
  - Fix: always round `n` down to a multiple of `n_features`, which is harmless
    for unaligned windows, or add the slice offset to the phase.
  - Test: `test_target_mask_keeps_only_the_target_column_at_any_phase`
    ([test_data_loader_phase.py:57](tests/diff_gpt/test_data_loader_phase.py#L57))
    passes only because its split is a multiple of F by luck. Check the val
    split with n % F != 0.

- [x] **`train()` evaluates before the last step, so the last update is never
  scored** (new)
  - Where: the eval runs at the top of the iteration
    ([train.py:167](src/diff_gpt/train.py#L167)), before `optimizer.step()`
    ([train.py:214](src/diff_gpt/train.py#L214)).
  - Effect: with `save_best` and a val split, the last update is always
    discarded, and `max_iters=1` returns the untrained weights. Otherwise the
    returned loss is one step stale: `max_iters=20, eval_interval=10` returns
    0.9929, while the returned weights score 0.9105 on the same batch.
  - Fix: evaluate once more after the loop, and keep early stopping on the
    in-loop evaluations.
  - Test: `test_train_save_best_false_returns_final_loss`
    ([test_diff_gpt.py:776](tests/diff_gpt/test_diff_gpt.py#L776)) only checks
    that the loss is finite. Check that the returned loss belongs to the
    returned weights on a fixed val batch, and add the `max_iters=1` case.

- [x] **Multi-node `train_numa` fails whenever CUDA is visible** (new)
  - Where: [train_ddp.py:81-85](src/diff_gpt/train_ddp.py#L81) picks NCCL
    whenever `torch.cuda.is_available()`, whatever the model's device, and puts
    rank r on GPU `r % device_count`.
  - Effect: with a CPU model, as in the README's NUMA example, every worker
    fails with `No backend type associated with device type cpu`. With a CUDA
    model and two ranks on one GPU, NCCL fails with `Duplicate GPU detected`,
    although the comment at lines 78-80 says the single-GPU case works. Only a
    machine without CUDA, or one GPU per rank with the model already on it,
    works. Tests cover only the single-node path.
  - Fix: pick the backend from the model's device (gloo for CPU) and refuse
    NCCL when there are more ranks than GPUs.
  - Test: the spawn path with a CPU model and CUDA visible (`nodes=[0, 0]`
    reaches it on one node), marked slow.

## P2: crashes, hangs, silent misuse

- [x] **`DiffSQLiteDataLoader`** (new)
  - `get_batch` loops forever when no table has room for a window in the
    requested split ([data_loader.py:276](src/diff_gpt/data_loader.py#L276)).
    Tables are filtered on their total length, but train windows are drawn
    from [0, train_end - block_size). Repro: 100 rows by 2 features, k=1,
    block_size=150, train_part=0.7. Raise instead, as the DataFrame loader does.
  - `use_decimal=True` always fails: the rows go to `encode` as float64,
    without `np_to_decimal` ([data_loader.py:215](src/diff_gpt/data_loader.py#L215)).
  - Tables of different widths fail with the real encoder, because one
    `domain_of_definition` serves every table
    ([data_loader.py:221](src/diff_gpt/data_loader.py#L221)). The mixed-width
    tests in `test_data_loader.py` use a pass-through encoder.

- [x] **Negative `target_columns` silently disable sampling** (new).
  `iloc[ctx_len:, -1] = 0.0` zeroes the target's future, but
  `p % n_features` is never -1 ([diff_gpt.py:587](src/diff_gpt/diff_gpt.py#L587),
  [diff_gpt.py:631](src/diff_gpt/diff_gpt.py#L631),
  [diff_gpt.py:652](src/diff_gpt/diff_gpt.py#L652)), so every position is forced
  to the zero placeholder. Normalise negative indices and reject out-of-range
  ones.

- [x] **With `use_decimal=True`, predict's output dtype depends on the path**
  (new). The teacher-forced path returns Decimal objects
  ([diff_gpt.py:674](src/diff_gpt/diff_gpt.py#L674)), the autoregressive path
  float64 ([diff_gpt.py:285](src/diff_gpt/diff_gpt.py#L285)).
  `benchmarks/diff_bench_decimal.py:117-120` subtracts the float forecast from
  Decimal truth and raises `TypeError` after all three training runs. Return
  one dtype.

- [x] **A context of exactly `max_k` rows passes the guards** (new). They use
  `>` where `>=` is needed ([diff_gpt.py:249](src/diff_gpt/diff_gpt.py#L249),
  [diff_gpt.py:379](src/diff_gpt/diff_gpt.py#L379),
  [diff_gpt.py:606](src/diff_gpt/diff_gpt.py#L606)). The call then fails inside
  the Engine ("tokens must be non-empty") or in `anomaly_scores` ("cannot
  reshape tensor of 0 elements").

- [x] **DDP aborts or hangs when `has_val()` differs across ranks** (new). Each
  rank decides it locally ([train.py:155](src/diff_gpt/train.py#L155)), so a
  rank with a val split runs one more all-reduce. Gloo aborts with a collective
  mismatch; NCCL would hang until its timeout. A `loader_factory(rank, world)`
  that shards data by rank reaches this. Agree on `has_val` across ranks or
  raise.

- [x] **The encoder accepts NaN and inf** (new). `get_domain_of_definition`
  returns NaN, `_compute_scale` then treats the column as constant (scale
  2^20), and LGT emits garbage tokens (for example [255 255 0 255]) with only
  numpy's "invalid value encountered in cast" warning. Raise `ValueError` in
  `encode` and in `get_domain_of_definition`. The float LGT path likewise breaks
  when |s·x| leaves the int64 range, e.g. for a constant column at |x| > 2^43.

- [x] **Clipping is silent** (audit #1). `encode` gives no count or warning when
  tokens clip, and every clipped step leaves a level error after integration.
  Report the number of clipped tokens. For MASH at k=1 the audit has a
  prototype that saturates inside the feedback loop.

- [x] **NUMA pinning** (new; confirmed by simulation, since this machine has
  one node)
  - Without `os.sched_getaffinity` (macOS, Windows), `cpus_on_node` returns
    `[0]`, and `pin_to_node` sets torch to one thread
    ([numa.py:46-56](src/diff_gpt/numa.py#L46),
    [numa.py:69-75](src/diff_gpt/numa.py#L69)). The module docstring promises
    a silent no-op. `DiffGPT.train(pin_node=…)` and `train_numa` reach this.
  - The thread count is the node's CPU count, not the number of CPUs the
    process may use (cpuset containers, SLURM). If none of the node's CPUs is
    allowed, `sched_setaffinity` raises `OSError`. Memory-only nodes (CXL, HBM)
    fall back to the full affinity and overlap the other ranks. Intersect with
    the affinity at start, and skip nodes without allowed CPUs.

- [x] **The tokenizer is not stored with the model** (new). `DiffGPT` and the
  loaders take `DIFF_GPT_TOKENIZER` at import
  ([encoder_decoder.py:102](src/diff_gpt/encoder_decoder.py#L102)), so a
  checkpoint trained with `mash` decodes silently wrong in a default process.
  Make the tokenizer a `DiffGPT` and loader parameter.

## P3: edge cases and broken contracts

- [x] **MASH with a scalar k=0 goes through the carry scheme** (audit #2).
  [encoder_decoder.py:596](src/diff_gpt/encoder_decoder.py#L596) does this, while
  the order `[0]` rounds plainly
  ([encoder_decoder.py:507](src/diff_gpt/encoder_decoder.py#L507)). The tokens
  differ (467 of 2000), and the error reaches 1.97·Δ/2, against README
  Theorem 1. Use the per-column k=0 branch; the sin/cos k=0 threshold can then
  be halved.
- [x] **`integrate` does not check the form of `start`** (audit #3).
  `differentiate(x, 2)` followed by `integrate(start, diff, np.array([2, 2, 2]))`
  is off by 998 without an exception, although the `differentiate` docstring
  ([encoder_decoder.py:312](src/diff_gpt/encoder_decoder.py#L312)) says
  `integrate` detects the form. Use the raw prefix on both paths, or validate.
  `DiffGPT` always passes matching forms.
- [x] **`KVCache.prefill` compares capacities.** It checks the destination's
  capacity against the source's capacity, not against `other.pos`
  ([kv_cache.py:96](src/diff_gpt/model/kv_cache.py#L96)). A grown source (2048
  slots, pos 5) cannot prefill a 16-slot cache. `Engine` does not reach this.
- [x] **`GPT.forward` uses `targets.view(B * T)`**
  ([gpt.py:194](src/diff_gpt/model/gpt.py#L194)), so `model(x[:, :-1], x[:, 1:])`
  raises. Use `reshape`.
- [x] **Samplers break their logits contract.** They promise logits of shape
  `[..., n_tokens]` ([sampler.py:18](src/diff_gpt/sampler/sampler.py#L18)), but
  `torch.multinomial` ([temperature.py:28](src/diff_gpt/sampler/temperature.py#L28))
  takes only 1-D or 2-D input. `TemperatureSampler(T > 0)` and `NucleusSampler`
  therefore fail on (2, 3, V). Flatten the leading dimensions or narrow the
  contract.
- [x] **`n_layer=0` never advances the cache.** `pos` advances only in the last
  layer's `insert_kv` ([kv_cache.py:152](src/diff_gpt/model/kv_cache.py#L152)),
  so with no blocks the cache stays at position 0. Cached logits then differ by
  0.38, and `Engine.generate` fails. Reject `n_layer < 1`; only
  `M4_N_LAYER=0` or `ETT_N_LAYER=0` reaches this.
- [x] **`NucleusSampler(p=1.0)` is not an exact no-op.** The float32 cumsum
  reaches 1 early and drops up to 4e-7 of the mass
  ([nucleus.py:38](src/diff_gpt/sampler/nucleus.py#L38)). Short-circuit p >= 1.
- [x] **`Engine.generate` clamps silently.** It caps `max_tokens` at
  `block_size - len(tokens)` without saying so, and returns a prompt longer
  than `block_size` unchanged ([engine.py:86](src/diff_gpt/model/engine.py#L86)).
  Raise or document; `DiffGPT` checks this itself.
- [x] **`train_numa` checkpoints of a `torch.compile`d model carry `_orig_mod.*`
  keys** ([train_ddp.py:98](src/diff_gpt/train_ddp.py#L98),
  [train_ddp.py:153](src/diff_gpt/train_ddp.py#L153)). A plain `GPT` cannot
  load them, although the comment says the keys match the bare class. Unwrap
  `_orig_mod` before saving.
- [x] **`get_domain_of_definition` disagrees with itself on short series**
  (audit #6). With a scalar k on T <= k rows it raises numpy's "zero-size
  array" error, while the per-column path returns 0. Raise a clear
  `ValueError` on both paths.
- [x] **`np_is_decimal` fails on arrays without columns.** It reads
  `inp[0, 0]`, so an (n, 0) array raises `IndexError`
  ([encoder_decoder.py:40](src/diff_gpt/encoder_decoder.py#L40)).
- [x] *Unconfirmed, needs two GPUs:* **`device_type` drops the device index.**
  It returns `"cuda"` without the index
  ([gpt.py:246](src/diff_gpt/model/gpt.py#L246)), while the Engine
  ([engine.py:100](src/diff_gpt/model/engine.py#L100)) and `anomaly_scores`
  build tensors on the current device. A model on `cuda:1` should fail in
  prefill.
- [x] *Unconfirmed, read only:* **`m4_eval.py` ignores two settings.**
  `prepare()` in `benchmarks/m4_eval.py` ignores `M4_DOMAIN_QUANTILE` and
  `M4_MIN_LEN`, which `m4_bench.py` uses. A checkpoint trained with either gets
  another domain (silent re-tokenization) or another block size.

## Docs

- [x] **AMSGrad docstring.** It says the effective per-coordinate learning rate
  is non-increasing
  ([adamw_schedulefree.py:58-62](src/diff_gpt/optimizer/adamw_schedulefree.py#L58)).
  Bias correction makes it rise, by 4.4x over 60 steps with betas (0.9, 0.95).
  The update itself matches torch's AMSGrad; only the claim is wrong.
- [x] **`_sigma_delta_k` docstring.** The "fast vectorized" path loops over T in
  Python (audit #8). Decimal is said to be "needed for k >= 5", but float64
  already degrades at k = 4 from T of about 10^4 (audit #4).
- [x] **`train(optimizer=…)` after a best-snapshot restore.** The optimizer's
  state (z, exp_avg_sq, k) comes from the last step, not the snapshot, so
  resuming starts from y = β1·x_best + (1 - β1)·z_last. That is a valid
  schedule-free state and did no harm in a small run, but the docstring's
  "matching the weights handed back" holds only for the mode. Say so.
- [x] **[encoder_decoder_issues.md](encoder_decoder_issues.md).**
  - Add the LGT k=1 tie finding.
  - Drop appendix B: the attention residuals left with the move to the plain
    GPT.
  - Note that 9a now raises `ValueError`, not `AssertionError`.
  - Mark items fixed as the fixes land.
- [x] **README.** "Tight Δ/2 … at any order k" holds for LGT only after the k=1
  fix, and Theorem 1 holds for MASH at k=0 only after audit #2.

## Token scheme version 2 (2026-10-20)

- [x] **One level grid for every k, for both tokenizers.** L = floor(s·x - B
  + 1/2), q = Δ^k L + V/2 and s = (V - max(2, 2^k))/(2D), with the anchor
  B = 0 (LGT) or the polynomial of degree < k through the prefix (MASH). The
  Σ-Δ cascade, the k <= 1 carry scheme and the k = 1 saturation are gone:
  every token lies in [1, V-1], the error stays within D/(V - max(2, 2^k)),
  and a clipped step offsets the values after it with either tokenizer.
  Checkpoints record `encoder_version` 2, and `load` refuses other versions.
- [x] **float64 and int64 only.** The MASH anchor enters through its
  fractional part, continued modulo 2^64 in uint64, so the exact anchors no
  longer need Python ints; MASH encodes about 50x faster.
- [x] **Decimal removed**: `use_decimal`, `np_to_decimal`, `np_is_decimal`
  and `benchmarks/diff_bench_decimal.py`. The Decimal items above are moot.
- [x] **Guards.** `encode` raises where |s·x| reaches 2^min(53, 62 - k):
  past 2^53 float64 misses levels (one sin/cos period sampled every 10^-6
  clipped over a thousand tokens at k = 3), past 2^(62 - k) the k-th
  differences overflow int64. `decode` raises on tokens whose levels leave
  int64.
- [x] **Lean.** `lean/DiffGPT/LevelGrid.lean` proves the scheme in exact
  arithmetic; `Float64.lean`, `Int64.lean` and `Machine.lean` model float64
  rounding and int64 wrapping and prove the implementation: exact half-up
  rounding, the uint64 anchor, the overflow test in `_cumsum`, the decoded
  levels, and the float64 token range up to the rounding of s·x, which
  version 3 removed. 62 theorems, standard axioms only.

## Token scheme version 3 (2026-10-20)

- [x] **Exact levels.** `_levels` reads floor(s·x - F + 1/2) off the
  float64 rounding of s·x and its exact error (Dekker's product), the scale
  rounds down and the domain max rounds up, so a step within D never clips,
  at rounding ties included. Version 2 clipped such steps on constructed
  series; a k = 2 one drifted to 1999·Δ/2 over 1000 rows. Checkpoints
  record `encoder_version` 3.
- [x] **Lean.** `ExactLevels.lean` proves `_levels` exact, and
  `encode_token_mem` in `Machine.lean` the token range of `encode` as
  written, with no float64 slack. 81 theorems, standard axioms only.
- [x] **Lean end to end.** `encode_decode` in `Machine.lean` takes a column
  from its floats to the decoded values: every token `encode` emits is its
  int64 token sum and lies in [1, V-1], `decode`'s `_integrate_levels` does
  not raise, and every decoded value lies within 1/(2s) +
  2^-52·(1 + 2^-54)·(|x| + 1/(2s)) + 2^-1075·(1 + (1 + 2^-53)/s) of x. The
  token range needs only the float64 guard 2^53, since int64 wraps modulo
  2^64; the cap 2^(62 - k) keeps the differences `decode` integrates
  through in int64, and the token sum of a step outside the domain exact,
  so it clips toward its own side (`toInt_tokenSum`). 105 theorems,
  standard axioms only.

## Token scheme version 4 (2026-10-20)

- [x] **Operators c(B).** An order is an int k, for Δ^k = (1 - B)^k, or a
  `Difference(*lags)`, for c(B) = (1 - B^S_1)···(1 - B^S_m):
  `Difference(24)` is the daily seasonal difference of hourly data, and
  `Difference(1, 24)` the SARIMA one, (1 - B)(1 - B^24). Version 3 carries
  over with ‖c‖₁ = Σ|c_j| in place of 2^k: the levels, the tokens
  c(B) L + V/2, s = (V - ‖c‖₁)/(2D), the exact max of |c(B) x|, the
  decoding by one `_cumsum` per factor, exact in int64, and the MASH anchor
  A with c(B) A = 0 through the p = S_1 + ... + S_m prefix rows. The guard
  is 2^min(53, 62 - m) for m factors. `DiffGPT` and the loaders keep the p
  rows of the largest degree in front of the tokens (`prefix_rows`), and
  checkpoints record each column's lags.
- [x] **The order 0 uses all V tokens.** It takes A = 1/2, so the levels
  are floor(s·x), and s·D < V/2: the tokens lie in [0, V-1], and the error
  1/(2s) is a hair above D/V, the least that any V values allow on
  [-D, D]. Version 3 left token 0 unused, with the error D/(V - 2).
- [x] **Lean.** `LevelGrid.lean`, `Int64.lean` and `Machine.lean` take the
  list of lags: Theorem 1 with ‖c‖₁ (`abs_diffs_levels_le`), the sharpness
  of the scale (`exists_levels_ge`), the decoding in int64
  (`toInt_invertI64`), the uint64 anchor (`anchor_numerators`) and
  `encode_decode` for any c. `encode_decode_zero` proves the order 0 end to
  end, and `le_mul_of_forall_exists_abs_sub_le` its optimality.
  `Optimality.lean` proves Theorem 2: for V >= 2‖c‖₁ no encoder for this
  decoder, whatever the scale, the prefix and the anchor, keeps every series
  within less than D/(V - ‖c‖₁) (`le_mul_of_encodes`, `isLeast_encodes`).
  139 theorems, standard axioms only. Checkpoints record `encoder_version` 4.

## Observations, no action planned

- Above the order 0 token 0 is never emitted for in-domain input (audit
  #7), and no larger scale could use it: one pushes a token past V - 1
  (`exists_levels_ge`). No other encoder gains by it either (Theorem 2).
  The order 0 uses it since version 4.
- MASH tokens depend on the window's prefix, the anchor of its grid, so a
  separately encoded window differs from a slice (audit #5). LGT, the
  default, does not have this.
- **Checked and correct:**
  - the optimizer matches schedulefree 1.4.1 bit for bit with `amsgrad=False`;
  - cached and uncached logits agree, and the Engine reproduces uncached
    generation for every sampler and seed;
  - the metrics match their definitions;
  - window starts and split boundaries are exact in all three loaders;
  - predict's row alignment stays within the bound for both tokenizers and
    every kind of order when nothing clips.

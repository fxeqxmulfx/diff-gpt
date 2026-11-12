# Stress tests for GPT: the 165-case sweep

This is a task-by-task guide to the original 165-case grokking and generalization
sweep. It includes the tasks outside the later 35-task short/long benchmark,
negative controls, forecasting and language tasks, and cases that were never
completed. The case IDs below are the exact names in
[`research/current_grokking/plan.json`](research/current_grokking/plan.json).

The original sweep used the then-current public GPT at commit
`01f18f86787511279fe52fe41883495fce0e3105`, with ordinary AdamW. Each run trained
a fresh model on one task. This is a historical benchmark specification: it
does **not** establish that the current
Rust/Kernite main model has passed all 165 cases. The current model and its
separate 35-task results are documented in
[`research/main_model/README.txt`](research/main_model/README.txt).

## How to read this catalog

- Each row is one case. A family may contain several cases that differ by
  modulus, depth, encoding, noise level, or answer format.
- `D/L/H` means model width, number of layers, and number of heads. Budgets are
  optimizer updates, not epochs or wall-clock limits.
- Unless stated otherwise, the historical sweep used data seed 1, model seed 1,
  AdamW with betas `(0.9, 0.98)`, epsilon `1e-8`, weight decay `0.1`, a 50-update
  linear warmup and the learning rate in the table below. Its first update had
  zero learning rate. The fixed training sampler and data fingerprints were
  recorded with each run.
- After the first observed task target, the historical run continued until the
  first scheduled observation at or after 500 additional updates, capped by the
  original budget. A target crossing, final retention, and delayed
  generalization are separate observations. A budget exhaustion is a bounded
  miss, not proof that the task can never be learned.
- Most cases observe every 250 updates; Basis recall observes every 50. Series
  cases observe once per complete training epoch. Original task targets differ:
  95% accuracy, 99% accuracy, numeric error, cross-entropy gain and skill are
  **not** interchangeable.
- The `History` column in the task tables records the older GPT's saved
  seed-1 status: `passed` means its declared task-specific target was observed;
  `budget` means the update budget was exhausted; `pending`, `skipped` and
  `stopped` mean no completed result. These are the retained snapshot statuses,
  not claims about the current model. The `Tier` column identifies the task's
  later 35-task `short` or `long` membership; `—` means it was not selected.
- The primary held-out split is validation, not necessarily an independent
  final test set. For generative synthetic tasks, evaluation uses free generation;
  a novel validation input/prompt is preferred when available. Non-generative
  tasks score the supervised tokens in their declared held-out split.

| Cases | D/L/H | Batch | LR | Maximum updates | Declared target |
| --- | --- | ---: | ---: | ---: | --- |
| Arithmetic | 128/1/4 | 256 | 0.001 | 100,000; random table 10,000 | 99% exact sequences |
| Modular chains | 128/4/4 | 256 | 0.001 | 100,000 | 99% exact sequences |
| Synthetic algorithms | 64/2/4 | 32 | 0.001 | 100,000; random LM 10,000 | 95% exact sequences |
| Basis and languages | Varies by case below | Varies | 0.001 or 0.0003 | 3,000–100,000 | 99% exact sequences |
| Differenced division | 128/1/4 | 256 | 0.001 | 100,000 | 99% exact sequences |
| Recurrences | 128/1/4 for division; 128/1/1 for Fibonacci | 512 | 0.001 | 100,000 | 99% decoded accuracy |
| Associative arithmetic | 128/2/4 | 256 | 0.001 | 100,000 | 99% exact sequences |
| Weather and forecasting series | 64/1/4 | 512 | 0.001 | 30,000 | Skill 0.9 or relative CE gain 0.1 |
| Noisy arithmetic and recurrence series | 128/1/4 for division; 128/1/1 for Fibonacci | 512 | 0.001 | 100,000 | Accuracy 0.99 |
| GPT probes | 64/2/4 | 32 | 0.001 | 30,000 | Numeric MSE at most 0.001, or 99% exact sequences |
| Fashion-MNIST | 64/2/4 | 64 | 0.001 | 30,000 | 90% classification accuracy |
| Tiny Shakespeare | 64/4/4 | 32 | 0.001 | 10,000 | Validation cross-entropy at most 1.63 |

The historical [progress snapshot](research/current_grokking/results/progress_snapshot.csv)
records one planned seed per case: 48 cases reached their own declared target,
78 exhausted their budget, 29 were pending, nine were skipped at the user's
request, and one was stopped. Of the 48 target crossings, 40 retained their
target at the final observation and eight fell below it. The per-case
`final_target_met` field records this distinction. Those counts do not mean
48 cases reached 99%: targets and metrics vary by task.

The later selection retained 35 nonconstant tasks with a recorded seed-1
held-out 99% crossing within five minutes, including setup. The short tier has
25 tasks at or below one minute; the long tier has ten tasks over one and up to
five minutes. The thresholds selected the tasks; they are not time limits for
new runs. The [historical qualifying list](research/current_grokking/results/historical_99_within_five_minutes_seed1.csv)
also contains three constant-output power-zero controls, excluded from those 35.

## Arithmetic functions: 40 cases

All cases in this section use `p = 97`, inputs `a, b` as individual residue
tokens and only the final answer as the supervised token. Pairs are shuffled
with data seed 1 and split into disjoint halves: 4,704 train and 4,705 validation
pairs for the full 97² tables, or 4,656 each for division. Division excludes
`b = 0`; the other operations include it. Reductions are
modulo 97 except where integer operations are explicitly named. In the table,
`r = (a + b) mod p` and `c = (a³ + ab) mod p`. The
[arithmetic registry](../../rust_projects/grokking/experiments/training/src/training/domain/tasks.py)
defines the exact fixed permutations, random tables and sets.

For the set-membership rows, `I₀ = {0, …, (p−3)/2}` and `Iⱼ` exchanges `j`
deterministically selected members of `I₀` with `j` nonmembers. `R` is a fixed
random set of `(p−1)/2` residues; `Q` is the set of nonzero quadratic residues.

| Case | Task and target | History | Tier |
| --- | --- | --- | --- |
| `arithmetic-div` | Predict `a × b⁻¹ mod p` for nonzero `b`. | passed | short |
| `arithmetic-abs-diff` | Predict the ordinary integer distance `abs(a − b)`. | passed | long |
| `arithmetic-abs-diff-blind` | Predict `floor(abs(a − b) / 2)`, hiding the lowest answer bit. | passed | long |
| `arithmetic-add` | Predict `(a + b) mod p`. | passed | short |
| `arithmetic-affine-product` | Predict `(a + ab) mod p`. | passed | short |
| `arithmetic-cube-plus` | Predict `(a³ + b) mod p`. | passed | short |
| `arithmetic-cubic-mixed` | Predict `(a³ + ab² + b) mod p`. | budget | — |
| `arithmetic-cubic-product` | Predict `(a³ + ab) mod p`. | budget | — |
| `arithmetic-floor-product` | Predict the integer quotient `floor(ab / p)`. | budget | — |
| `arithmetic-half-sum` | Predict `(a + b) × 2⁻¹ mod p`. | passed | short |
| `arithmetic-max` | Predict the ordinary integer `max(a, b)`. | budget | — |
| `arithmetic-mixed` | Predict modular `a / b` when `b` is odd; predict `(a − b) mod p` when `b` is even. | budget | — |
| `arithmetic-mul` | Predict `ab mod p`. | passed | short |
| `arithmetic-parity-cubic` | Predict `[2 floor(c / 2) + ((a + b) mod 2)] mod p`. | budget | — |
| `arithmetic-parity-cubic-blind` | Predict the integer floor of the preceding parity-cubic answer divided by two. | budget | — |
| `arithmetic-perm-add` | Predict `π(r)` for a fixed random permutation `π` of the residues. | passed | short |
| `arithmetic-quadratic` | Predict `(a² + ab + b²) mod p`. | budget | — |
| `arithmetic-quadratic-linear` | Predict `(a² + ab + b² + a) mod p`. | budget | — |
| `arithmetic-random` | Predict entries of a fixed random answer table; this is a memorization/control task with a 10,000-update budget. | budget | — |
| `arithmetic-square-product` | Predict `(a² + ab) mod p`. | budget | — |
| `arithmetic-square-sum` | Predict `(a² + b²) mod p`. | passed | short |
| `arithmetic-square-times` | Predict `a²b mod p`. | passed | short |
| `arithmetic-sub` | Predict `(a − b) mod p`. | passed | short |
| `arithmetic-sum-cube` | Predict `r³ mod p`. | passed | long |
| `arithmetic-sum-interval` | Predict the indicator `1[r ∈ I₀]`. | budget | — |
| `arithmetic-sum-interval-12` | Predict `1[r ∈ I₁₂]`, the interval with 12 membership swaps. | budget | — |
| `arithmetic-sum-interval-2` | Predict `1[r ∈ I₂]`, the interval with two membership swaps. | budget | — |
| `arithmetic-sum-interval-4` | Predict `1[r ∈ I₄]`, the interval with four membership swaps. | budget | — |
| `arithmetic-sum-interval-8` | Predict `1[r ∈ I₈]`, the interval with eight membership swaps. | budget | — |
| `arithmetic-sum-pow12` | Predict `r¹² mod p`. | budget | — |
| `arithmetic-sum-pow16` | Predict `r¹⁶ mod p`. | budget | — |
| `arithmetic-sum-pow24` | Predict `r²⁴ mod p`. | budget | — |
| `arithmetic-sum-pow32` | Predict `r³² mod p`. | budget | — |
| `arithmetic-sum-pow4` | Predict `r⁴ mod p`. | passed | long |
| `arithmetic-sum-pow48` | Predict `r⁴⁸ mod p`. | budget | — |
| `arithmetic-sum-pow6` | Predict `r⁶ mod p`. | budget | — |
| `arithmetic-sum-pow8` | Predict `r⁸ mod p`. | budget | — |
| `arithmetic-sum-random-half` | Predict the indicator `1[r ∈ R]`. | budget | — |
| `arithmetic-sum-square` | Predict `r² mod p`. | passed | short |
| `arithmetic-sum-squares` | Predict the indicator `1[r ∈ Q]`. | budget | — |

## Modular and powered division: 14 cases

These use the same half-table split, token representation and settings as the
arithmetic cases. `k` here is an **exponent**, not a difference order. Each
powered case predicts `(a × b⁻¹)ᵏ mod p` with `b ≠ 0`. The three `k = 0`
cases have constant target 1, including for `a = 0`, and are controls.

| Case | Task | History | Tier |
| --- | --- | --- | --- |
| `division-p17` | Raw `a / b mod 17`. | budget | — |
| `division-p193` | Raw `a / b mod 193`. | passed | short |
| `power-division-p17-k0` | Power-zero control at modulus 17. | passed | — |
| `power-division-p17-k2` | Squared division at modulus 17. | budget | — |
| `power-division-p17-k3` | Cubed division at modulus 17. | budget | — |
| `power-division-p17-k5` | Fifth-power division at modulus 17. | budget | — |
| `power-division-p97-k0` | Power-zero control at modulus 97. | passed | — |
| `power-division-p97-k2` | Squared division at modulus 97. | passed | short |
| `power-division-p97-k3` | Cubed division at modulus 97. | passed | short |
| `power-division-p97-k5` | Fifth-power division at modulus 97. | passed | short |
| `power-division-p193-k0` | Power-zero control at modulus 193. | passed | — |
| `power-division-p193-k2` | Squared division at modulus 193. | passed | short |
| `power-division-p193-k3` | Cubed division at modulus 193. | passed | short |
| `power-division-p193-k5` | Fifth-power division at modulus 193. | passed | short |

## Synthetic algorithms and control: 38 cases

These use the source generators and scorers from the Lean transformer lab's
[synthetic suite](../../lean_projects/transformer/experiments/synthetic_amsgradw/experiment.py),
but train the historical public GPT described above. The standard suite draws
128 training, 32 validation and 64 test examples before the novel-input
selection. Most cases train on problem lengths 8–16 and have 32- and 64-length
probes; the Dyck variants train on lengths 12–16. MQAR and lookup use length 24
with 48- and 96-length probes; addition uses problem lengths 2–4 with 8- and
16-digit probes (individual operands can be shorter). The `random-lm-control`
has its own smaller data sizes below. For generative cases, full generations,
including termination, determine exact-sequence accuracy; separate final-answer
and teacher-forced scores may also be recorded. MQAR, lookup, Dyck,
alternating-block and C-RASP cases score all supervised positions of the supplied
input sequence.

Addition operands are always supplied most-significant digit first, padded to
the problem length plus one leading digit; the answer keeps that width.
`forward`/`reverse` changes the answer order. Hints precede each digit in both
the prompt and answer. Hard-carry probes also force a carry chain through every
unpadded digit at lengths 4, 8 and 16.

| Case | Task | History | Tier |
| --- | --- | --- | --- |
| `suite-mqar` | Recall values bound to queried keys: four key/value writes and two queries among filler tokens. | budget | — |
| `suite-lookup` | Follow a two-hop path through four key/value links for each of two queries. | budget | — |
| `suite-dyck` | Label each prefix of a one-type bracket string as balanced, still open or invalid. | passed | — |
| `suite-blocks` | Label each prefix for membership in the three alternating-block language, ignoring neutral symbols. | passed | — |
| `suite-histogram-bos` | Start with BOS, then generate the total frequency of the symbol at every input position. | budget | — |
| `suite-histogram-no-bos` | Generate the same per-symbol frequencies without the beginning-of-sequence marker. | budget | — |
| `suite-histogram2` | Start with BOS, then for every input position generate the number of distinct symbols with the same frequency as its symbol. | budget | — |
| `suite-mode-none` | Generate the uniquely most frequent symbol without a scratchpad. | budget | — |
| `suite-mode-counts` | Generate count/symbol pairs ordered by increasing count, ties by first occurrence, then the mode. | budget | — |
| `suite-mode-itemized` | Generate count/symbol pairs in order of first appearance, then the mode. | budget | — |
| `suite-most_freq` | Start with BOS, then generate distinct symbols in decreasing frequency, ties by first occurrence; pad that symbol list to the input length with BOS. | budget | — |
| `suite-copy-repeat` | Copy an input drawn from two symbols selected for each example; symbols may repeat. | budget | — |
| `suite-copy-unique` | Copy an input sampled without repeated symbols. | budget | — |
| `suite-reverse-repeat` | Reverse an input drawn from two symbols selected for each example; symbols may repeat. | budget | — |
| `suite-reverse-unique` | Generate the reverse of an input with unique symbols. | budget | — |
| `suite-sort-repeat` | Generate input symbols in increasing order, retaining repetitions. | budget | — |
| `suite-sort-unique` | Sort symbols sampled without replacement. | budget | — |
| `suite-dyck-2` | Label bracket-prefix status with two types of brackets. | budget | — |
| `suite-dyck-3` | Label bracket-prefix status with three types of brackets. | budget | — |
| `suite-count` | Given start and end integers, generate every integer in the inclusive interval. | budget | — |
| `suite-addition-forward-plain-standard` | Generate a decimal sum most-significant digit first, with no position hints and ordinary operand sampling. | budget | — |
| `suite-addition-forward-plain-balanced` | Generate a forward decimal sum without hints, with balanced carry-chain sampling. | budget | — |
| `suite-addition-forward-hints-standard` | Generate a forward decimal sum with digit-position hints and ordinary carry sampling. | budget | — |
| `suite-addition-forward-hints-balanced` | Generate a forward decimal sum with position hints and balanced carry chains. | budget | — |
| `suite-addition-reverse-plain-standard` | Generate a decimal sum least-significant digit first, without hints and with ordinary sampling. | budget | — |
| `suite-addition-reverse-plain-balanced` | Generate a reversed decimal sum without hints and with balanced carry chains. | budget | — |
| `suite-addition-reverse-hints-standard` | Generate a reversed decimal sum with position hints and ordinary sampling. | budget | — |
| `suite-addition-reverse-hints-balanced` | Generate a reversed decimal sum with position hints and balanced carry chains. | budget | — |
| `suite-parity-none` | Generate the parity of the input bits with no scratchpad. | budget | — |
| `suite-parity-running` | Generate an initial EVEN token, then the running parity after every bit, then EOS. | passed | — |
| `suite-parity-running-hints` | Generate an initial EVEN token, then running parity after every bit with matching index hints, then EOS. | budget | — |
| `suite-parity-ones-hints` | Generate an initial EVEN token, then parity only after one-bits with matching index hints, then EOS. | budget | — |
| `suite-boolean-and-shift` | Decide whether all bits are one; with probability 0.5 insert one zero before the last quarter, with late-position transfer probes. | passed | short |
| `suite-boolean-and-random` | Decide whether all bits are one; with probability 0.5 insert one zero at any position. | passed | short |
| `suite-crasp-depth-1` | Label each prefix with a fixed nonconstant past-counting formula of syntactic count depth 1. | passed | — |
| `suite-crasp-depth-2` | Label each prefix using a fixed nonconstant past-counting formula of syntactic count depth 2. | budget | — |
| `suite-crasp-depth-3` | Label each prefix using a fixed nonconstant past-counting formula of syntactic count depth 3. | budget | — |
| `random-lm-control` | Generate eight independent uniform symbols from a four-symbol alphabet given a length prompt; no input-to-answer rule exists. Uses eight training, 64 validation and 128 test examples, plus length-16/32 probes. | budget | — |

The C-RASP depths describe the selected formulas' syntax, not a proven minimal
depth for an equivalent formula. The random-LM case is a memorization control,
not an ordinary deterministic generalization task.

## Basis depth, recall, parity and languages: 14 cases

The underlying [Basis specification](../../lean_projects/transformer/python/src/lab/domain/basis.py)
uses 20,000 train, 512 validation and 512 test rows, except the explicitly
noisy parity case. Depth tasks label every prefix for membership in `Eₖ`: after
neutral symbols are removed, the string consists of `k` alternating blocks
starting with `a`. Easy depth selects on training-length validation; hard depth
selects a length-128 validation probe after training on lengths 32–64. Recall
uses 256 possible keys and 256 possible values. Parity asks whether the number
of ones is odd, without a scratchpad.

| Case | Task and historical configuration | History | Tier |
| --- | --- | --- | --- |
| `basis-depth-easy` | `E₂`, training lengths 32–64; D64/L2/H4, batch 16, LR 0.001, 20,000 updates. | passed | short |
| `basis-depth-hard` | `E₄`, training lengths 32–64 and selection at length 128; D128/L6/H4, batch 16, LR 0.001, 100,000 updates. | passed | short |
| `basis-recall-easy` | Eight distinct key/value writes and eight queries in sequences of length 32–64; D64/L2/H4, batch 64, LR 0.0003, 20,000 updates. | passed | long |
| `basis-recall-hard` | Sixteen writes to eight keys, including eight overwrites, in sequences of length 48–64; answer eight queries with each key's latest value. D128/L6/H4, batch 64, LR 0.0003, 3,000 updates. | passed | — |
| `basis-parity16` | Parity of sequences of 1–16 bits; D64/L2/H4, batch 16, LR 0.0003, 100,000 updates. | passed | long |
| `basis-parity20` | Parity of sequences of 1–20 bits; same model, batch, LR and budget as parity16. | passed | long |
| `basis-parity24` | Parity of sequences of 1–24 bits; same model, batch, LR and budget as parity16. | passed | long |
| `basis-parity-noise20` | Parity of 1–16 bits with each training parity label flipped with probability 0.2, fixed by noise seed 2; 512 train, 128 validation and 256 test examples with disjoint inputs and clean held-out labels; D64/L2/H4, batch 64, LR 0.0003, 100,000 updates. | pending | — |
| `language-l1` | `E₁` with no neutral letters; D128/L6/H4, batch 16, LR 0.001, 100,000 updates. | pending | — |
| `language-l2` | `E₂` with no neutral letters; same model, batch, LR and budget as language-l1. | pending | — |
| `language-l3` | `E₃` with no neutral letters; same model, batch, LR and budget as language-l1. | pending | — |
| `language-l4` | `E₄` with no neutral letters; same model, batch, LR and budget as language-l1. | pending | — |
| `language-l5` | `E₅` with no neutral letters; same model, batch, LR and budget as language-l1. | pending | — |
| `language-l6` | `E₆` with no neutral letters; same model, batch, LR and budget as language-l1. | pending | — |

The L1–L6 cases inherit the hard-depth data protocol and its length-128
selection probe. They retain the historical GPT's positional encoding; they
are not implementations of a no-position theoretical model.

## Modular composition chains: 12 cases

The input is `[z₀, b₀, c₀, …]`. For each step, compute
`z ← (z³ + bᵢz + cᵢ) mod p`; predict only the final `z`. Intermediate values
are not supplied or supervised. Each case declares 2,048 unique training
inputs, 1,024 held-out inputs, and 1,024 separate inputs for a test with twice
as many composition steps (`1 + 4l` input tokens, up from `1 + 2l`).
See the [chain adapter](research/current_grokking/adapters.py) and
[source specification](../../lean_projects/transformer/python/src/lab/domain/modular_composition.py).

| Case | Task | History | Tier |
| --- | --- | --- | --- |
| `modular-chain-p17-l1` | One composition step modulo 17. | budget | — |
| `modular-chain-p17-l2` | Two composition steps modulo 17. | budget | — |
| `modular-chain-p17-l4` | Four composition steps modulo 17. | budget | — |
| `modular-chain-p17-l8` | Eight composition steps modulo 17. | skipped | — |
| `modular-chain-p97-l1` | One composition step modulo 97. | skipped | — |
| `modular-chain-p97-l2` | Two composition steps modulo 97. | skipped | — |
| `modular-chain-p97-l4` | Four composition steps modulo 97. | skipped | — |
| `modular-chain-p97-l8` | Eight composition steps modulo 97. | skipped | — |
| `modular-chain-p193-l1` | One composition step modulo 193. | skipped | — |
| `modular-chain-p193-l2` | Two composition steps modulo 193. | skipped | — |
| `modular-chain-p193-l4` | Four composition steps modulo 193. | skipped | — |
| `modular-chain-p193-l8` | Eight composition steps modulo 193. | skipped | — |

In the retained snapshot, the first three mod-17 cases exhausted their budgets.
The mod-17 length-eight case was stopped at 77,000 updates, and it and the
eight mod-97/mod-193 cases were marked skipped at the user's request; their
original entries and any checkpoints remain separate from completed passes.

## Associative arithmetic: eight cases

These cases combine key/value memory with arithmetic over the mod-97 field.
The `n` component is the number of loads, `l` is the number of computed rows,
and `w` is the number of overwrites. The adapter fixes complete contexts before
training. Dividend/divisor cases are operand-copying controls; they do not by
themselves demonstrate division. The random-label case is a negative control.
See [`mqar_div97.py`](src/diff_gpt/application/grokking/mqar_div97.py) and its
[copy-shortcut audit](src/diff_gpt/application/grokking/mqar_div97_copy_audit.py).

There are 32 possible keys. Division cases partition the operand equations;
copy controls partition entire memory maps and query both operands independently,
allowing repeated keys. Every computed row's answer is supervised. For multiple
computed rows, the saved exact-sequence score uses teacher forcing: later rows
can read the true values of earlier computations.

| Case | Task | History | Tier |
| --- | --- | --- | --- |
| `associative-dividend-n2-l1-w0` | Load two operands and report the queried dividend. | budget | — |
| `associative-divisor-n2-l1-w0` | Load two operands and report the queried divisor. | passed | long |
| `associative-division-n2-l1-w0` | Load two operands and compute one division. | budget | — |
| `associative-division-n8-l1-w0` | Load eight values and compute one division from retrieved operands. | budget | — |
| `associative-division-n8-l2-w0` | Load eight values and perform two computed rows. | stopped | — |
| `associative-division-n8-l4-w0` | Load eight values and perform four computed rows. | pending | — |
| `associative-division-n8-l1-w4` | Load eight values, make four overwrites, then compute one division. | pending | — |
| `associative-random-n2-l1-w0` | Two-load, one-computation control with random labels. | pending | — |

## Differenced division equations: two cases

Each row begins with a mod-97 division equation `q = a / b`, with nonzero `b`.
The 9,312 equations are shuffled and split into 4,656 train and 4,656
validation rows. An LGT codec with a fixed zero prefix encodes differences;
only the next encoded token is supervised. For underlying codec levels
`A, B, Q`, the actual transformed tokens before the shared vocabulary offset
are listed below. These are independent equations, not time-series recurrences.

| Case | Task | History | Tier |
| --- | --- | --- | --- |
| `differenced-division-k1` | Input `[A, B − A]`; predict `Q − B`, the first difference of the answer. | budget | — |
| `differenced-division-k2` | Input `[A, B − 2A]`; predict `Q − 2B + A`, the second difference of the answer. | budget | — |

The previous short-context differenced-division studies did not establish
successful generalization. See the codec and masks in
[`div97.py`](src/diff_gpt/application/grokking/div97.py).

## Temporal field recurrences: seven cases

These cases forecast a future value from a context of a mod-97 trajectory.
Division follows `x[t+2] = x[t] / x[t+1] mod 97` on nonzero pairs; Fibonacci
follows `x[t+2] = x[t] + x[t+1] mod 97`, excluding the absorbing `(0, 0)` state.
Whole trajectory orbits are split between training and validation. In these
IDs `k` is difference order, `c` is the number of encoded context tokens, and
the suffix is the tokenizer. Each window contains `c + k` raw past values and
one future value, without a synthetic zero prefix. The target is the decoded
next value; native-token accuracy is recorded separately. Fibonacci with LGT
uses the field readout; the MASH variants use native argmax.

| Case | Task | History | Tier |
| --- | --- | --- | --- |
| `recurrence-division-k1-c4-lgt` | Division recurrence, first differences, context 4, LGT codec. | budget | — |
| `recurrence-division-k2-c4-lgt` | Division recurrence, second differences, context 4, LGT codec. | budget | — |
| `recurrence-division-k1-c48-lgt` | Division recurrence, first differences, context 48, LGT codec; a period-copy shortcut is possible. | passed | — |
| `recurrence-fibonacci-k1-c2-lgt` | Fibonacci recurrence, first differences, context 2, LGT codec and field readout. | passed | long |
| `recurrence-fibonacci-k1-c2-mash` | Fibonacci recurrence, first differences, context 2, MASH codec and native argmax. | budget | — |
| `recurrence-fibonacci-k1-c2-mash-half` | Fibonacci recurrence, first differences, context 2, MASH-half codec and native argmax. | budget | — |
| `recurrence-fibonacci-k2-c2-lgt` | Fibonacci recurrence, second differences, context 2, LGT codec and field readout. | budget | — |

The long-context division result must be interpreted alongside the
[period-copy audit](src/diff_gpt/application/grokking/recurrence_period_probe.py).
The [recurrence adapter](research/current_grokking/adapters.py) uses the public
encode/decode implementation and records the orbit split.

## Weather equations, series, and noisy anchors: 22 cases

Weather-equation cases use two observed weather columns to predict a third on
held-out quantized input pairs. The physical equations describe the intended
relationships; the labels are measured dataset values, not newly synthesized
exact formula outputs. The codec is fit on a training-only calibration subset.
Series use 60/20/20 chronological segments with gaps between splits and
train-fitted codecs; the Gaussian cases use independent stationary realizations
for the three splits and an independent training-only calibration realization.
Shuffled-time controls shuffle independently inside each segment. Context `c`
counts encoded prior rows and `k` is difference order, requiring `c + k` raw
past rows. The series evaluator retains only held-out windows whose input and
target tokens occurred in the optimized training set, checked per column.

The primary series CE uses teacher forcing within the next row: later columns
can read earlier true columns of that row. Separate free-running forecasts for
chronological real datasets generate those columns and later rows, comparing
up to eight eligible test origins with persistence and seasonal-naive forecasts.
Those auxiliary forecast errors are not the declared pass criterion. See the
[forecast evaluator](research/current_grokking/forecast.py).

For a case with a known or estimated floor, the reported skill is
`(baseline CE − model CE) / (baseline CE − floor CE)` when defined. For real
series without that floor, relative CE gain is
`(baseline CE − model CE) / baseline CE`. The weather-equation floor is an
empirical diagnostic of the quantized table, not a universal Bayes guarantee.
Metrics are averaged over the dataset's columns where applicable.

The CE baseline is a smoothed marginal distribution fitted to training labels.
For lattice and Gaussian AR, held-out skill uses expected CE under the known
conditional token distribution. CE on realized targets is recorded separately.
The Gaussian oracle is also conditioned on the retained target-token coverage.
Weather equations and the four synthetic AR cases target skill 0.9; real series
and shuffled-time controls target relative CE gain 0.1. Noisy anchors target
accuracy 0.99 with argmax readout for division and field readout for Fibonacci.

| Case | Task | History | Tier |
| --- | --- | --- | --- |
| `weather-formula-VPact` | Predict actual vapor pressure from relative humidity and maximum vapor pressure; expected relation `VPact ≈ VPmax × rh / 100`. | budget | — |
| `weather-formula-VPdef` | Predict vapor-pressure deficit from maximum and actual vapor pressure; expected relation `VPdef ≈ VPmax − VPact`. | pending | — |
| `weather-formula-H2OC` | Predict water-vapor concentration from pressure and actual vapor pressure; expected relation `H2OC ≈ 1000 × VPact / p`. | pending | — |
| `weather-formula-sh` | Predict specific humidity from pressure and actual vapor pressure; expected relation `sh ≈ 622 × VPact / (p − 0.378 × VPact)`. | pending | — |
| `series-weather-k0-c4` | Forecast weather levels from four prior rows. | pending | — |
| `series-weather-k1-c4` | Predict the next encoded first-difference row from four encoded prior rows, formed from five raw past rows. | pending | — |
| `series-ar-k0-c1` | Forecast `x[t] = floor(0.75 x[t−1]) + ε[t]`, with independent uniform noise on `{−1, 0, 1}`, using context 1. | passed | — |
| `series-seasonal_ar-k0-c24` | Forecast the same lattice process with lag 24 in place of lag 1, using context 24. | pending | — |
| `series-gaussian_ar-k0-c1` | Forecast stationary `x[t] = 0.75 x[t−1] + ε[t]`, with independent standard-normal noise, using context 1. | pending | — |
| `series-gaussian_seasonal_ar-k0-c24` | Forecast the same stationary Gaussian process with lag 24 in place of lag 1, using context 24. | pending | — |
| `series-weather-k0-c8` | Forecast weather levels from eight prior rows. | passed | — |
| `series-etth1-k0-c24` | Forecast ETTh1 levels from 24 prior rows. | pending | — |
| `series-m4_hourly-k0-c24` | Forecast M4 Hourly series levels from 24 prior rows. | pending | — |
| `noise-div97-0.01` | Modular-division anchor with 1% of training answers selected for random replacement. | pending | — |
| `noise-div97-0.05` | Modular-division anchor with 5% of training answers selected for random replacement. | pending | — |
| `noise-div97-0.2` | Modular-division anchor with 20% of training answers selected for random replacement. | pending | — |
| `noise-fibonacci-0.01` | First-difference Fibonacci anchor with 1% of training answers selected for random replacement. | pending | — |
| `noise-fibonacci-0.05` | First-difference Fibonacci anchor with 5% of training answers selected for random replacement. | pending | — |
| `noise-fibonacci-0.2` | First-difference Fibonacci anchor with 20% of training answers selected for random replacement. | pending | — |
| `series-weather-shuffled-time` | Weather context-8 control with time order shuffled independently within each split. | pending | — |
| `series-etth1-shuffled-time` | ETTh1 context-24 control with time order shuffled independently within each split. | pending | — |
| `series-m4_hourly-shuffled-time` | M4 Hourly context-24 control with time order shuffled independently within each split. | pending | — |

Noise selection is fixed for each data seed; a replacement can coincidentally
equal the original answer. Held-out answers remain clean. Series task
definitions and defaults are in
[`series_data.py`](src/diff_gpt/application/grokking/series_data.py), and the
historical scorer and train-only codec handling are in the
[series adapter](research/current_grokking/adapters.py).

## Full-GPT mechanism probes: six cases

These are new tasks inspired by earlier component-only witnesses. They train
the whole historical GPT. Four use independently sampled numeric vectors and freely
generate a vector answer. Their primary metric is decoded MSE against the
unquantized teacher; each invalid or missing component receives squared error
64. A target of `negative_mse >= -0.001` means MSE at most 0.001. Their split
sizes are 1,024 train, 256 validation and 256 test examples. Exact generated
sequences, including EOS, are scored separately and supply the later 35-task
benchmark's criterion for the selected numeric probes.

| Case | Task | History | Tier |
| --- | --- | --- | --- |
| `gpt-probe-routing-mixture` | Given eight marked width-16 source vectors, generate their explicit normalized soft-attention mixture. | budget | — |
| `gpt-probe-branch-amplitudes` | Given width-32 sign vectors `x, u`, generate `x + 0.25 u / RMSε(u)`. | passed | short |
| `gpt-probe-conditional-amplitudes` | Generate `x + Σ αᵢuᵢ/RMSε(uᵢ)` for four sign-vector branches; choose the coefficient vector from the first sign of `x`. | passed | — |
| `gpt-probe-linear-transport` | Given a width-32 sign vector `x`, generate `x / 4`. | passed | short |
| `gpt-probe-order-discrimination` | For distinct tokenized integers 0–31, predict whether `a < b` on held-out ordered pairs; target 99% exact accuracy. | budget | — |
| `gpt-probe-key-value-binding` | Recall one independently queried value from two bindings over 32 possible keys/values; 2,048 train and 512 each validation/test, target 99% exact accuracy. | passed | short |

The exact generation grids, teacher definitions, and error penalties are in
[`probes.py`](research/current_grokking/probes.py). For the branch probes,
`RMSε(u) = sqrt(mean(u²) + 2⁻²³)`. The conditional coefficient vector is
`(0.25, 0.5, 0.75, 0.25)` when `x[0]` is negative and
`(0.75, 0.25, 0.5, 0.75)` when it is positive. Numeric probe success does not
by itself reproduce a query-only or gate-only optimization result.

## Image classification and language modeling: two cases

| Case | Task | History | Tier |
| --- | --- | --- | --- |
| `fashion-mnist-gpt` | Classify Fashion-MNIST using BOS plus 196 row-major rounded means of 2×2 image patches. The official training set is split into 50,000 train and 10,000 validation images; the official 10,000-image test set remains separate. Target: 90% validation accuracy. | budget | — |
| `tiny-shakespeare-gpt` | Character-level next-token prediction on fixed context-128 windows of Tiny Shakespeare. Text is split chronologically 80/10/10; 50,000 training windows and 1,024 windows per held-out split. Target: validation CE at most 1.63. | passed | — |

Fashion is a trained GPT classifier, not the earlier fixed-random-feature
double-descent experiment. Shakespeare is an LM quality check, not by itself
a delayed-grokking benchmark. Both data adapters are in
[`probes.py`](research/current_grokking/probes.py).

## Sources and reproducibility

- [Frozen 165-case plan](research/current_grokking/plan.json): exact IDs, task
  parameters, model sizes, budgets, targets and the recorded user-requested skips.
- [Original sweep README](research/current_grokking/README.md) and
  [task inventory](research/grokking_task_inventory.json): provenance,
  historical task sources, controls and limitations.
- [Per-case progress and metrics](research/current_grokking/results/progress_snapshot.csv)
  and [provenance audit](research/current_grokking/results/progress_provenance_audit.json):
  the retained snapshot, including pending, skipped and stopped cases.
- [Current main model](research/main_model/model.json) and its
  [35-task manifest](research/complex_geometry_suite/suite.json): the later,
  selected Rust/Kernite model and its separate benchmark protocol.

The original plan and adapters are historical evidence. New transformer
experiments in this repository must use Rust and Kernite; the historical Python
sources can be read to understand or port the tasks, but must not be executed.

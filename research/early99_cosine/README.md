# Cosine AdamW on the inverse-width shared model

The [declared experiment](plan.json), committed as `2a72520`, changes the
post-warmup rate of the selected native-AdamW model. The peak stays at the
original case LR×2. Keep the original 50-completed-update warmup, including
its zero-rate first update. Then decay by cosine over 200 actual training
epochs to 0.1 of the peak, and hold that floor. Short final batches contribute
their actual examples to the schedule clock.

All model equations, initialization/RNG/capacity, Q/K normalization, SSMax,
fixed RoPE, data, seeds, observations, original update budgets and exact
held-out stopping rule remain inherited. Native AdamW retains betas(.9,.98),
epsilon1e-8 and original weight decay, with the existing compiled CUDA loss
and fused capturable updates. The prior 40/50 full-model numerical result
and its ten failures remain unresolved.

The new [execution journal](../early99_execution/journal.py) durably records a
pending batch before each genuine update, synchronizes its completion on CUDA,
then records completion. It uses atomic replacement and file/directory fsync.
Preparation issues the original capture-neutral warmup without journal writes.
After an interruption, earlier journals are archived and all replayed examples
are counted. Assuming filesystem fsync durability, uncertainty is at most the
one pending batch per interrupted segment. This adds I/O/synchronization rather
than changing the loss, batch, optimizer update or RNG. Schedule position follows
the restored trajectory; the learning metric includes replayed work.

All **13 implementation checks passed**, including two real CUDA cases. They
check actual rates with tail batches; constructor/preparation neutrality;
native parameters/moments/RNG through warmup, decay and floor; nonempty-state
CUDA restoration; complete trainer CPU resumption; exact trainer AST outside
journal additions; pending/complete ordering; repeated replay intervals; archived
record tampering; and zero-update checkpoint finalization. Initial constructor
and audit-fixture failures are preserved alongside corrected results in
[validation.json](results/validation.json).

[Four checked real statements](Guarantees.lean) establish the cosine multiplier
and nonnegative rate bounds, the peak endpoint and the floor endpoint, with
actual floor1/10 examples. They compile without warnings or extra axioms. These
are scalar schedule properties; they do not establish faster grokking or a
transformer convergence theorem. All previous inverse-width model bounds still
apply pointwise.

Complete the fixed six-task, three-seed screen. Promotion requires 18/18 passes
and three improved median paired epoch ratios versus the unchanged parent.
Keep all five controls and every miss/error. No screen geometric mean is used.
An advancing candidate must complete all75short cells; only a new complete-suite
winner receives one reused long30diagnostic. Interruption intervals must support
the same screen decision and any claimed full-suite improvement.

The older cosine experiment and both completed Schedule-Free negatives remain
controls. Muon/AdamW and all-matrix Muon are the queued optimizer follow-ups.

The completed screen passes **18/18 cells**, but improves **0/6 task medians**
against the constant-rate inverse-width parent, so it does not advance. Median
paired epoch ratios are 1.0 for Boolean AND, division193 and division97; 1.2857
for key/value binding; 1.5294 for square-sum; and 1.25 for linear transport.
The largest observed slowdown is square-sum seed0: 129 to 207 epochs (1.6047×).
There is no screen geometric mean, full75 run or new long diagnostic.

The [complete audit](results/complete-audit.json) reconstructs all five controls'
18-cell JSON/CSV reports and checks every checkpoint, history, initialization,
native optimizer state, journal and 58 frozen source files. All five additional
reporting checks pass. An interruption left the next cell's directory empty
before any update; all15 completed cells were preserved byte-for-byte, and the
remaining three completed from their declared initialization with no replay.
The constant-rate parent remains selected. This result tests the declared
200-epoch, 0.1-floor schedule; it does not rule out all possible schedules.

Quartic complex-linear MD optimizers and paper schedule ablations

Question: on (a+b)^4 mod97, how do gradual decay, a short WSD cooldown and warmup
affect AdamW, AdamMD, AdamMDW and complex MuonMD? This is a diagnostic on a
previously examined task. The frozen protocol and every deviation are in PLAN.txt.

Core design: each optimizer x linear/WSD1-sqrt x warmup0/1000 x seeds0,1,2.
This gives48 core cells. Six original AdamW/MuonMD linear/no-warmup runs are
reused unchanged from31d0a10. The schedule figure also shows cube-root and
fourth-root WSD cooldowns; include both for MuonMD without warmup, three seeds
each. Final combined total54 cells, of which48 are newly scheduled.

Optimizer   Direction peak   Gain peak   Tied embedding peak   Weight decay
AdamW       0.002            none        0.002                 0.1
AdamMD      0.002            0.002       0.002                 0
AdamMDW     0.002            0.002       0.002                 0.1
MuonMD      0.01             0.01        0.002                 0

All runs use100000 updates and the same width128, four-head, one-block
complex-linear model with real XSA and radial complex-product FFN. Data,
initial parameters and batch order are paired by seed. Training continues
after99%. Observe every19 updates and save every500 epochs plus initialization,
first99, completion and explicit clean stop. No adaptive retries or LR tuning.

What the paper says

Local paper2606.25971v2, §4.1 Learning-Rate Schedules on the Sphere, explains
that fixed direction norms remove the implicit reduction in relative update
caused by norm growth. Full linear annealing outperforms its WSD comparison
in the shown sweep. The optimal schedule remains open. The adjacent warmup
ablation reports a benefit from dropping warmup for MD. Appendix B.1 specifies
linear decay to1e-8, WSD with a20%1-sqrt cooldown, and1000 warmup steps for the
AdamW baseline. The schedule-sweep figure additionally displays1-u^(1/3) and
1-u^(1/4) alternatives. See SOURCE_REFERENCES.txt for pinned local sources.

This experiment uses the paper's1000-step warmup count with a100000-update
budget; its dense runs used50000 updates, so the warmup fractions differ.
Warmup starts at0 and reaches peak at update1000. Linear decay then spans
the remaining run. WSD holds peak until update80000 and cools to1e-8 at99999.
Every group has its own absolute floor. Zero LR advances adaptive histories
without moving the physical weights or gains.

The paper uses equal-budget peak-LR sweeps; these runs hold the peaks fixed
to isolate schedule effects. They cannot identify the best-tuned optimizer
or a universally optimal schedule. AdamMDW is a local Lean-defined extension,
not an optimizer proposed under that name in the paper.

AdamMD and AdamMDW

AdamMD uses ordinary bias-corrected Adam on the complex direction's independent
real and imaginary coordinates, and on both raw softplus gain vectors. All
gradients are computed at the old factors. The direction candidate is projected
to its initial complex Frobenius radius before fusing with the new gains.
AdamMDW then subtracts eta_direction*decay*OLD physical weight. It repairs the
row-gain storage to restore the direction radius while preserving the corrected
physical matrix, column gains and all moment histories. This is the complex
extension of Transformer.MDW.adamMDWProposal and rebalanceStorage. A zero result
keeps finite raw gains. These real Lean definitions do not prove complex
transformer convergence. Coordinate Adam is not claimed to be phase equivariant.

Native validation and execution

Rust/Kernite only. Model and Muon kernels import immutable sources from the
earlier cohorts. Native tests cover FP64 complex oracle updates at schedule
boundaries, Hermitian products, gain finite differences, physical MDW decay,
zero correction, stable inverse-softplus, exact zero-rate behavior, full state
continuation, all100000 linear schedule values, and bit-identical trajectories
against the frozen original optimizer. CPU audits reject changed baselines,
mispaired cells, corrupt states and inconsistent metrics.

  cargo fmt --manifest-path research/quartic_md_schedules/native/Cargo.toml -- --check
  cargo clippy --offline --locked --all-targets --manifest-path research/quartic_md_schedules/native/Cargo.toml -- -D warnings
  cargo test --release --offline --locked --manifest-path research/quartic_md_schedules/native/Cargo.toml -- --test-threads=1

Binary commands: archive ROOT; smoke ROOT; suite ROOT; cell ROOT ARM SEED;
resume ROOT ARM SEED; audit ROOT; combined ROOT BASELINE_ROOT. A19-update
smoke test per new arm checks wiring and finite state, not generalization.
STOP in a cohort or cell directory checkpoints at the next observation.
Unclean windows are retained as failures and never replayed automatically.

run-after-quartic.sh waits for the existing six-cell queue's PID/start time,
requires its completed audit, checks frozen artifact hashes, and reruns the
native tests before training. Only one training cohort owns the GPU at a time.
Combined reporting preserves all outcomes and each cohort's source identity.

Results: stopped at the user's request on 2026-11-07 at 22:11:31 UTC.

Of 54 declared cells, 46 completed 100000 updates: 40 new cells and six reused
baselines. One cell was cleanly stopped and seven were never started. The
stopped AdamW/WSD1-sqrt/warmup1000 seed 2 cell saved all state at 19057 updates
(epoch 1003); its 97.747078% endpoint is a partial-run observation and is excluded
from full-budget averages. The STOP flag remains in the runtime directory.

Best completed endpoint: AdamMD/WSD1-sqrt/warmup1000 seed 1, 99.383634% held-out
accuracy (4676/4705), with a 99.659936% peak at epoch 4669 and 100% final training
accuracy. The same configuration ended at 96.620616% on seed 0; seed 2 was not
started. AdamW/linear/warmup1000 seed 2 also finished above 99%, at 99.086079%.
These are the only two completed endpoints above 99%.

Ten configurations have all three seeds complete. The largest three-seed mean
final accuracy among those is 98.583068%, from AdamW/linear/warmup1000. Matched
linear/warmup1000 means are 97.810840% for AdamMD and 98.391782% for AdamMDW.
With WSD1-sqrt and no warmup, the means are 98.030464% for AdamW, 98.384697% for
AdamMD, 98.370528% for AdamMDW and 96.840241% for MuonMD. The incomplete factorial
and fixed peak rates do not establish an overall or best-tuned optimizer winner.

results/ contains every declared cell in the combined native audit, per-cell
summary.tsv, complete-seed means in arm-summary.tsv, the explicit stop record,
checkpoint/protocol manifests, frozen artifact identities and validation logs.
Native CPU audits were rerun after stopping: zero integrity failures, including
the saved partial cell. Seven audit entries retain the frozen auditor's label
pending_or_running; their absent directories were checked and stop.json records
them as not started. No training was restarted for reporting.

Full histories, weights and optimizer states remain under
/home/misha/python_projects/gpt/runs/quartic-md-schedules and the reused
/home/misha/python_projects/gpt/runs/quartic-muonmd. See results/README.txt for
verification commands and the limits of this stopped snapshot.

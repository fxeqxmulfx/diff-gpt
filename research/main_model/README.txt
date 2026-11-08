Main research model: Complex Peri-RMS + squared-imaginary score + phase-mHC + LeRoPE
Selected on2026-11-08 under the user's standing epoch/coverage promotion rule.
Machine-readable definition: model.json; routing details: mhc.json.
Use this architecture as the starting point for future shared-model research.

Squared-imaginary attention is promoted after75/75short and30/30long independently
audited passes,zero audit failures and no lost phase-mHC parent passes. Short
epochs decrease1.0718% (6earlier/67tied/2later); long epochs increase0.9485%
(10earlier/3tied/17later),using geometric means of every paired ratio. Promotion
uses fewer short epochs and retained long coverage,the predeclared rule.
All19regressions remain in results/phase-score-suite-01/epoch-regressions.json.
All105paired epochs and30long rows/plots: results/phase-score-suite-01/report-final/.
Decision and prior manifests: results/phase-score-suite-01/promotion/.
Quartic seeds0/1/2 regress901/687/822 to1248/922/945epochs. There is no uniform
improvement,statistical significance,sustained99 or convergence claim.
Do not restart this completed cohort. The final covariance candidate retained
105/105 but used4.009%more short epochs. The1DYaRN inference study lost the
basis-depth-hard seed0 pass at64-to128context (100%control,95.3125%candidate);
60passes were retained and44later cells left unrun. Axial2DLeRoPE+YaRN retained
105/105 but used2.608%more short and8.998%more long epochs. None replaces main.
All24axial epoch regressions are retained in results/axial-suite-01/.
The additional36-run DiffGPT comparison with MASH,LGT and MASH-half is complete
and independently audited. Both1Dand2Dmissed90%in every5000-update cell; no
first90epoch ratio exists. Best2Dendpoint:89.328%MASH-half,F1,seed17;F2and
longer-context results are poor. Allindividual pairs and native/YaRN context
results remain in results/diffgpt-axial-suite-01/report-final/. These new failures
are separate from the fixed105passes and do not imply universal task coverage.
The final experiment queue and native integration are complete. The installed
src/native_transformer passed 18 CPU tests, formatting, strict Clippy, rustdoc,
exact 100-update trained-checkpoint replay, and both CUDA sanitizers with zero
errors. See src/native_transformer/VALIDATION.txt for evidence and limitations.
The remaining handoff action is the requested commit, then stop; no new research.

The native DiffGPT interface uses2Donly,at the user's explicit request:
axial time x feature LeRoPE,with inference YaRN beyond actual training context.
It rejects1Dcheckpoints and never changes their positional interpretation.
Keep1Dfor the generic main and saved research comparisons. This interface
choice is separate from the generic-main promotion criterion above.

The phase-mHC parent passed105/105 with1.8747% fewer short epochs and11.5692%
fewer long epochs versus its LeRoPE parent; all14regressions remain in
results/phase-mhc-suite-01/epoch-regressions.json. It improved quartic seeds
1724/2416/1307 to901/687/822epochs. The historical evidence remains closed.

The parent LeRoPE main passed105/105 with0.5339% fewer short epochs and1.5099%
more long epochs versus its standard-RoPE parent; all17regressions remain in
results/lerope-suite-01/report-final/. Log-Scaled LeRoPE passed105/105 but used
6.2493% more short and14.1047% more long epochs;32regressions are retained.
Complex-pair AdamW passed105/105 but used2.4349% more short and11.7655% more
long epochs;28regressions are retained. Both were rejected. Their closed reports
are results/log-lerope-suite-01/FINAL_RESULTS.txt and
results/pair-adamw-suite-01/FINAL_RESULTS.txt. Ordinary AdamW remains selected.

The phase routing implementation is frozen in results/phase-mhc-validation-final/;
registration: results/phase-mhc-suite-01/registration.sha256.
Its corrected gate-adjoint arithmetic passes1440normalized-state cases and
preserves exact zero-phase main logits/gradients. All540CPU finite differences,
96optimizer cases,continuation,maximum shapes and both sanitizers pass.
The189extreme primitive failures,203strict mixed-precision AD misses and147
near-zero FP32 router misses remain recorded. Promotion does not erase them.
The multiplicative gate was stopped by the user after
75short passes and13.1626% more epochs (5earlier/20tied/50later); its long
outcomes are4passes,one stopped and25unrun. It does not replace main.
The two previous gate candidates were stopped by the user after7.3898% and
8.1284% more short epochs; neither replaces main. See STATUS.txt,queue.json
and ROTARY_RETEST_PLAN.txt. LeRoPE passes360implementation checks and five
GPU tests; five strict mixed-precision gradient misses remain recorded,while
the FP32 diagnostic passes160/160. See results/rotary-validation-final/REPORT.txt.
Both original rotary experiments and the new fixed-suite YaRN studies are complete.

The score implementation is frozen in results/phase-score-validation-final/;
registration: results/phase-score-suite-01/registration.sha256. All4800primitive
cases,990CPU finite differences,112optimizer cases,continuation,maximum shapes
and both sanitizers pass. Preserve147strict mixed AD misses including6score
directions; the fixed FP32 diagnostic passes26/26score directions and retains
24near-zero router misses. This does not erase any inherited numerical misses.

The current.sh entrypoint selects results/phase-score-validation-final/candidate
from phase_score_native, benchmark arm "phase-score-lerope". Its shared frequency parameters
start at zero and use a separate clip1,weight-decay0 optimizer group.
Forward affine GEMMs use FP16 operands, backward GEMMs use TF32, and accumulation,
remaining model math, master weights and moments use FP32. There is no runtime
precision fallback. The previous standard-RoPE main passed75short/30long, with
1.7953% fewer short epochs and11.5413% fewer long epochs than the prior mHC main
(geometric means over every pair). No main pass was lost. Decision and previous
manifests: results/gpu-suite-01/promotion/. All task/seed epochs and long plots:
results/gpu-suite-01/report-final/. Runtime evidence: results/gpu-opt-05/REPORT.txt.

That prior improvement was not uniform:5short and9long cells used more epochs. Quartic
seed1 takes2133versus1480 epochs; Fibonacci recurrence seed0 takes1499versus991.
Retain386strict scalar-AD misses and15337small differences in the19-update
FP64 optimizer comparison. Full-suite empirical success does not turn those
failed strict checks into passes or establish a floating-point proof.

Peri-RMS + causal SSMax with squared-imaginary scores and ordinary values + plain complex-product FFN +
Q/K RMS normalization + residual scaling after RMSNorm + tied embedding/readout
weights + parameter-free RMSNorm, including embedding RMSNorm + LeRoPE
+ four-stream phase-conjugated mHC residual routing.

Architecture
Let D be the real model width,L the number of layers,H the number of heads,
d=D/H the real head width,and N_i=i+1 the number of visible causal positions.
R_n(x)=x/sqrt(mean(x^2)+epsilon),epsilon=2^-23. Every R is parameter-free.
The learned embedding table E is shared with the output classifier.

  h_i,s^0 = R_D(E[token_i]) for each of four streams s
  z_i = R_D(sum_s read_i,s * h_i,s)
  q_i   = LeRoPE_i(R_d(W_Q z_i))  per head
  k_i   = LeRoPE_i(R_d(W_K z_i))  per head
  v_i   = W_V z_i               per head
  Z_ij  = sum_c conj(q_i,c)*k_j,c = Re(Z_ij)+i*Im(Z_ij)
  p_ij  = softmax_j((log(N_i)/d)*(Re(Z_ij)+beta_l,h*Im(Z_ij)^2/d)), j<=i
  a_i   = concatenate_heads(sum_{j<=i} p_ij*v_j)
  h'_i,s = sum_t P_i,s,t*r_s*conj(r_t)*h_i,t + write_i,s*alpha*R_D(W_O a_i)
  z'_i = R_D(sum_s read'_i,s * h'_i,s)
  g_i = W_g z'_i; u_i = W_u z'_i
  f_i = R_D(W_F((g_i complex-multiply u_i)/sqrt(2)))
  h''_i,s = sum_t P'_i,s,t*r'_s*conj(r'_t)*h'_i,t + write'_i,s*alpha*f_i
  logits_i = E * R_D(mean_s(h_i,s^L))

Each attention/FFN sublayer has its own dynamic router: RMS of the flattened
four-stream state, real projection to24logits, three learned scales and24biases.
Read gates are sigmoid, write gates are2*sigmoid. The4x4 skip map uses20
column-then-row Sinkhorn cycles: first cycle in log space with two FP32 words
for offset corrections, then19 in FP32 positive space. Router RMS uses fixed
rescaling to handle its validated large finite input cases. These numerical
choices are static; mathematical proof statements retain their declared scope.

For each sublayer,r_0=1 and r_s=(1+i*b_s)/sqrt(1+b_s^2) for s=1,2,3.
The6Lreal b coordinates start at zero without additional random draws and are
shared across tokens,heads and complex channels within each sublayer. The skip
map is D P D*,where D=diag(r),with unchanged real read/write gates. Exact-real
unitary conjugation preserves the singular values of frozen P and its energy
bounds; ordinary constant streams are replaced by a phase-aligned conserved
mode when P has unit row sums. This does not bound the full input-dependent
Jacobian or prove finite Sinkhorn exactly doubly stochastic. FP32 coefficients
are approximate. See proofs/ComplexPhaseRouting.lean and the numerical report.

alpha=1/sqrt(2L) is applied AFTER each branch output RMSNorm. Q/K RMS uses
the head width; embeddings,branch inputs/outputs and final readout input use D.
LeRoPE rotates Q/K using angle(i,m)=i*10000^(-2m/d)*exp(alpha_m).
The d/2 learned alpha values are shared across Q/K, heads and layers, with
zero initialization and no logarithmic position factor in the selected main.
The implementation recomputes tables every forward, using fixed compensated
FP32 exponential/phase reduction. SSMax uses log(visible prefix length),
with inverse real head width1/d; it is not the usual1/sqrt(d) temperature.
Ordinary attention retains first-token and repeated-token value contributions.
There is no XSA exclusion and no subtraction of the uniform attention mean.

beta_l,h=a_l,h/sqrt(1+a_l,h^2),one zero-initialized real coordinate per layer/head.
The L*H score coordinates follow the6Lresidual phases and precede frequencies.
At zero score,the original dot-product logits and all existing gradients match
the parent exactly; the new coordinate gradients need not vanish. The imaginary
square is a nonlinear relative-phase feature,not merely a renamed Hermitian dot.
With exact-real Q/K norm bounds,the score magnitude is at most5/4*ln(N),looser
than the parent's ln(N). Common-phase rotation and simultaneous conjugation
preserve the score. These statements do not prove floating-point stability or
convergence. See proofs/ComplexPhaseScore.lean and COMPLEX_PHASE_SCORE_PLAN.txt.

The FFN uses plain complex multiplication,with no radial sigmoid or SwiGLU.
Projections are bias-free complex-linear maps represented in real arithmetic
as[[A,-B],[B,A]]. Real/imaginary coordinates are grouped per head for attention
and in halves for the FFN. The even real FFN width is the implementation's
round-up-to-even(floor((8D+1)/3));D=128 gives342 real/171 complex hidden units.
Embedding lookup is normalized,while the tied readout uses the raw learned E.
All projection/embedding weights learn; the RMS normalizers have no parameters.

Native implementation
research/main_model/phase_score_native/ supplies the selected Rust/Kernite runtime.
The immutable benchmark label "main" denotes the former ordinary-residual
control. The labels "mhc" and "lerope" denote prior main models;
"phase-mhc-lerope" denotes its phase-routing parent; "phase-score-lerope" selects
main. Routing is defined in mhc.json and COMPLEX_PHASE_ROUTING_PLAN.txt;
the score is defined in COMPLEX_PHASE_SCORE_PLAN.txt. Historical labels stay fixed.
All use corrected loss and attention adjoints, ordinary attention and the
plain complex product. current.sh identifies and dispatches the selected arm.
The ordinary attention, product, data and metric modules are reused from the
frozen geometry suite. phase_score_native/src/layout.rs preserves complex ties.

The earlier completed architecture benchmark uses wsd_suite_native/ and the immutable binary
runs/main-model-mhc-wsd-01/main-transformer-wsd. Both arms pass75/75 short and
30/30 long, independently audited. The full report is
results/mhc-wsd/FINAL_RESULTS.txt; long-epochs.tsv includes all30 paired epochs,
updates, budgets and observation intervals. Do not restart this completed cohort.
The older native/ optimizer and stopped main-model-mhc-02 cohort remain separate
historical evidence. See STATUS.txt and queue.json for the next experiment.

The historical archive arm name "complex-hermitian" identifies the shared
complex parameter layout. The explicit attention tag and engine switches
select ordinary attention; the archive name alone must not select XSA.
model.json is an architecture/provenance manifest,not a new generic CLI config.
The original diagnostic/no-xsa-* commands remain the frozen quartic reference.

Reference state and training
The historical ordinary-residual reference is
runs/complex-pow4-no-xsa-seed1-01/checkpoint-050312.bin,
which stores parameters,AdamW first moments and second moments. The archived
ordinary manifest under results/mhc-wsd/ pins its provenance. Dimensions:D128,H4,L1,
vocabulary97,hidden342;209280 expanded and110848 independent real parameters.
Initialization draws Gaussian(0,.02) entries with native Random,then enforces
the complex matrix ties. The seed1 reference uses the exact archived initial
state and sample order. Optimizer settings are recorded as a reference recipe,
not as a task-independent optimal learning rate: AdamW LR.002,betas(.9,.98),
epsilon1e-8,weight decay.1,50-update warmup and constant LR afterward.
The selected score+phase-mHC LeRoPE reference is
runs/main-model-phase-score-01/phase-score-lerope/arithmetic-sum-pow4/seed1/checkpoint.bin:
first99 at922epochs/17518updates,233936expanded/135504independent parameters.
mHC adds24630real router parameters atD128,L1; phase routing adds6;
the score adds4 and LeRoPE adds16 atH4. model.json pins this reference.
results/phase-score-suite-01/promotion/model-before.json preserves the phase-mHC
parent and its687-epoch reference.
results/phase-mhc-suite-01/promotion/model-before.json preserves the LeRoPE
parent and its2416-epoch reference; results/lerope-suite-01/promotion/model-before.json
preserves the standard-RoPE main and its2133-epoch reference. The prior mHC
manifest and1480-epoch reference remain in results/gpu-suite-01/promotion/.
The older ordinary manifest remains
under results/mhc-wsd/. Keep buffered history and500-epoch/first99 saves.

The user-selected recipe for subsequent research is peak LR0.001, AdamW
beta(.9,.95), epsilon1e-8, weight decay.1, independent-coordinate gradient
clip1, 2% warmup and20% cosine decay to1e-8 over each original fixed budget.
LeRoPE alpha uses a separate L2 clip1,weight-decay0 group with the same
learning rate,betas,epsilon and schedule. Base clipping excludes alpha.
The residual-phase and score coordinates use the ordinary base clip1/wd.1
group,each with independent real-coordinate AdamW moments.
The new real phase coordinates join base clipping and weight decay with
ordinary independent-coordinate second moments; no complex-pair AdamW is used.
The completed fresh comparison uses this recipe for both arms. The older
reference above retains its original optimizer and provenance.

Evidence and limits
The matched(a+b)^4 mod97,seed1 run reaches4658/4705=99.001063% at epoch2648,
update50312. Training4704/4704;held-out diagonal32/42,non-diagonal4626/4663.
The exact ordinary candidate passes42 full-model cases,4 XSA regressions,
1652 parameter directions and23232 primitive gradient coordinates; all7 saved
states replay. Later shared-engine changes also replay the reference and pass
ordinary regression checks during the centering study.

Pure causal centering is a rejected alternative: peak4.1870%,final2.9968%
after100000updates. Lean proves that it removes constant values and their
shared-parameter attention gradients,while the residual survives. The result
does not discredit RMSNorm,residual scaling,tied weights,or complex products.

The earlier main architecture passed75/75short and30/30long under the fresh
LR0.001 WSD recipe. The earlier mHC also passed105/105, with10.996% fewer short epochs and
19.680% fewer long epochs than this matched control (full-pair geometric means).
Both fresh recipes remain slower on short tasks than the historical main:
main2.659x and mHC2.366x. The original frozen historical gate retained the old
incumbent; its output remains unchanged. The user subsequently explicitly
promoted mHC based on matched LR0.001 epochs and105/105 coverage. This decision
is recorded separately in results/mhc-wsd/user-promotion.json. Do not transfer
the new coverage claim to the old
recipe, whose quartic seed0 budget miss remains recorded. First99 stopping
does not establish sustained99% or universal convergence.

The subsequent numerical candidate in runs/main-model-gpu-suite-01 independently
passes all105 cells with the same initialization, data, sampler, recipe, budgets
and observation cadence. The75short pairs use0.9820473743x epochs and30long pairs
use0.8845865559x epochs versus the prior mHC. Short earlier/tied/later4/66/5;
long18/3/9. Four fresh hard cells reproduce every observed metric and all final
weight/moment bits of the earlier FP16/TF32 candidate, excluding timing fields.
The bundle changes several precision/implementation choices; it does not isolate
which numerical choice causes each epoch difference. Selection is based on the
complete epoch/coverage gate, not the measured3.46x/6.06x update speedups.

Evidence: ../complex_geometry_suite/results/pow4-no-xsa.txt,
no-xsa-stress.txt,pow4-centering.txt and pow4-centering/CenteringCheck.lean.
The broader task remains ../early99/ASTRA_TASK.txt. Use native tooling only.

Research continuation
The active goal is to reduce epochs while preserving every measured main-model
task/seed pass. The selected FP16/TF32 mHC is the parent/control for subsequent LeRoPE, Log-Scaled
LeRoPE and gated-attention experiments at LR0.001. The user requested profiling
the current mHC PTX first; runtime does not determine model selection.
RESEARCH_RULES.txt records
the user's strict rejection rule. The independent report applies coverage_gate.jq:
one lost pass rejects a candidate; pending cells and execution errors cannot count
as successes. A complete coverage pass is necessary, but an epoch improvement is
also required. Full-suite geometric means are reported only when all declared
pairs pass; any solved-pairs-only mean is explicitly provisional or censored.

mHC validation
The existing Lean PeriRMS mHC proofs build and use only standard axioms. They
bound each stream's RMS by 1+2*sqrt(2L), using finite row normalization and
bounded write gates, and bound final RMS features by one. They do not prove
optimizer convergence or the full nonlinear gradient bound. See proofs/README.txt.
The native candidate passes 20 full-model cases, 1308 parameter directions and
29 Sinkhorn/RMS stress cases, plus checkpoint resume and maximum-shape checks.
Read mixtures and their adjoints, coefficient adjoints and Sinkhorn reductions
use FP64 to preserve near-zero routing derivatives. Failed engineering attempts
and the explicitly stopped first cohort remain archived separately.
The local MHCNormalization.lean certificate additionally proves that after the
first Sinkhorn cycle, subsequent column/row denominators lie in [1/4,4]. This
supports the equivalent positive-space kernel optimization; FP32 error remains
covered by separate numerical tests. Four new statements use standard axioms.

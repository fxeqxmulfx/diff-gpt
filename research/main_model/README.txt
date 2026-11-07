Main research model: Complex Peri-RMS, version1
Selected by the user on2026-11-08. Machine-readable definition: model.json.
Use this architecture as the starting point for future shared-model research.

Peri-RMS + ordinary causal SSMax attention + plain complex-product FFN +
Q/K RMS normalization + residual scaling after RMSNorm + tied embedding/readout
weights + parameter-free RMSNorm, including embedding RMSNorm + standard RoPE.

Architecture
Let D be the real model width,L the number of layers,H the number of heads,
d=D/H the real head width,and N_i=i+1 the number of visible causal positions.
R_n(x)=x/sqrt(mean(x^2)+epsilon),epsilon=2^-23. Every R is parameter-free.
The learned embedding table E is shared with the output classifier.

  x_i^0 = R_D(E[token_i])
  z_i   = R_D(x_i)
  q_i   = RoPE_i(R_d(W_Q z_i))  per head
  k_i   = RoPE_i(R_d(W_K z_i))  per head
  v_i   = W_V z_i               per head
  p_ij  = softmax_j((log(N_i)/d) * dot(q_i,k_j)), j<=i
  a_i   = concatenate_heads(sum_{j<=i} p_ij*v_j)
  y_i   = x_i + alpha*R_D(W_O a_i)
  g_i   = W_g R_D(y_i)
  u_i   = W_u R_D(y_i)
  x_i'  = y_i + alpha*R_D(W_F((g_i complex-multiply u_i)/sqrt(2)))
  logits_i = E * R_D(x_i^L)

alpha=1/sqrt(2L) is applied AFTER each branch output RMSNorm. Q/K RMS uses
the head width; embeddings,branch inputs/outputs and final readout input use D.
RoPE has base10000 and rotates Q/K only. SSMax uses log(visible prefix length),
with inverse real head width1/d; it is not the usual1/sqrt(d) temperature.
Ordinary attention retains first-token and repeated-token value contributions.
There is no XSA exclusion and no subtraction of the uniform attention mean.

The FFN uses plain complex multiplication,with no radial sigmoid or SwiGLU.
Projections are bias-free complex-linear maps represented in real arithmetic
as[[A,-B],[B,A]]. Real/imaginary coordinates are grouped per head for attention
and in halves for the FFN. The even real FFN width is the implementation's
round-up-to-even(floor((8D+1)/3));D=128 gives342 real/171 complex hidden units.
Embedding lookup is normalized,while the tied readout uses the raw learned E.
All projection/embedding weights learn; the RMS normalizers have no parameters.

Native implementation
research/complex_geometry_suite/diagnostic/ supplies the Rust/Kernite engine.
The model is selected by attention="ordinary_causal" and
ffn="plain_complex_product",with normalized=true and ssmax=true.
Initialize Engine using the complex-linear weight layout,then enable stable
loss,stable attention,plain product and ordinary attention. These are the
existing enable_stable_loss,enable_stable_attention,enable_plain_product and
enable_ordinary_attention methods. ordinary_attention.rs and plain_product.rs
are the authoritative branch kernels; native/layout.rs defines parameter ties.

The historical archive arm name "complex-hermitian" identifies the shared
complex parameter layout. The explicit attention tag and engine switches
select ordinary attention; the archive name alone must not select XSA.
model.json is an architecture/provenance manifest,not a new generic CLI config.
The existing no-xsa-* commands reproduce the matched quartic study; its driver
is specific to that study and is not a general75/30-suite runner.

Reference state and training
The saved reference is runs/complex-pow4-no-xsa-seed1-01/checkpoint-050312.bin,
which stores parameters,AdamW first moments and second moments. model.json pins
its hash and the validated source/kernel fingerprints. Dimensions:D128,H4,L1,
vocabulary97,hidden342;209280 expanded and110848 independent real parameters.
Initialization draws Gaussian(0,.02) entries with native Random,then enforces
the complex matrix ties. The seed1 reference uses the exact archived initial
state and sample order. Optimizer settings are recorded as a reference recipe,
not as a task-independent optimal learning rate: AdamW LR.002,betas(.9,.98),
epsilon1e-8,weight decay.1,50-update warmup and constant LR afterward.
Keep history buffered and checkpoint every500epochs or first99/completion.

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

This is the user's selected MAIN RESEARCH MODEL,not a new claim of75/75short
or30/30long coverage. Those full frozen-suite checks for this exact combination
remain outstanding. Do not combine passes from different architectures into
a synthetic coverage result. The reference stops atfirst99 and still has
large earlier accuracy drops; sustained99% and universal convergence are not
established. Existing benchmark controls and their promotion rules remain.

Evidence: ../complex_geometry_suite/results/pow4-no-xsa.txt,
no-xsa-stress.txt,pow4-centering.txt and pow4-centering/CenteringCheck.lean.
The broader task remains ../early99/ASTRA_TASK.txt. Use native tooling only.

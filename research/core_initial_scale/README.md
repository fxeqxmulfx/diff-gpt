# Initial residual scales

This diagnostic measures the original model, embedding RMS alone, scaled Peri-RMS
alone, and their existing combination on the six fixed screen tasks and seeds
0/1/2. It is declared after the embedding-only screen and while the Peri screen
is running. It is descriptive and does not change candidate settings or selection.

Use the first 32 stored training examples, with the actual task adapter and
original initialized parameters. Measure embedding RMS and the first block's
attention and feed-forward residual contributions, including the actual guarded
prefix sharing. Include every token position, including zero attention outputs.
Compare each branch's RMS with the state to which it is added. No optimizer,
held-out scoring, trained checkpoint or new learning run is involved.

The [plan](plan.json) fixes the cohort, measurements and source hashes before
execution. The hook observations must reconstruct both actual residual additions
exactly; instrumented and uninstrumented logits must match exactly. All four
models must retain equal original parameter tensors, ordering, buffers and RNG,
and tied embedding/readout identity. These checks validate the diagnostic path;
the separate architecture study already validates the model equations.

Initial branch magnitude is not a convergence guarantee or an explanation of
long-suite failures. Forward CPU FP32 observations, reduced in FP64, are numerical
measurements rather than real-arithmetic or GPU trajectory guarantees.

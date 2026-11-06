# Initialization audit across depth, width and attention gain

The user suggested different initialization rules for different model parts.
This [plan](plan.json) first measures the current initializer without changing it.
The final matrices currently use `Normal(0, 0.02)`; tied embedding/readout weights
receive one shared final draw, and SwiGLU replacements receive their own draws.

Measure four shapes, three seeds, and three architectures: original, full
normalized inverse-width SSMax, and full normalized usual-gain SSMax. All 36
states use the same first 64 training examples and fixed unit logit probe.
No optimizer updates or held-out scores are computed. The already validated
observer checks exact forward/gradient neutrality and unchanged model/RNG for
every state, and records every component plus residual and tied-weight paths.

For an independent Gaussian matrix acting on a fixed unit-RMS input, the
per-output variance is `fan_in * sigma²`. With sigma 0.02, fan-in 128 gives
0.0512, while 256 gives 0.1024. This identity helps interpret the measurements;
it does not assume independence between tied readout weights and hidden states
that already depend on those weights. Post-normalization outputs can remain
bounded while their local gradient gains change.

Initialization changes will be separate experiments, declared after this audit
and the current capacity/gain evidence. A local scale difference does not show
that an initializer is wrong or establish the best alternative. The running
capacity grid and queued usual-gain experiment remain unchanged.

Status: measurement plan and reused-observer driver prepared; commit before
measuring. Validation is inherited from the frozen observer and additionally
checked on every actual observed model, including the two-layer shapes.

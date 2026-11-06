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

The audit completes all **36 states and 72 objective measurements**. All
forward/parameter-gradient checks are exact; all **288 residual/tied path
decompositions** reconstruct with zero error. Active values and gradients are
finite. The [full measurements](results/measurements.json),
[6,108 component rows](results/components.csv), and
[1,044 parameter rows](results/parameters.csv) retain every observation.

For the one-layer inverse-width model, three-seed medians show how widening
changes the scales under the existing initialization recipe:

| Measurement | Width 128 | Width 256 |
| --- | ---: | ---: |
| Q RMS before Q/K normalization | 0.22775 | 0.31824 |
| SwiGLU product RMS | 0.02589 | 0.05282 |
| FFN output RMS before branch normalization | 0.00966 | 0.02760 |
| Attention output RMS local probe-gradient gain | 40.37 | 19.78 |
| FFN output RMS local probe-gradient gain | 105.06 | 36.52 |
| Final logit RMS | 0.29783 | 0.50702 |

The weight standard deviations remain near the declared 0.02 (observed range
0.01968–0.02027 across all recorded matrix pieces). The change in activation
scales is consistent with the changed fan-in and nonlinear composition.
Large local normalization gains do not by themselves imply a harmful global
gradient or establish a defective initializer.

The [paired gain comparisons](results/paired_gain_gradients.csv) add a more
direct observation: restoring only the usual attention gain multiplies Q/K
unit-probe parameter-gradient RMS by medians **4.80×/4.81×**, while the V ratio
is **1.005×**. These are 18 matched size/seed/layer pairs; all normalizations,
parameters and tying are retained. This supports completing the gain experiment
before interpreting a different initializer as the remedy. It remains an
initial local measurement, with no training or generalization conclusion.

The [summary](results/summary.json) retains every seed and layer. No new
initializer is selected. A future block-specific initialization comparison
must declare its intervention and preserve the current experiments unchanged.

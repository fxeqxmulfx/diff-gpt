# Complete inverse-width model numerical diagnostic

The [declared plan](plan.json), committed as `f10bdd3`, extends the selected
model's 120 isolated attention checks to complete transformer forwards and
parameter gradients. It uses the actual inverse-width attention, scaled Peri-RMS,
embedding RMS, centered XSA, SwiGLU, Q/K RMS, tied weights and original fixed RoPE.
It does not apply the separate numerical repairs or modify the active benchmark.

The original five depth/context shapes and ten embedding/QKV/SwiGLU/control
interventions give **50 cases**, at width 32 and four heads. The original 15
learned-frequency cases are explicitly inapplicable: this candidate has no
learned-frequency parameter. The previous 65-case Log-Scaled LeRoPE studies
remain separate evidence.

Each CPU FP32 model is compared with its FP64 copy using the same quantized
parameters, tokens, targets and FP32 RMS epsilon. The FP32 rotary tables are
promoted unchanged, so this checks arithmetic with stored phases rather than
the accuracy of independently recomputed angles. The forward criterion requires
finite values and relative L2 logit error at most 0.001. The stronger criterion
also requires every parameter gradient to exist and remain finite, global
gradient relative L2 error at most 0.001, and relative loss error at most 0.001.
Per-parameter errors and all failures are retained. No optimizer update occurs.

Source and plan must be committed before the diagnostic runs. The program
archives the frozen production and diagnostic sources, fingerprints parameters
and data, verifies unchanged weights, and rejects changed archived evidence.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.inverse_width_stack_stress.run
```

These are freshly initialized stress models, not replays of trained checkpoints
or a large-task learning evaluation. Results cannot tune or reselect the current
short-suite winner or its running long diagnostic.

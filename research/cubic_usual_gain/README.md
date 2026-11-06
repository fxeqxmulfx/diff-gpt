# Cubic-sum test of the usual attention gain

The user requested a retest with `1/sqrt(d)`. The [plan](plan.json) declares
three seeds of the full normalized SSMax model at the original one-layer,
width-128 size. Only the attention gain changes from `1/d` to the parent's
existing `exp(log_alpha)/d`, which represents `1/sqrt(d)`. SSMax's `log(N)`,
Q/K RMS, fixed RoPE and every other block remain intact. The original LR1,
original LR2 and inverse-width controls are reused, including all failures.

Run this after the complete [depth/width study](../cubic_capacity/README.md)
has been audited and committed. The capacity grid remains unchanged. Validate
and freeze the exact implementation before these three training runs.

This tests a mechanism suggested by the weaker Q/K gradients and nearly uniform
attention observed in the existing-model diagnostic. It does not yet establish
that scaling causes the generalization failure. The one-task result cannot
select a new winner of the fixed 75-cell benchmark.

The isolated adapter and runner are implemented. All three initializations match
the inverse-width controls in parameter values/order, RNG and shared buffers.
Replacing only the inverse gain override reproduces the usual-gain model's
forward values and every parameter gradient exactly; tied weights remain shared.

Five CPU checks pass, including actual interrupted checkpoint continuation and
guards against premature training or changed source archives. The existing native
CUDA test is ready to run after the capacity queue finishes. It retains the
previous tolerances, short-tail batches, compiled/captured comparison and nonempty
state restoration checks. Scientific training remains blocked by its automatic
readiness check until that test passes and the full capacity outcomes are
committed.

After the capacity study is complete, run the CUDA check, then record and commit
the complete validation before launching all three seeds:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_cubic_usual_gain.py -k cuda --junitxml=research/cubic_usual_gain/results/validation-cuda.xml > research/cubic_usual_gain/results/validation-cuda.txt 2>&1
PYTHONPATH=. .venv/bin/python -m research.cubic_usual_gain_reporting.record_validation
# Commit the validated sources and evidence before training.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.cubic_usual_gain.run --run
```

Status: implementation and CPU validation complete; CUDA validation and the three
scientific runs are queued. No new scientific training has started.

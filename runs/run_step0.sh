#!/bin/bash
# Baseline (step 0 + step 1 readout) on M4 Hourly, seeds 0 and 1.
# All PIDs are recorded in runs/step0.pid; stop ONLY with: kill $(cat runs/step0.pid)
cd /home/misha/python_projects/gpt
echo $$ > runs/step0.pid
for seed in 0 1; do
  PYTHONUNBUFFERED=1 M4_N_SERIES=414 M4_MAX_ITERS=20000 \
  M4_SEED=$seed M4_SAMPLERS=argmax,median M4_SAVE=runs/step0_seed$seed.pt \
    uv run --no-sync python -m benchmarks.m4_bench > runs/step0_seed$seed.log 2>&1 &
  uv_pid=$!
  echo $uv_pid >> runs/step0.pid
  sleep 5
  pgrep -P $uv_pid >> runs/step0.pid
  wait $uv_pid
done
touch runs/step0.done

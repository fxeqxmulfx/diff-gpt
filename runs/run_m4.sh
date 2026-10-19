#!/bin/bash
# usage: runs/run_m4.sh <step> <seed>...   (extra M4_* env vars pass through)
# PIDs go to runs/<step>.pid; stop ONLY with: kill $(cat runs/<step>.pid)
step=$1; shift
cd /home/misha/python_projects/gpt
export DIFF_GPT_TOKENIZER=${DIFF_GPT_TOKENIZER:-lgt}   # accepted at step 2 (now the package default)
export M4_HEAD=${M4_HEAD:-mixture}                       # accepted at step 3
export M4_LABEL_SMOOTHING_SIGMA=${M4_LABEL_SMOOTHING_SIGMA:-0}
echo $$ > runs/$step.pid
for seed in "$@"; do
  PYTHONUNBUFFERED=1 M4_N_SERIES=${M4_N_SERIES:-414} M4_MAX_ITERS=${M4_MAX_ITERS:-20000} \
  M4_SEED=$seed \
  M4_SAMPLERS=${M4_SAMPLERS:-argmax,median} M4_SAVE=runs/${step}_seed$seed.pt \
    uv run --no-sync python -m benchmarks.m4_bench > runs/${step}_seed$seed.log 2>&1 &
  uv_pid=$!; echo $uv_pid >> runs/$step.pid; sleep 5; pgrep -P $uv_pid >> runs/$step.pid
  wait $uv_pid
done
touch runs/$step.done

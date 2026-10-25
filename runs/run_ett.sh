#!/bin/bash
# usage: runs/run_ett.sh <step> <seed>...   (extra ETT_* / DIFF_GPT_* env vars pass through)
# PIDs go to runs/<step>.pid; stop ONLY with: kill $(cat runs/<step>.pid)
step=$1; shift
cd "$(dirname "$0")/.."
echo $$ > runs/$step.pid
for seed in "$@"; do
  PYTHONUNBUFFERED=1 ETT_PRED_LENS=${ETT_PRED_LENS:-96} ETT_MAX_ITERS=${ETT_MAX_ITERS:-2000} \
  ETT_SEED=$seed ETT_SAVE=runs/$step \
    uv run --no-sync python -m diff_gpt.interfaces.ett_bench > runs/${step}_seed$seed.log 2>&1 &
  uv_pid=$!; echo $uv_pid >> runs/$step.pid; sleep 5; pgrep -P $uv_pid >> runs/$step.pid
  wait $uv_pid
done
touch runs/$step.done

#!/bin/bash
cd /home/misha/python_projects/gpt
export M4_N_SERIES=414 M4_MAX_ITERS=1500 M4_EVAL_INTERVAL=100000 M4_SAMPLERS=argmax DIFF_GPT_TOKENIZER=lgt PYTHONUNBUFFERED=1
run() { M4_SEED=$1 M4_COMPILE_MODE=$2 OMP_NUM_THREADS=${3:-6} uv run --no-sync python -m benchmarks.m4_bench > runs/probe_$4.log 2>&1; }
echo "== single, default compile =="; t0=$(date +%s); run 0 max-autotune-no-cudagraphs 6 single; echo "wall $(( $(date +%s)-t0 )) s, $(grep -o 'train_time_s=[0-9]*' runs/probe_single.log)"
echo "== 4 concurrent, default compile, OMP=2 =="; t0=$(date +%s); for s in 0 1 2 3; do run $s max-autotune-no-cudagraphs 2 par$s & done; wait; echo "wall $(( $(date +%s)-t0 )) s for 4 runs; $(grep -ho 'train_time_s=[0-9]*' runs/probe_par*.log | tr '\n' ' ')"
for mode in reduce-overhead max-autotune; do echo "== single, compile=$mode =="; t0=$(date +%s); run 0 $mode 6 $mode; echo "wall $(( $(date +%s)-t0 )) s, $(grep -o 'train_time_s=[0-9]*' runs/probe_$mode.log) $(grep -c Traceback runs/probe_$mode.log) tracebacks; sMAPE $(grep -o 'sMAPE=[0-9.]*' runs/probe_$mode.log)"; done
echo "== single reference sMAPE: $(grep -o 'sMAPE=[0-9.]*' runs/probe_single.log) =="
touch runs/probe_speed.done

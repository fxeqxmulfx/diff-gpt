#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
root=runs/main-model-tensor-hard-01
evidence=research/main_model/results/tensor-model-02
[[ ! -e "$root" ]]
sha256sum --check --status "$evidence/training-build.sha256"
mkdir -p "$root"
cp research/main_model/TENSOR_HARD_PLAN.txt "$root/PLAN.txt"
cp research/main_model/run_tensor_hard.sh "$root/launcher.sh"
cp "$evidence/fp16-tf32" "$root/tensor-transformer"
chmod a-w "$root/tensor-transformer"
jq -n '{purpose:"four_cell_hard_task_screen_not_full_suite_promotion",candidate:"static_fp16_forward_tf32_backward",main_control:"runs/main-model-mhc-wsd-01/mhc",cells:[{task:"basis-depth-hard",seed:0,tier:"short"},{task:"arithmetic-sum-pow4",seed:1,tier:"long"},{task:"arithmetic-sum-pow4",seed:0,tier:"long"},{task:"basis-parity24",seed:0,tier:"long"}],strict_ad_misses_retained:true,full_suite_promotion:false}' > "$root/screen.json"
while read -r task seed tier; do
    "$root/tensor-transformer" cell "$root" "$task" "$seed" mhc "$tier" > "$root/$task-seed$seed.log" 2>&1
    result="$root/mhc/$task/seed$seed/result.json"
    if ! jq -e '.status == "passed"' "$result" > /dev/null; then
        jq -n --arg task "$task" --argjson seed "$seed" '{status:"rejected_lost_main_pass",task:$task,seed:$seed,remaining_cells:"unrun"}' > "$root/screen-status.json"
        exit 0
    fi
done <<'CELLS'
basis-depth-hard 0 short
arithmetic-sum-pow4 1 long
arithmetic-sum-pow4 0 long
basis-parity24 0 long
CELLS
jq -n '{status:"four_hard_cells_completed",full_suite_promotion:false}' > "$root/screen-status.json"

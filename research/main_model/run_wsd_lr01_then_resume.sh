#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
results=research/main_model/results/wsd-lr01
cohort=runs/main-model-quartic-wsd-lr01-01
binary="$cohort/main-quartic-wsd-lr01"
previous=runs/main-model-mhc-02
paused="$previous/mhc/arithmetic-sum-pow4/seed0"

check_paused() {
    sha256sum --check research/main_model/results/mhc-binary.sha256
    sha256sum --check "$results/paused-mhc-artifacts.sha256"
    jq -e --slurpfile expected "$results/paused-mhc-checkpoint.json" \
        '{source_sha256,ptx_sha256,completed,state_sha256,history_sha256} == $expected[0]' \
        "$paused/checkpoint.json" > /dev/null
    jq -e '.clean == true and .durable_completed == 66082 and .possible_completed == 66082' \
        "$paused/execution.json" > /dev/null
    jq -e '.status == "stopped" and .completed == 66082' "$paused/result.json" > /dev/null
    test ! -e "$paused/STOP"
}

check_preflight() {
    jq -e '.passed == true and .peak_lr == 0.01 and .json_float_roundtrip == true and .warmup_updates == 2000 and .adam_betas == [0.9,0.95] and .weight_decay == 0.1 and .decay_updates == 20000 and .decay_shape == "cosine"' \
        "$results/validation.json" > /dev/null
    sha256sum --check "$results/artifacts.sha256"
    sha256sum --check research/main_model/results/wsd/completion-artifacts.sha256
    jq -e '.completed == 3 and (.integrity_failures | length) == 0' research/main_model/results/wsd/audit.json > /dev/null
    jq -e --slurpfile validation "$results/validation.json" \
        '.source_sha256 == $validation[0].source_sha256 and .ptx_sha256 == $validation[0].ptx_sha256 and .expected_cells == 3' \
        "$cohort/protocol.json" > /dev/null
    test ! -e "$cohort/STOP"
    for seed in 0 1 2; do
        test ! -e "$cohort/adamw-wsd-cosine20-warmup2000/seed$seed"
    done
    check_paused
}

if [[ ${1:-run} == --check ]]; then
    check_preflight
    exit 0
fi
[[ ${1:-run} == run ]]
exec 9> "$cohort/launcher.lock"
flock --nonblock 9
check_preflight

phase() {
    jq -n --arg phase "$1" --arg timestamp "$(date --utc --iso-8601=seconds)" \
        '{phase:$phase,updated_at:$timestamp}' > "$results/launcher-state.pending"
    mv "$results/launcher-state.pending" "$results/launcher-state.json"
}

phase wsd_running
"$binary" suite "$cohort" >> "$cohort/cohort.log" 2>&1
"$binary" audit "$cohort" > "$results/audit-output.txt"
jq -e '.expected == 3 and .completed == 3 and (.integrity_failures | length) == 0' \
    "$cohort/audit.json" > /dev/null
cp "$cohort/audit.json" "$results/audit.json"
jq --argjson peak 0.01 -f research/main_model/wsd_summary.jq \
    "$cohort/audit.json" > "$results/summary.json"
jq -r '["seed","status","first99_epochs","first99_update","best_accuracy","best_epoch","final_accuracy","final_epochs","completed_updates","seconds"],(.results[]|[.seed,.status,.first99_epochs,.first99_update,.best_accuracy,.best_epoch,.final_accuracy,.final_epochs,.completed_updates,.seconds])|@tsv' \
    "$results/summary.json" > "$results/summary.tsv"
jq -n --slurpfile base research/main_model/results/wsd/summary.json --slurpfile higher "$results/summary.json" \
    -f research/main_model/wsd_compare.jq > "$results/comparison.json"
phase wsd_audited

check_paused
rm "$previous/STOP"
phase mhc_resuming
"$previous/main-transformer" resume "$previous" arithmetic-sum-pow4 0 mhc long >> "$previous/cohort.log" 2>&1
phase main_mhc_long_running
"$previous/main-transformer" suite "$previous" long >> "$previous/cohort.log" 2>&1
"$previous/main-transformer" report "$previous" >> "$previous/cohort.log" 2>&1
research/main_model/report.sh >> "$previous/cohort.log" 2>&1
phase comparison_audited

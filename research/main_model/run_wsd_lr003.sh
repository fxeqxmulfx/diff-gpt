#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
results=research/main_model/results/wsd-lr003
cohort=runs/main-model-quartic-wsd-lr003-01
binary="$cohort/main-quartic-wsd-lr003"
previous=runs/main-model-quartic-wsd-lr001-01

phase() {
    jq -n --arg phase "$1" --arg timestamp "$(date --utc --iso-8601=seconds)" \
        '{phase:$phase,updated_at:$timestamp}' > "$results/launcher-state.pending"
    mv "$results/launcher-state.pending" "$results/launcher-state.json"
}
check_build() {
    sha256sum --check "$results/build-artifacts.sha256"
    test ! -e "$cohort/STOP"
    for seed in 0 1 2; do
        test ! -e "$cohort/adamw-wsd-cosine20-warmup2000/seed$seed"
    done
    jq -e --slurpfile control runs/main-model-quartic-wsd-01/protocol.json \
        '.ptx_sha256 == $control[0].ptx_sha256 and .train_sha256 == $control[0].train_sha256 and .validation_sha256 == $control[0].validation_sha256 and .peak_lr == 0.003 and .json_float_roundtrip == true' \
        "$cohort/protocol.json" > /dev/null
}
if [[ ${1:-run} == --check ]]; then
    check_build
    exit 0
fi
[[ ${1:-run} == run ]]
exec 9> "$cohort/launcher.lock"
flock --nonblock 9
check_build
phase waiting_for_lr001_completion
# Wait for the exact prior launcher and its inherited child descriptors to exit.
# This does not poll result files or stop the still-running scientific cells.
exec 8<> "$previous/launcher.lock"
flock 8
for audit in research/main_model/results/wsd/audit.json research/main_model/results/wsd-lr01/audit.json research/main_model/results/wsd-lr001/audit.json; do
    jq -e '.expected == 3 and .completed == 3 and (.integrity_failures|length) == 0' "$audit" > /dev/null
done
test -f runs/main-model-mhc-02/STOP
flock --unlock 8
check_build

phase gpu_validation
"$cohort/native-tests" --ignored --test-threads=1 > "$results/gpu-tests.txt" 2>&1
jq '.passed=true | .gpu_tests_passed=4 | .gpu_validation="passed"' \
    "$results/validation-before-gpu.json" > "$results/validation.json"
sha256sum "$results/validation.json" "$results/gpu-tests.txt" > "$results/validation-artifacts.sha256"
phase wsd_running
"$binary" suite "$cohort" >> "$cohort/cohort.log" 2>&1
"$binary" audit "$cohort" > "$results/audit-output.txt"
jq -e '.expected == 3 and .completed == 3 and (.integrity_failures|length) == 0' "$cohort/audit.json" > /dev/null
cp "$cohort/audit.json" "$results/audit.json"
jq --argjson peak 0.003 -f research/main_model/wsd_summary.jq "$results/audit.json" > "$results/summary.json"
jq -r '["seed","status","first99_epochs","first99_update","first99_accuracy","seconds"],(.results[]|[.seed,.status,.first99_epochs,.first99_update,.first99_accuracy,.seconds])|@tsv' \
    "$results/summary.json" > "$results/summary.tsv"
jq -s -f research/main_model/wsd_compare_four.jq research/main_model/results/wsd/summary.json \
    research/main_model/results/wsd-lr01/summary.json research/main_model/results/wsd-lr001/summary.json "$results/summary.json" > "$results/comparison-all.json"
jq -e '.complete == true and .selected_peak_lr != null' "$results/comparison-all.json" > /dev/null
phase lr_studies_complete_fresh_main_mhc_restart_pending

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
results=research/main_model/results/mhc-wsd
cohort=runs/main-model-mhc-wsd-01
binary="$cohort/main-transformer-wsd"
phase() {
    jq -n --arg phase "$1" --arg timestamp "$(date --utc --iso-8601=seconds)" \
      '{phase:$phase,updated_at:$timestamp,peak_lr:0.001,cohort:"runs/main-model-mhc-wsd-01"}' > "$results/launcher-state.pending"
    mv "$results/launcher-state.pending" "$results/launcher-state.json"
}
check_build() {
    sha256sum --check --quiet "$results/build-artifacts.sha256"
    sha256sum --check --quiet "$results/source-artifacts.sha256"
    jq -e '.passed and .cpu_tests_passed==10 and .gpu_tests_passed==5 and .gradient_directions==1308 and .native_data_preflight_tasks==35' "$results/validation.json" > /dev/null
    jq -e '.source_verified and (.audit_failures|length)==0 and all(.rows[];.status=="pending")' "$results/audit-before-run.json" > /dev/null
    test -f runs/main-model-mhc-02/STOP
    test -f runs/main-model-quartic-wsd-lr003-01/STOP
    test ! -e "$cohort/STOP"
    test ! -e "$cohort/main"
    test ! -e "$cohort/mhc"
    test ! -e "$cohort/started.json"
}
if [[ ${1:-run} == --check ]]; then
    check_build
    exit 0
fi
[[ ${1:-run} == run ]]
exec 9> "$cohort/launcher.lock"
flock --nonblock 9
check_build
phase starting_fresh
cp "$results/launcher-state.json" "$cohort/started.json"
trap 'phase execution_or_audit_failure; exit 1' ERR
phase running_short
"$binary" suite "$cohort" short >> "$cohort/cohort.log" 2>&1
bash research/main_model/mhc_wsd_report.sh > "$results/short-audit-output.txt" 2>&1
cp "$results/audit.json" "$results/audit-after-short.json"
cp "$results/comparison.json" "$results/comparison-after-short.json"
if [[ -e "$cohort/STOP" ]]; then
    phase clean_stopped_by_request
    exit 0
fi
jq -e '(.audit_failures|length)==0 and all(.rows[]|select(.tier=="short");.audit=="passed" and (.status=="passed" or .status=="budget_exhausted"))' "$results/audit.json" > /dev/null
jq '{phase:"short_complete",totals:.totals,short:[.comparisons[]|select(.tier=="short")]}' "$results/comparison.json"
phase running_long
"$binary" suite "$cohort" long >> "$cohort/cohort.log" 2>&1
bash research/main_model/mhc_wsd_report.sh > "$results/final-audit-output.txt" 2>&1
if [[ -e "$cohort/STOP" ]]; then
    phase clean_stopped_by_request
    exit 0
fi
jq -e '.complete and .source_verified and (.audit_failures|length)==0 and (.rows|length)==210' "$results/audit.json" > /dev/null
sha256sum "$results/audit.json" "$results/comparison.json" "$results/promotion-gate.json" "$results/long-epochs.tsv" > "$results/completion-artifacts.sha256"
phase complete_audited_promotion_and_next_research_pending
jq '{phase:"complete",totals,promotion}' "$results/comparison.json"

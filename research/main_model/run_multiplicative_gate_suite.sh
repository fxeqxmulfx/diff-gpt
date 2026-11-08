#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
root=runs/main-model-multiplicative-gate-01
results=research/main_model/results/multiplicative-gate-suite-01
binary="$root/multiplicative-gated-transformer"
auditor="$results/independent-auditor"
phase() {
    jq -n --arg status "$1" --arg task "${2:-}" --argjson seed "${3:-null}" \
      --arg timestamp "$(date --utc --iso-8601=seconds)" \
      '{status:$status,task:$task,seed:$seed,updated_at:$timestamp}' > "$root/experiment-status.pending"
    mv "$root/experiment-status.pending" "$root/experiment-status.json"
}
check_build() {
    sha256sum --check --quiet research/main_model/results/multiplicative-gate-validation-final/evidence.sha256
    sha256sum --check --quiet research/main_model/results/gpu-suite-01/promotion-evidence.sha256
    sha256sum --check --quiet "$results/registration.sha256"
    jq -e '.audit_passed and .source_verified and .phase=="prepared" and .unrun_cells==105 and .promotion.eligible_for_promotion==false' "$results/audit-before.json" > /dev/null
    test ! -e "$root/mhc"
    test ! -e "$root/gated"
    test ! -e "$root/main"
    test ! -e "$root/STOP"
    test ! -e "$root/started.json"
}
if [[ ${1:-run} == --check ]]; then check_build; exit 0; fi
[[ ${1:-run} == run ]]
exec 9> "$root/launcher.lock"
flock --nonblock 9
check_build
cp "$root/experiment-status.json" "$root/started.json"
trap 'phase launcher_or_audit_error; exit 1' ERR
for tier in short long; do
    phase "running_$tier"
    while IFS=$'\t' read -r task seed; do
        if [[ -e "$root/STOP" ]]; then
            phase stopped_between_cells
            "$auditor" "$root" "$results/stopped-audit.json" > "$results/stopped-audit.log" 2>&1
            exit 0
        fi
        if "$binary" cell "$root" "$task" "$seed" gated "$tier" > "$root/$task-seed$seed.log" 2>&1; then
            :
        else
            code=$?
            phase execution_error "$task" "$seed"
            jq -n --arg task "$task" --argjson seed "$seed" --argjson code "$code" \
              '{status:"execution_error",task:$task,seed:$seed,exit_code:$code,retry_allowed:false}' > "$root/execution-error.json"
            exit "$code"
        fi
        result="$root/gated/$task/seed$seed/result.json"
        status=$(jq -r '.status' "$result")
        jq '{case,seed,status,first99_epochs,completed}' "$result"
        case "$status" in
            passed) ;;
            budget_exhausted)
                phase rejected_lost_main_pass "$task" "$seed"
                "$auditor" "$root" "$results/rejection-audit.json" > "$results/rejection-audit.log" 2>&1
                exit 0 ;;
            stopped)
                phase stopped "$task" "$seed"
                "$auditor" "$root" "$results/stopped-audit.json" > "$results/stopped-audit.log" 2>&1
                exit 0 ;;
            *) phase invalid_terminal_status "$task" "$seed"; exit 1 ;;
        esac
    done < <(jq -r --arg tier "$tier" '.cells[]|select(.tier==$tier)|[.task,.seed]|@tsv' "$root/experiment.json")
    if [[ $tier == short ]]; then
        phase short_complete
        "$auditor" "$root" "$results/short-audit.json" > "$results/short-audit.log" 2>&1
        jq -e '.audit_passed and .promotion.short.solved==75' "$results/short-audit.json" > /dev/null
    else
        phase complete
        "$auditor" "$root" "$results/final-audit.json" > "$results/final-audit.log" 2>&1
        jq -e '.audit_passed and .solved_cells==105' "$results/final-audit.json" > /dev/null
    fi
done
sha256sum "$results/short-audit.json" "$results/final-audit.json" > "$results/completion.sha256"

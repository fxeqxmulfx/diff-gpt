#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
root=runs/main-model-axial-01
results=research/main_model/results/axial-suite-01
binary="$root/axial-transformer"
auditor="$results/independent-auditor"
if [[ ${1:-run} == --prepare ]]; then
    validation=research/main_model/results/axial-validation-final
    test ! -e "$root"
    test ! -e "$results"
    sha256sum --check --quiet "$validation/evidence.sha256"
    jq -e '.implementation_passed' "$validation/validation.json" > /dev/null
    mkdir -p "$results"
    cargo build --offline --release --manifest-path research/main_model/axial_audit/Cargo.toml > "$results/auditor-build.log" 2>&1
    cp research/main_model/axial_audit/target/release/main-model-axial-audit "$auditor"
    cp -a "$validation/archive" "$root"
    cp "$validation/candidate" "$binary"
    cp research/main_model/AXIAL_LEROPE_YARN_PLAN.txt "$root/PLAN.txt"
    binary_hash=$(sha256sum "$binary" | cut -d' ' -f1)
    plan_hash=$(sha256sum "$root/PLAN.txt" | cut -d' ' -f1)
    jq -n --slurpfile native "$validation/validation.json" --slurpfile suite research/complex_geometry_suite/suite.json \
      --arg binary "$binary_hash" --arg plan "$plan_hash" \
      '{candidate:"axial_time_feature_lerope_with_inference_yarn",candidate_arm:"axial-lerope-yarn",
        control_root:"runs/main-model-phase-score-01",control_arm:"phase-score-lerope",
        control_audit_sha256:"a9d297f7bea99ab9e294ab80e7a3c022a8476432278553b744a260954aba26e7",
        source_sha256:$native[0].source_sha256,ptx_sha256:$native[0].ptx_sha256,binary_sha256:$binary,
        plan_sha256:$plan,compiled_plan_sha256:$plan,expected_cells:105,short_cells:75,long_cells:30,
        fresh_initialization:true,parameter_count_matched:true,additional_real_parameters:0,shared_initialization_matched:true,
        strict_numerical_misses_retained:true,checkpoint_epochs:500,
        optimizer:"unchanged AdamW/WSD; base clip1 wd0.1; frequencies separate clip1 wd0",
        stop_rule:"Reject first verified lost selected-main pass at the unchanged budget; retain failure and remaining unrun cells.",
        promotion_rule:"All105main passes and fewer geometric-mean short epochs; report all long rows and regressions.",
        cells:[(["short","long"][] as $tier|$suite[0].tasks[]|select(.tier==$tier)|.name as $name|range(0;3)|{task:$name,seed:.,tier:$tier})]}' > "$root/experiment.json"
    jq -n '{status:"prepared",task:"",seed:null}' > "$root/experiment-status.json"
    "$auditor" "$root" "$results/audit-before.json" > "$results/audit-before.log" 2>&1
    sha256sum "$root/experiment.json" "$root/PLAN.txt" "$root/protocol.json" "$root/data-validation.json" "$binary" "$auditor" "$results/audit-before.json" > "$results/registration.sha256"
    exit 0
fi
phase() {
    jq -n --arg status "$1" --arg task "${2:-}" --argjson seed "${3:-null}" \
      --arg timestamp "$(date --utc --iso-8601=seconds)" \
      '{status:$status,task:$task,seed:$seed,updated_at:$timestamp}' > "$root/experiment-status.pending"
    mv "$root/experiment-status.pending" "$root/experiment-status.json"
}
check_build() {
    sha256sum --check --quiet research/main_model/results/axial-validation-final/evidence.sha256
    sha256sum --check --quiet research/main_model/results/phase-score-suite-01/promotion-evidence.sha256
    sha256sum --check --quiet "$results/registration.sha256"
    jq -e '.audit_passed and .source_verified and .phase=="prepared" and .unrun_cells==105 and .promotion.eligible_for_promotion==false' "$results/audit-before.json" > /dev/null
    test ! -e "$root/mhc"
    test ! -e "$root/lerope"
    test ! -e "$root/axial-lerope-yarn"
    test ! -e "$root/phase-score-lerope"
    test ! -e "$root/phase-mhc-lerope"
    test ! -e "$root/log-lerope"
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
        if "$binary" cell "$root" "$task" "$seed" axial-lerope-yarn "$tier" > "$root/$task-seed$seed.log" 2>&1; then
            :
        else
            code=$?
            phase execution_error "$task" "$seed"
            jq -n --arg task "$task" --argjson seed "$seed" --argjson code "$code" \
              '{status:"execution_error",task:$task,seed:$seed,exit_code:$code,retry_allowed:false}' > "$root/execution-error.json"
            exit "$code"
        fi
        result="$root/axial-lerope-yarn/$task/seed$seed/result.json"
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

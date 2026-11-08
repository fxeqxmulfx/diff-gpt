#!/usr/bin/env bash
# Additional three-encoder study; never substitutes for fixed105 coverage.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
root=runs/main-model-diffgpt-axial-01
results=research/main_model/results/diffgpt-axial-suite-01
validation=research/main_model/results/axial-validation-final
binary="$root/diffgpt-transformer"
auditor="$results/independent-auditor"
digest() { sha256sum "$1" | cut -d' ' -f1; }
phase() {
    jq -n --arg status "$1" --arg timestamp "$(date --utc --iso-8601=seconds)" \
      '{status:$status,updated_at:$timestamp}' > "$root/status.pending"
    mv "$root/status.pending" "$root/experiment-status.json"
}
case "${1:-}" in
--prepare)
    test ! -e "$root"
    test ! -e "$results"
    sha256sum --check --quiet "$validation/evidence.sha256"
    jq -e '.implementation_passed and .diffgpt_engineering_cases==12' "$validation/validation.json" > /dev/null
    sha256sum --check --quiet research/main_model/results/diffgpt-data-preparation-01/evidence.sha256
    mkdir -p "$root" "$results"
    cp "$validation/candidate" "$binary"
    cargo build --offline --release --manifest-path research/main_model/diffgpt_audit/Cargo.toml > "$results/auditor-build.log" 2>&1
    cp research/main_model/diffgpt_audit/target/release/diffgpt-axial-audit "$auditor"
    data=research/main_model/results/diffgpt-data-preparation-01/data-registration.json
    jq -n --slurpfile v "$validation/validation.json" --arg binary "$binary" --arg bh "$(digest "$binary")" \
      --arg validation "$validation/validation.json" --arg vh "$(digest "$validation/validation.json")" \
      --arg archive "$validation/context" --arg data "$data" --arg dh "$(digest "$data")" \
      '{kind:"diffgpt_axial_lerope_yarn",source_sha256:$v[0].source_sha256,ptx_sha256:$v[0].ptx_sha256,
        binary:$binary,binary_sha256:$bh,validation:$validation,validation_sha256:$vh,source_archive:$archive,
        data_registration:$data,data_registration_sha256:$dh,budget_updates:5000,batch_size:512,observation_updates:50,
        cells:[(["mash","lgt","mash-half"][] as $name|[1,2][] as $f|[11,17,23][] as $s|["sequence","axial"][]|
          {tokenizer:$name,features:$f,seed:$s,arm:.})],
        promotion_scope:"Additional evidence only; cannot erase a fixed105 regression.",
        first90_is_not_sustained90:true,checkpoint_epochs:500,no_python_execution:true}' > "$root/experiment.json"
    phase prepared
    "$auditor" "$root" "$results/audit-before.json" > "$results/audit-before.log" 2>&1
    jq -e '.audit_passed and .unrun_cells==36' "$results/audit-before.json" > /dev/null
    sha256sum "$root/experiment.json" "$binary" "$auditor" "$results/audit-before.json" > "$results/registration.sha256"
    ;;
--run)
    exec 9> "$root/launcher.lock"
    flock --nonblock 9
    test ! -e "$root/started.json"
    test ! -e "$root/STOP"
    sha256sum --check --quiet "$results/registration.sha256"
    sha256sum --check --quiet "$validation/evidence.sha256"
    cp "$root/experiment-status.json" "$root/started.json"
    trap 'phase execution_or_audit_error; exit 1' ERR
    phase running
    while IFS=$'\t' read -r tokenizer features seed arm; do
        if [[ -e "$root/STOP" ]]; then phase stopped; exit 0; fi
        "$binary" diffgpt-cell "$root" "$tokenizer" "$features" "$seed" "$arm" \
          > "$root/$tokenizer-f$features-seed$seed-$arm.log" 2>&1
        result="$root/$arm/$tokenizer/features$features/seed$seed/result.json"
        if [[ $(jq -r '.status' "$result") == stopped ]]; then phase stopped; exit 0; fi
        jq -e '.completed==5000 and (.status=="reached90" or .status=="budget_exhausted")' "$result" > /dev/null
    done < <(jq -r '.cells[]|[.tokenizer,.features,.seed,.arm]|@tsv' "$root/experiment.json")
    "$auditor" "$root" "$results/final-audit.json" > "$results/final-audit.log" 2>&1
    jq -e '.audit_passed and .completed_cells==36 and .unrun_cells==0' "$results/final-audit.json" > /dev/null
    phase complete
    printf 'All36DiffGPT cells completed and audited.\n'
    ;;
*) echo 'usage: run_diffgpt_suite.sh --prepare|--run' >&2; exit 2;;
esac

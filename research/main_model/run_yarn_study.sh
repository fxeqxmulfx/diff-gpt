#!/usr/bin/env bash
# Read-only checkpoint study, with a separately compiled CPU auditor.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
validation=research/main_model/results/yarn-validation-01
evidence=research/main_model/results/yarn-study-01
output=runs/main-model-yarn-study-01
control=runs/main-model-phase-score-01
audit=research/main_model/results/phase-score-suite-01/final-audit.json
suite=research/complex_geometry_suite/suite.json
digest() { sha256sum "$1" | cut -d' ' -f1; }
case "${1:-}" in
--prepare)
    test ! -e "$evidence"
    test ! -e "$output"
    sha256sum -c "$validation/artifacts.sha256"
    sha256sum -c "$validation/evidence.sha256"
    jq -e '.implementation_passed' "$validation/validation.json" >/dev/null
    jq -e '.audit_passed and .solved_cells==105' "$audit" >/dev/null
    mkdir -p "$evidence"
    cargo build --offline --release --manifest-path research/main_model/yarn_audit/Cargo.toml > "$evidence/auditor-build.log" 2>&1
    cp research/main_model/yarn_audit/target/release/main-model-yarn-audit "$evidence/independent-auditor"
    : > "$evidence/cells.jsonl"
    while IFS=$'\t' read -r name seed tier; do
        dir="$control/phase-score-lerope/$name/seed$seed"
        extent=$(jq -er --arg name "$name" '.tasks[]|select(.task==$name)|.training_extent' "$validation/data-lengths.json")
        jq -cn --arg task "$name" --argjson seed "$seed" --arg tier "$tier" \
          --argjson extent "$extent" --arg cp "$(digest "$dir/checkpoint.bin")" \
          --arg meta "$(digest "$dir/checkpoint.json")" --arg result "$(digest "$dir/result.json")" \
          --arg history "$(digest "$dir/history.jsonl")" \
          '{task:$task,seed:$seed,tier:$tier,training_extent:$extent,checkpoint_sha256:$cp,checkpoint_metadata_sha256:$meta,result_sha256:$result,history_sha256:$history}' >> "$evidence/cells.jsonl"
    done < <(jq -r '.rows[]|[.case,.seed,.tier]|@tsv' "$audit")
    jq -n --slurpfile cells "$evidence/cells.jsonl" --slurpfile v "$validation/validation.json" \
      --arg validation "$validation/validation.json" --arg vh "$(digest "$validation/validation.json")" \
      --arg binary "$validation/candidate" --arg binary_hash "$(digest "$validation/candidate")" \
      --arg archive "$validation/archive" --arg lengths "$validation/data-lengths.json" \
      --arg lh "$(digest "$validation/data-lengths.json")" --arg suite "$suite" \
      --arg control "$control" --arg audit "$audit" --arg ah "$(digest "$audit")" \
      '{kind:"lerope_yarn_frozen_checkpoint_study",training_updates:0,binary:$binary,binary_sha256:$binary_hash,
        source_sha256:$v[0].source_sha256,ptx_sha256:$v[0].ptx_sha256,validation:$validation,validation_sha256:$vh,
        source_archive:$archive,lengths:$lengths,lengths_sha256:$lh,suite:$suite,cells:$cells,
        control:{root:$control,arm:"phase-score-lerope",audit:$audit,audit_sha256:$ah,
          source_sha256:"7968240c10a78200255a400bb280186bd66647a121817720173e814eee40dced",
          ptx_sha256:"17c689bd6f9923534f8eed69dc047780274a28c3e45fbb25a597fccbddfb597f"}}' > "$evidence/registration.json"
    "$evidence/independent-auditor" "$evidence/registration.json" "$output" "$evidence/audit-before.json"
    jq -e '.audit_passed and .unrun_cells==105' "$evidence/audit-before.json" >/dev/null
    sha256sum "$evidence/registration.json" "$evidence/independent-auditor" "$evidence/audit-before.json" > "$evidence/registration.sha256"
    ;;
--run)
    test ! -e "$output"
    sha256sum -c "$evidence/registration.sha256"
    sha256sum -c "$validation/artifacts.sha256"
    "$validation/candidate" context-evaluate "$evidence/registration.json" "$output" > "$evidence/evaluation.log" 2>&1
    "$evidence/independent-auditor" "$evidence/registration.json" "$output" "$evidence/final-audit.json"
    jq '{audit_passed,phase,short_retained,long_retained,unrun_cells,lost_main_passes,training_updates}' "$evidence/final-audit.json"
    ;;
--report)
    test ! -e "$evidence/REPORT.txt"
    jq -e '.audit_passed' "$evidence/final-audit.json" >/dev/null
    jq -r '["task","seed","tier","status","training_extent","extended_examples","control_accuracy","candidate_accuracy"],
      (.rows[]|[.task,.seed,.tier,.status,.training_extent,.primary_extended_examples,.grouped_control.sequence_accuracy,.candidate.sequence_accuracy])|@tsv' \
      "$output/result.json" > "$evidence/primary.tsv"
    jq -r '["task","seed","split","examples","control_accuracy","candidate_accuracy"],
      (.rows[] as $r|($r.auxiliary_extrapolation_splits // {}|to_entries[])|
      [$r.task,$r.seed,.key,.value.control.examples,.value.control.sequence_accuracy,.value.candidate.sequence_accuracy])|@tsv' \
      "$output/result.json" > "$evidence/auxiliary.tsv"
    jq -r '"LeRoPE+YaRN frozen-checkpoint study",
      "Independent audit passed: \(.audit_passed). Phase: \(.phase).",
      "Retained short: \(.short_retained)/75. Retained long: \(.long_retained)/30. Unrun: \(.unrun_cells).",
      "Lost passes: \(.lost_main_passes|tojson).",
      "No training updates, no new first99 epochs, no epoch promotion.",
      "All primary/auxiliary accuracy pairs are retained separately in primary.tsv and auxiliary.tsv.",
      "The stop rule leaves subsequent cells unrun; zero evaluated long cells is not zero long coverage.",
      "The selected score+phase-mHC main remains unchanged."' "$evidence/final-audit.json" > "$evidence/REPORT.txt"
    sha256sum "$evidence/final-audit.json" "$output/result.json" "$evidence/primary.tsv" "$evidence/auxiliary.tsv" "$evidence/REPORT.txt" > "$evidence/closure.sha256"
    ;;
*) echo 'usage: run_yarn_study.sh --prepare|--run|--report' >&2; exit 2;;
esac

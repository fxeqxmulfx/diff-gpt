#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
audit=${1:?usage: diffgpt_report.sh AUDIT_JSON FRESH_OUTPUT}
out=${2:?usage: diffgpt_report.sh AUDIT_JSON FRESH_OUTPUT}
test ! -e "$out"
mkdir -p "$out"
jq -f research/main_model/diffgpt_report.jq "$audit" > "$out/report.json"
jq -r '["tokenizer","features","seed","sequence_status","axial_status","sequence_first90_epochs","axial_first90_epochs","epoch_ratio","sequence_final_decoded","axial_final_decoded","sequence_final_reanchored","axial_final_reanchored","lost_sequence_pass"],
  (.paired_rows[]|[.tokenizer,.features,.seed,.sequence_status,.axial_status,.sequence_first90_epochs,.axial_first90_epochs,.axial_over_sequence_first90_epochs,
    .sequence_endpoint.decoded_accuracy,.axial_endpoint.decoded_accuracy,.sequence_endpoint.forecast_reanchored_accuracy,.axial_endpoint.forecast_reanchored_accuracy,.lost_sequence_first90_pass])|@tsv' "$out/report.json" > "$out/paired-epochs.tsv"
jq -r '["tokenizer","features","seed","arm","context","native_decoded","yarn_decoded","native_forecast_reanchored","yarn_forecast_reanchored"],
  (.rows[] as $r|$r.extrapolation[]|[$r.cell.tokenizer,$r.cell.features,$r.cell.seed,$r.cell.arm,.context,
    .native.decoded_accuracy,.yarn.decoded_accuracy,.native.forecast_reanchored_accuracy,.yarn.forecast_reanchored_accuracy])|@tsv' "$audit" > "$out/extrapolation.tsv"
jq -r '"Additional native DiffGPT axial/sequence comparison:36audited runs",
  (.groups[]|"\(.tokenizer),features\(.features): sequence\(.sequence_reached90)/3,axial\(.axial_reached90)/3 reached90; lost sequence passes\(.axial_lost_sequence_passes)."),
  "All18paired rows:paired-epochs.tsv. All72context/arm rows:extrapolation.tsv.","",.caveats[]' "$out/report.json" > "$out/REPORT.txt"
sha256sum "$audit" "$out/report.json" "$out/paired-epochs.tsv" "$out/extrapolation.tsv" "$out/REPORT.txt" > "$out/evidence.sha256"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
audit=${1:?usage: phase_mhc_report.sh AUDIT_JSON FRESH_REPORT_DIRECTORY}
out=${2:?usage: phase_mhc_report.sh AUDIT_JSON FRESH_REPORT_DIRECTORY}
test ! -e "$out"
mkdir -p "$out"
jq --slurpfile control research/main_model/results/lerope-suite-01/final-audit.json \
  -f research/main_model/phase_mhc_report.jq "$audit" > "$out/report.json"
jq -r '["tier","task","seed","status","audit","main_epochs","candidate_epochs","epoch_ratio","main_updates","candidate_updates","previous_observed_epochs","heldout_accuracy","observations","checkpoint_sha256"],
  (.rows[]|[.tier,.case,.seed,.status,.audit,.main_epochs,.candidate_epochs,.candidate_over_main_epochs,.main_updates,.candidate_updates,.previous_observed_epochs,.heldout_accuracy,.observations,.checkpoint_sha256])|@tsv' \
  "$out/report.json" > "$out/paired-epochs.tsv"
jq -r '["task","seed","status","main_epochs","candidate_epochs","epoch_ratio","main_updates","candidate_updates","heldout_accuracy"],
  (.rows[]|select(.tier=="long")|[.case,.seed,.status,.main_epochs,.candidate_epochs,.candidate_over_main_epochs,.main_updates,.candidate_updates,.heldout_accuracy])|@tsv' \
  "$out/report.json" > "$out/long-epochs.tsv"
jq -r '"Phase-conjugated mHC versus selected LeRoPE main",
  "Phase: \(.phase). Independent audit passed: \(.audit_passed).",
  "Short: \(.promotion.short.solved)/75. Full-tier epoch ratio: \(.promotion.short.geometric_mean_epoch_ratio // "unavailable").",
  "Long: \(.promotion.long.solved)/30. Full-tier epoch ratio: \(.promotion.long.geometric_mean_epoch_ratio // "unavailable").",
  "Gate: \(.promotion.status). Eligible for promotion: \(.promotion.eligible_for_promotion).",
  "Verified lost main passes: \(.promotion.lost_main_passes|length).",
  "",
  "All task/seed rows and regressions: paired-epochs.tsv; all30 long rows: long-epochs.tsv.",
  "",
  .caveats[]' "$out/report.json" > "$out/REPORT.txt"
if jq -e '.audit_passed and .promotion.long.complete' "$out/report.json" > /dev/null; then
    read -r lower upper logarithmic < <(jq -r '[.rows[]|select(.tier=="long")|.main_epochs,.candidate_epochs]|
      (min) as $lo|(max) as $hi|if $lo>0 then [$lo*0.7,$hi*1.4,1] else [0,$hi*1.1,0] end|@tsv' "$out/report.json")
    gnuplot -c research/main_model/phase_mhc_epochs.gnuplot "$out/long-epochs.tsv" "$out/long-epochs" "$lower" "$upper" "$logarithmic"
fi
sha256sum "$audit" "$out/report.json" "$out/paired-epochs.tsv" "$out/long-epochs.tsv" "$out/REPORT.txt" > "$out/evidence.sha256"
if [[ -e "$out/long-epochs.svg" ]]; then
    sha256sum "$out/long-epochs.svg" "$out/long-epochs.png" >> "$out/evidence.sha256"
fi

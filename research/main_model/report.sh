#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
results=research/main_model/results
research/main_model/audit/target/release/main-model-audit runs/main-model-mhc-02 "$results/mhc-audit.json"
jq -L research/main_model -f research/main_model/report.jq "$results/mhc-audit.json" > "$results/mhc-comparison.json"
jq -r '["task","seed","main_status","main_first99_epochs","main_updates","main_final_epochs","main_final_accuracy","mhc_status","mhc_first99_epochs","mhc_updates","mhc_final_epochs","mhc_final_accuracy","mhc_over_main_epochs"],(.long_epochs[]|[.task,.seed,.main_status,.main_first99_epochs,.main_updates,.main_final_epochs,.main_final_accuracy,.mhc_status,.mhc_first99_epochs,.mhc_updates,.mhc_final_epochs,.mhc_final_accuracy,.mhc_over_main_epochs])|@tsv' "$results/mhc-comparison.json" > "$results/mhc-long-epochs.tsv"
jq -r '"Main model versus four-stream mHC", "", "Complete: \(.complete)", "Coverage gate: \(.coverage_gate.status)", "Lost main passes: \(.coverage_gate.lost_main_passes|tojson)", "Epochs count actual training examples, including partial batches.", "", (.totals[]|"\(.tier) / \(.arm): \(.statuses|tojson)"), "", (.comparisons[]|"\(.tier): \(.both_passed) pairs both passed; mHC earlier/tied/later = \(.mhc_fewer_epochs)/\(.same_epochs)/\(.mhc_more_epochs); full-suite geometric mean mHC/main epochs = \(.full_suite_geometric_mean_mhc_over_main); solved-pairs-only mean = \(.solved_pairs_geometric_mean_mhc_over_main) [\(.aggregate_scope)]"), "", "Long tasks (first99 epochs; a budget miss shows its final epoch count):", (.long_epochs[]|"\(.task), seed \(.seed): main \(if .main_status=="passed" then .main_first99_epochs else "\(.main_status) @ \(.main_final_epochs)" end); mHC \(if .mhc_status=="passed" then .mhc_first99_epochs else "\(.mhc_status) @ \(.mhc_final_epochs)" end); ratio \(.mhc_over_main_epochs)"), "", .limitations[]' "$results/mhc-comparison.json" > "$results/mhc-comparison.txt"
paired=$(jq -r '.comparisons[]|select(.tier=="long")|.both_passed' "$results/mhc-comparison.json")
if (( paired > 0 )); then
    min_epochs=$(jq '[.long_epochs[]|select(.main_status=="passed" and .mhc_status=="passed")|.main_first99_epochs,.mhc_first99_epochs|select(.>0)]|min*0.8' "$results/mhc-comparison.json")
    max_epochs=$(jq '[.long_epochs[]|.main_final_epochs,.mhc_final_epochs|select(.!=null)]|max*1.2' "$results/mhc-comparison.json")
    gnuplot -e "paired=$paired; min_epochs=$min_epochs; max_epochs=$max_epochs" research/main_model/long_epochs.gnuplot
fi

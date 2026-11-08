#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
results=research/main_model/results/mhc-wsd
cohort=runs/main-model-mhc-wsd-01
"$cohort/main-model-wsd-audit" "$cohort" "$results/audit.json"
jq -L research/main_model -f research/main_model/report.jq "$results/audit.json" > "$results/comparison.pending"
jq -L research/main_model --slurpfile baseline "$results/incumbent-passes.json" \
    'include "restart_gate"; restart_decision($baseline[0];75;105)' \
    "$results/audit.json" > "$results/promotion-gate.pending"
jq --slurpfile gate "$results/promotion-gate.pending" --slurpfile audit "$results/audit.json" \
    --slurpfile suite research/complex_geometry_suite/suite.json \
    '. + {optimizer:{peak_lr:0.001,betas:[0.9,0.95],weight_decay:0.1,warmup_fraction:0.02,cosine_decay_fraction:0.2,gradient_clip:1},
      promotion:$gate[0]} |
     .long_epochs |= map(. as $item |
       ([$suite[0].tasks[]|select(.name==$item.task)]|first) as $task |
       ([$audit[0].rows[]|select(.case==$item.task and .seed==$item.seed and .arm=="main")]|first) as $main |
       ([$audit[0].rows[]|select(.case==$item.task and .seed==$item.seed and .arm=="mhc")]|first) as $mhc |
       . + {budget_updates:$task.case.max_updates,observation_interval_updates:$task.case.observe,
         main_previous_observed_epochs:$main.previous_observed_epochs,mhc_previous_observed_epochs:$mhc.previous_observed_epochs})' \
    "$results/comparison.pending" > "$results/comparison-final.pending"
mv "$results/comparison-final.pending" "$results/comparison.json"
mv "$results/promotion-gate.pending" "$results/promotion-gate.json"
jq -r '["task","seed","budget_updates","observation_interval_updates","main_status","main_first99_epochs","main_previous_observed_epochs","main_updates","main_final_accuracy","mhc_status","mhc_first99_epochs","mhc_previous_observed_epochs","mhc_updates","mhc_final_accuracy","mhc_over_main_epochs"],
  (.long_epochs[]|[.task,.seed,.budget_updates,.observation_interval_updates,.main_status,.main_first99_epochs,.main_previous_observed_epochs,.main_updates,.main_final_accuracy,.mhc_status,.mhc_first99_epochs,.mhc_previous_observed_epochs,.mhc_updates,.mhc_final_accuracy,.mhc_over_main_epochs])|@tsv' \
  "$results/comparison.json" > "$results/long-epochs.tsv"
jq -r '"Fresh main versus mHC, LR0.001 / AdamW(.9,.95) / WSD2%-20%", "", "Complete: \(.complete)",
  (.totals[]|"\(.tier) / \(.arm): \(.statuses|tojson)"),
  (.comparisons[]|"\(.tier): mHC earlier/tied/later \(.mhc_fewer_epochs)/\(.same_epochs)/\(.mhc_more_epochs); all-pairs epoch ratio \(.full_suite_geometric_mean_mhc_over_main)"),
  "", "Promotion decision: \(.promotion.selected_arm)",
  (.promotion.arms[]|"\(.arm): eligible \(.eligible); all-short ratio versus historical incumbent \(.full_short_geomean_vs_incumbent); lost historical passes \(.lost_historical_passes|tojson)"),
  "", "Long task first99 epochs (all fixed seeds, no pooled history):",
  (.long_epochs[]|"\(.task), seed\(.seed): main \(.main_status) \(.main_first99_epochs); mHC \(.mhc_status) \(.mhc_first99_epochs); observation interval \(.observation_interval_updates) updates"),
  "", .limitations[]' "$results/comparison.json" > "$results/comparison.txt"

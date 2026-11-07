#!/usr/bin/env bash
set -euo pipefail
centering_analysis=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
centering_repo=$(cd -- "$centering_analysis/../../.." && pwd)
centering_control="$centering_repo/runs/complex-pow4-no-xsa-seed1-01"
centering_candidate="$centering_repo/runs/complex-pow4-centered-seed1-01"

jq -e '.complete and .passed and .stress_source_matches_training
  and .input_artifacts_unchanged and .subgroup_accounting
  and ([.all_retained_replays[].passed]|all)' "$centering_candidate/audit.json" >/dev/null
jq -n --slurpfile ordinary "$centering_control/history.jsonl" \
  --slurpfile centered "$centering_candidate/history.jsonl" \
  -f "$centering_analysis/pow4-centering-analysis.jq" \
  > "$centering_analysis/pow4-centering.json"
jq -e '.steps_identical' "$centering_analysis/pow4-centering.json" >/dev/null

centering_stress=$(jq -r '.stress_root' "$centering_candidate/protocol.json")
jq -n --slurpfile audit "$centering_candidate/audit.json" \
  --slurpfile protocol "$centering_candidate/protocol.json" \
  --slurpfile result "$centering_candidate/result.json" \
  --slurpfile stress "$centering_stress/summary.json" \
  --slurpfile analysis "$centering_analysis/pow4-centering.json" \
  --slurpfile lean "$centering_analysis/pow4-centering/validation.json" '
  {complete:true,passed:($audit[0].passed and $stress[0].passed and $analysis[0].steps_identical),
   training_updates:$result[0].completed,status:$result[0].status,
   first99_epochs:$result[0].first99_epochs,epochs:$result[0].epochs,
   matched_observations:$audit[0].paired_observations,
   common_final_step:$analysis[0].common_final_step,
   ordinary_unobserved_after_common_prefix:$analysis[0].ordinary_unobserved_after_common_prefix,
   retained_state_replays:($audit[0].all_retained_replays|length),
   input_artifacts_unchanged:$audit[0].input_artifacts_unchanged,
   subgroup_accounting:$audit[0].subgroup_accounting,
   stress_source_matches_training:$audit[0].stress_source_matches_training,
   source_sha256:$protocol[0].source.sha256,
   initial_parameters_sha256:$protocol[0].initial_parameters_sha256,
   sampler_sha256:$protocol[0].sampler_sha256,attention:$protocol[0].attention,
   centered_attention_ptx_sha256:$protocol[0].centered_attention_ptx_sha256,
   ordinary_attention_ptx_sha256:$protocol[0].ordinary_attention_ptx_sha256,
   preflight:{passed:$stress[0].passed,model_cases:($stress[0].rows|length),
     ordinary_regression_cases:($stress[0].ordinary_regression|length),
     parameter_directions:([$stress[0].rows[],$stress[0].ordinary_regression[]]
       |map(.directional_checks|length)|add),
     primitive_cases:($stress[0].primitives|length),
     primitive_gradient_coordinates:($stress[0].primitives|map(.checked_gradient_coordinates)|add),
     canceled_equal_token_paths:$stress[0].canceled_equal_token_paths},
   lean:$lean[0],python_invoked:false,benchmark_modified:false,
   limitation:"Single known-miss seed; compare common observations only; first99 is not sustained convergence"}' \
  > "$centering_analysis/pow4-centering-validation.json"

for centering_variant in ordinary centered; do
  if [[ "$centering_variant" == ordinary ]]; then
    centering_run="$centering_control"
  else
    centering_run="$centering_candidate"
  fi
  jq -r '[.step,.epochs,100*.validation.sequence_accuracy,.train.loss,.validation.loss,
    100*.train.diagonal.sequence_accuracy,100*.validation.diagonal.sequence_accuracy] | @tsv' \
    "$centering_run/history.jsonl" > "$centering_analysis/pow4-centering-$centering_variant.tsv"
done

cd -- "$centering_analysis"
gnuplot pow4-centering.gnuplot
gnuplot -e 'render_png=1' pow4-centering.gnuplot

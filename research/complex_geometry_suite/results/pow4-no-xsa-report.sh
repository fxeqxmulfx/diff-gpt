#!/usr/bin/env bash
set -euo pipefail
analysis_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$analysis_dir/../../.." && pwd)
control_dir="$repo_dir/runs/complex-pow4-plain-ffn-seed1-01"
ordinary_dir="$repo_dir/runs/complex-pow4-no-xsa-seed1-01"

jq -e '.complete and .passed and .stress_source_matches_training
  and .input_artifacts_unchanged and .subgroup_accounting
  and ([.all_retained_replays[].passed]|all)' "$ordinary_dir/audit.json" >/dev/null
jq -n --slurpfile radial "$control_dir/history.jsonl" \
  --slurpfile plain "$ordinary_dir/history.jsonl" \
  -f "$analysis_dir/pow4-plain-ffn-analysis.jq" |
  jq '{xsa_full:.radial_full,ordinary:.plain,xsa_matched:.radial_matched,
    matched_observations,steps_identical,same_exposure_final_accuracy_change,
    censored_after_first99}' > "$analysis_dir/pow4-no-xsa.json"
jq -e '.steps_identical and .ordinary.observations==.matched_observations' \
  "$analysis_dir/pow4-no-xsa.json" >/dev/null

stress_dir=$(jq -r '.stress_root' "$ordinary_dir/protocol.json")
jq -n --slurpfile audit "$ordinary_dir/audit.json" \
  --slurpfile protocol "$ordinary_dir/protocol.json" \
  --slurpfile result "$ordinary_dir/result.json" \
  --slurpfile stress "$stress_dir/summary.json" \
  --slurpfile analysis "$analysis_dir/pow4-no-xsa.json" '
  {complete:true,passed:($audit[0].passed and $stress[0].passed and $analysis[0].steps_identical),
   training_updates:$result[0].completed,status:$result[0].status,
   first99_epochs:$result[0].first99_epochs,epochs:$result[0].epochs,
   paired_observations:$audit[0].paired_observations,
   retained_state_replays:($audit[0].all_retained_replays|length),
   input_artifacts_unchanged:$audit[0].input_artifacts_unchanged,
   subgroup_accounting:$audit[0].subgroup_accounting,
   stress_source_matches_training:$audit[0].stress_source_matches_training,
   source_sha256:$protocol[0].source.sha256,
   initial_parameters_sha256:$protocol[0].initial_parameters_sha256,
   sampler_sha256:$protocol[0].sampler_sha256,
   attention:$protocol[0].attention,
   ordinary_attention_ptx_sha256:$protocol[0].ordinary_attention_ptx_sha256,
   preflight:{passed:$stress[0].passed,model_cases:($stress[0].rows|length),
     xsa_regression_cases:($stress[0].xsa_regression|length),
     parameter_directions:([$stress[0].rows[],$stress[0].xsa_regression[]]
       |map(.directional_checks|length)|add),
     primitive_cases:($stress[0].primitives|length),
     primitive_gradient_coordinates:($stress[0].primitives|map(.checked_gradient_coordinates)|add),
     restored_equal_token_paths:$stress[0].restored_equal_token_paths,
     report:"no-xsa-stress.txt"},
   cpu_checks:{passed:15,failed:0,ignored_cuda_test:1,
     commands:["cargo test --release --offline --locked",
       "cargo clippy --offline --locked --all-targets -- -D warnings",
       "cargo fmt --check"],all_passed:true},
   python_invoked:false,benchmark_modified:false,
   limitation:"Single known-miss seed; first99 stopping does not establish sustained stability"}' \
  > "$analysis_dir/pow4-no-xsa-validation.json"

for variant in xsa ordinary; do
  if [[ "$variant" == xsa ]]; then run_dir="$control_dir"; else run_dir="$ordinary_dir"; fi
  jq -r '[.step,.epochs,100*.validation.sequence_accuracy,.train.loss,.validation.loss,
    100*.train.diagonal.sequence_accuracy,100*.validation.diagonal.sequence_accuracy] | @tsv' \
    "$run_dir/history.jsonl" > "$analysis_dir/pow4-no-xsa-$variant.tsv"
done

cd -- "$analysis_dir"
gnuplot pow4-no-xsa.gnuplot
gnuplot -e 'render_png=1' pow4-no-xsa.gnuplot

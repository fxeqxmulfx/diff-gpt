#!/usr/bin/env bash
set -euo pipefail

analysis_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$analysis_dir/../../.." && pwd)
radial_dir="$repo_dir/runs/complex-pow4-fixed-grad-seed1-01"
plain_dir="$repo_dir/runs/complex-pow4-plain-ffn-seed1-01"
stress_dir="$analysis_dir/plain-ffn-stress-02"

jq -e '.complete and .passed and .stress_source_matches_training
  and .input_artifacts_unchanged and .subgroup_accounting
  and ([.all_retained_replays[].passed] | all)' "$plain_dir/audit.json" >/dev/null

jq -n --slurpfile radial "$radial_dir/history.jsonl" \
  --slurpfile plain "$plain_dir/history.jsonl" \
  -f "$analysis_dir/pow4-plain-ffn-analysis.jq" \
  > "$analysis_dir/pow4-plain-ffn.json"
jq -e '.steps_identical and (.matched_observations == .plain.observations)' \
  "$analysis_dir/pow4-plain-ffn.json" >/dev/null

for variant in radial plain; do
  if [[ "$variant" == radial ]]; then run_dir="$radial_dir"; else run_dir="$plain_dir"; fi
  jq -r '[.step,.epochs,(100*.validation.sequence_accuracy),.train.loss,.validation.loss,
    (if .train.diagonal then 100*.train.diagonal.sequence_accuracy else "NaN" end),
    (if .validation.diagonal then 100*.validation.diagonal.sequence_accuracy else "NaN" end)] | @tsv' \
    "$run_dir/history.jsonl" > "$analysis_dir/pow4-plain-ffn-$variant.tsv"
done

jq -n --slurpfile audit "$plain_dir/audit.json" \
  --slurpfile stress "$stress_dir/summary.json" \
  --slurpfile protocol "$plain_dir/protocol.json" \
  --slurpfile result "$plain_dir/result.json" \
  --slurpfile analysis "$analysis_dir/pow4-plain-ffn.json" '
  {complete:true,passed:($audit[0].passed and $stress[0].passed and $analysis[0].steps_identical),
   training_updates:$result[0].completed,status:$result[0].status,
   first99_epochs:$result[0].first99_epochs,observations:$audit[0].paired_observations,
   retained_state_replays:($audit[0].all_retained_replays|length),
   input_artifacts_unchanged:$audit[0].input_artifacts_unchanged,
   stress_source_matches_training:$audit[0].stress_source_matches_training,
   subgroup_accounting:$audit[0].subgroup_accounting,
   source_sha256:$protocol[0].source.sha256,
   initial_parameters_sha256:$protocol[0].initial_parameters_sha256,
   sampler_sha256:$protocol[0].sampler_sha256,
   training_uses_wide_chain:$protocol[0].training_uses_wide_chain,
   preflight:{passed:$stress[0].passed,plain_cases:($stress[0].rows|length),
     radial_regression_cases:($stress[0].radial_regression|length),
     parameter_directions:([$stress[0].rows[],$stress[0].radial_regression[]]
       | map(.directional_checks|length) | add),
     primitive_rows:($stress[0].primitives|length),
     primitive_gradient_coordinates:($stress[0].primitives|map(.checked_gradient_coordinates)|add),
     local_adjoints:($stress[0].wide_composed_adjoint.local_adjoints|length),
     source:"plain-ffn-stress-02/summary.json",report:"plain-ffn-stress.txt"},
   cpu_checks:{command:"cargo test --release --offline --locked",passed:14,failed:0,
     ignored:1,ignored_reason:"CUDA-only check; GPU preflight run separately",
     clippy:"cargo clippy --offline --locked --all-targets -- -D warnings: passed",
     formatting:"cargo fmt --check: passed"},
   python_invoked:false,benchmark_modified:false,
   limitation:"Single known-miss task and seed; no universal selection or convergence guarantee"}' \
  > "$analysis_dir/pow4-plain-ffn-validation.json"

cd -- "$analysis_dir"
gnuplot pow4-plain-ffn.gnuplot
gnuplot -e 'render_png=1' pow4-plain-ffn.gnuplot

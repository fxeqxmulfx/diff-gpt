#!/usr/bin/env bash
# Compare independently completed runs; never pool earlier passes into coverage.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
out=research/main_model/results/gpu-suite-01/hard-reproduction
parent=runs/main-model-tensor-hard-01
candidate=runs/main-model-gpu-suite-01
test ! -e "$out"
jq -e '.audit_passed and .complete and .solved_cells==105' research/main_model/results/gpu-suite-01/final-audit.json > /dev/null
jq -e '.audit_passed and .solved_cells==4' research/main_model/results/tensor-model-02/hard-audit.json > /dev/null
mkdir -p "$out"
while read -r task seed; do
    old="$parent/mhc/$task/seed$seed"
    new="$candidate/mhc/$task/seed$seed"
    cmp "$old/initial.f32" "$new/initial.f32"
    cmp "$old/checkpoint.bin" "$new/checkpoint.bin"
    jq -cs 'map(del(.seconds))' "$old/history.jsonl" > "$out/$task-seed$seed-parent-history.json"
    jq -cs 'map(del(.seconds))' "$new/history.jsonl" > "$out/$task-seed$seed-new-history.json"
    cmp "$out/$task-seed$seed-parent-history.json" "$out/$task-seed$seed-new-history.json"
    jq -e -s '.[0] as $old|.[1] as $new|
      if $old.status!="passed" or $new.status!="passed" or $old.first99_epochs!=$new.first99_epochs
        or $old.completed!=$new.completed or $old.metadata.sampler_sha256!=$new.metadata.sampler_sha256
      then error("hard-cell reproduction changed")
      else {task:$new.case,seed:$new.seed,first99_epochs:$new.first99_epochs,completed:$new.completed,
        initial_bits_equal:true,checkpoint_bits_equal:true,all_observation_metrics_equal:true,
        timing_fields_excluded:true,sampler_sha256:$new.metadata.sampler_sha256} end' \
      "$old/result.json" "$new/result.json" > "$out/$task-seed$seed.json"
done <<'CELLS'
basis-depth-hard 0
arithmetic-sum-pow4 1
arithmetic-sum-pow4 0
basis-parity24 0
CELLS
jq -s '{passed:true,cells:.,scope:"Four newly trained full-suite cells reproduce the numerical parent exactly; no passes are pooled."}' \
  "$out/basis-depth-hard-seed0.json" "$out/arithmetic-sum-pow4-seed1.json" \
  "$out/arithmetic-sum-pow4-seed0.json" "$out/basis-parity24-seed0.json" > "$out/report.json"
rg --files "$out" | rg -v '/evidence.sha256$' | LC_ALL=C sort | xargs -d '\n' sha256sum > "$out/evidence.sha256"

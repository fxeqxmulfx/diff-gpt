#!/usr/bin/env bash
set -euo pipefail

# Compile only the mathematical certificates. No Rust, Python, or training.
centering_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
centering_repo=$(git -C "$centering_dir" rev-parse --show-toplevel)
centering_lean=${1:-/home/misha/lean_projects/transformer}
centering_fixture="$centering_repo/research/early99/guarantees/CausalCentering.lean"
centering_build=$(mktemp -d /tmp/pow4-centering.XXXXXX)
trap 'rm -rf "$centering_build"' EXIT

cd "$centering_lean"
git rev-parse HEAD > "$centering_dir/lean-commit.txt"
LAKE_ARTIFACT_CACHE=false lake env lean --version > "$centering_dir/lean-version.txt"
LAKE_ARTIFACT_CACHE=false lake env bash -s -- \
  "$centering_build" "$centering_fixture" "$centering_dir" <<'LEAN'
set -euo pipefail
export LEAN_PATH="$1:$LEAN_PATH"
lean -R "$(dirname -- "$2")" -o "$1/CausalCentering.olean" "$2" \
  > "$3/base-axioms.txt" 2>&1
lean -R "$3" -o "$1/CenteringCheck.olean" "$3/CenteringCheck.lean" \
  > "$3/check-axioms.txt" 2>&1
LEAN

if rg -n 'warning:|error:|sorryAx' "$centering_dir/base-axioms.txt" \
    "$centering_dir/check-axioms.txt"; then
  exit 1
fi
if rg -n '\b(sorry|axiom|native_decide|admit)\b' "$centering_fixture" \
    "$centering_dir/CenteringCheck.lean"; then
  exit 1
fi
awk '
  /depends on axioms:/ {
    axioms = $0
    sub(/^.*\[/, "", axioms)
    sub(/\].*$/, "", axioms)
    gsub(/propext|Classical\.choice|Quot\.sound|[, ]/, "", axioms)
    if (length(axioms) != 0) invalid = 1
    count++
  }
  END { if (invalid || count != 18) exit 1 }
' "$centering_dir/base-axioms.txt" "$centering_dir/check-axioms.txt"

sha256sum "$centering_fixture" "$centering_dir/CenteringCheck.lean" \
  "$centering_dir/check.sh" "$centering_lean/lean-toolchain" \
  "$centering_lean/lake-manifest.json" \
  "$centering_lean/src/Transformer/GPTMini/RMSNorm.lean" \
  "$centering_dir/base-axioms.txt" "$centering_dir/check-axioms.txt" \
  > "$centering_dir/sha256.txt"

jq -n --arg lean_commit "$(cat "$centering_dir/lean-commit.txt")" \
  --arg lean_version "$(cat "$centering_dir/lean-version.txt")" \
  '{passed:true,lean_commit:$lean_commit,lean_version:$lean_version,
    existing_theorems_checked:5,new_theorems_checked:13,
    warnings:0,extra_axioms:[],allowed_axioms:["propext","Classical.choice","Quot.sound"],
    checked_scope:"Exact causal-centering algebra, shared-parameter derivatives, and a restricted decoder impossibility result",
    full_model_impossibility_proved:false,training_convergence_proved:false,
    paper_random_matrix_theorems_formalized:false,
    rust_checks_run:false,training_run:false,python_used:false}' \
  > "$centering_dir/validation.json"
cat "$centering_dir/validation.json"

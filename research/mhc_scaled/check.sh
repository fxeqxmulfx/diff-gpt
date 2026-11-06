#!/usr/bin/env bash
# Check against the existing Lean project, writing build products only to /tmp.
set -euo pipefail

mhc_workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mhc_lean_project=${1:-/home/misha/lean_projects/transformer}
mhc_build=$(mktemp -d /tmp/mhc-scaled-check.XXXXXX)
trap 'rm -rf -- "$mhc_build"' EXIT

cd -- "$mhc_lean_project"
lake env bash -s -- "$mhc_workspace" "$mhc_build" <<'LEAN_CHECK'
set -euo pipefail
mhc_workspace=$1
mhc_build=$2
export LEAN_PATH="$mhc_build:$LEAN_PATH"
for mhc_source in \
  research/early99/guarantees/NormalizedScores.lean \
  research/normalization_guarantees/RMSLipschitz.lean \
  research/normalization_guarantees/AttentionSensitivity.lean \
  research/normalization_guarantees/SoftmaxLipschitz.lean \
  research/normalization_guarantees/AttentionSensitivityImproved.lean \
  research/normalization_guarantees/AttentionSelectivity.lean \
  research/early99_dimension_gain/Guarantees.lean \
  research/mhc_scaled/Bounds.lean \
  research/mhc_scaled/CurrentModel.lean \
  research/mhc_scaled/Checks.lean
do
  mkdir -p -- "$mhc_build/$(dirname -- "$mhc_source")"
  lean -DautoImplicit=false -DrelaxedAutoImplicit=false \
    -R "$mhc_workspace" -o "$mhc_build/${mhc_source%.lean}.olean" \
    "$mhc_workspace/$mhc_source"
done
LEAN_CHECK

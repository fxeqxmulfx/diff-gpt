#!/usr/bin/env bash
# Canonical selected-model dispatch; frozen historical benchmark labels stay fixed.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
manifest=research/main_model/model.json
if [[ $# == 0 || ( $# == 1 && $1 == info ) ]]; then
    jq '{id,name,architecture,implementation,research_default_optimizer,validation_scope,latest_promotion_decision}' "$manifest"
    exit 0
fi
if [[ $# != 5 || $1 != cell ]]; then
    echo 'usage: current.sh [info] | cell FRESH_ROOT CASE SEED short|long' >&2
    exit 2
fi
if [[ -e $2 ]]; then
    echo 'Refusing to alter an existing run; choose a fresh output root.' >&2
    exit 2
fi
arm=$(jq -er '.implementation.benchmark_arm' "$manifest")
binary=$(jq -er '.implementation.frozen_executable' "$manifest")
checksums=$(jq -er '.implementation.artifact_checksums' "$manifest")
[[ "$arm" == phase-score-lerope && -x "$binary" && -f "$checksums" ]]
sha256sum --check --status "$checksums"
exec "$binary" cell "$2" "$3" "$4" "$arm" "$5"

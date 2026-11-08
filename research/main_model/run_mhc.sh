#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
cohort=runs/main-model-mhc-02
binary="$cohort/main-transformer"
jq -e '.passed == true' research/main_model/results/mhc-stress-05/validation.json > /dev/null
sha256sum --check research/main_model/results/mhc-binary.sha256
"$binary" suite "$cohort" short
"$binary" suite "$cohort" long
"$binary" report "$cohort"

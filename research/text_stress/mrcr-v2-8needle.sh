#!/usr/bin/env bash
# Run the paired length regimes with the shared native MRCR evaluator.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo 'usage: mrcr-v2-8needle.sh {verify|prompts|score} SPLIT DATA_DIR [PREDICTIONS.jsonl MODEL {tools|no-tools}]' >&2
    exit 2
fi
op=$1
split=$2
data_dir=$3
shift 3
case "$op:$#" in verify:0|prompts:0|score:3) ;; *) echo 'invalid operation or arguments' >&2; exit 2 ;; esac
harness=${TRANSFORMER_ROOT:?point TRANSFORMER_ROOT to the transformer repository}/experiments/mrcr_v2
"$harness/fetch.sh" "$split" "$data_dir"
exec cargo +1.99.1 run --quiet --release --locked --manifest-path "$harness/Cargo.toml" \
    -- "$op" "$split" "$data_dir/$split.csv" "$@"

#!/usr/bin/env bash
set -euo pipefail
proof_dir=$(cd "$(dirname "$0")" && pwd)
cd /home/misha/lean_projects/transformer
LAKE_ARTIFACT_CACHE=false lake build Transformer.MHC.PeriRMS
LAKE_ARTIFACT_CACHE=false lake env lean scripts/MHCPeriRMSAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean "$proof_dir/MHCNormalization.lean"

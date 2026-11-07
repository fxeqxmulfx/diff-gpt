Completed quartic AdamW/MuonMD baseline: results snapshot

All six declared cells completed 100000 updates. audit.json records the native
audit and every result; summary.tsv gives compact per-cell metrics. Accuracy
values are fractions. All first99_epoch fields are blank because no observation
reached 99% held-out accuracy. All final training accuracies are 100%.

protocol.json and source-commit.txt identify the frozen study at 31d0a10.
artifacts.sha256 is the original manifest relative to runs/quartic-muonmd;
it verifies the binaries, queue script, source commit and protocol. SHA256SUMS
instead verifies this directory's compact result files. gpu-validation.txt is
the original pre-training log: seven native tests passed. queue-state.json
records successful completion. The results-only commit reran the frozen CPU
audit and artifact checks; it did not repeat GPU tests or restart training.

Full initial weights, histories and states remain in runs/quartic-muonmd.
The later schedule study reuses these exact six cells. Its combined audit and
checkpoint-manifests.json include their identities and checkpoint hashes.

Verification from the repository root with runtime archives available:
  (cd runs/quartic-muonmd && sha256sum --check artifacts.sha256)
  runs/quartic-muonmd/bin/quartic-muonmd audit runs/quartic-muonmd
  jq -e '.completed == 6 and (.integrity_failures | length == 0)' runs/quartic-muonmd/audit.json
  (cd research/quartic_muonmd/results && sha256sum --check SHA256SUMS)

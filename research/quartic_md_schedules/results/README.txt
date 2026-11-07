Stopped quartic MD schedule study: results snapshot

The user stopped the benchmark on 2026-11-07 at 22:11:31 UTC. Training and the
queue exited cleanly. This snapshot preserves 46 complete 100000-update cells,
one stopped cell and seven not-started cells out of the 54 declared cells.
It does not claim the planned factorial finished. The runtime STOP flag remains.

combined-audit.json is the frozen native auditor's output for all 54 cells.
audit.json covers the 48 newly scheduled cells: 40 complete, one stopped and
seven pending_or_running. The latter label is emitted for missing result files;
all seven directories were verified absent and stop.json identifies those cells
as not started. No incomplete endpoint is counted as a full-budget result.

summary.tsv contains all 54 outcomes, with accuracy as a fraction, not percent.
endpoint_accuracy is explicitly an endpoint: the stopped cell's value comes
from 19057 updates, while complete endpoints come from 100000 updates. Blank
metrics identify unstarted cells. first99_epoch and peak_accuracy summarize
the observed trajectory, not an independent test after checkpoint selection.

arm-summary.tsv has one row per declared configuration. Seed accuracy columns
contain only complete endpoints. A three_seed_mean_final_accuracy is present
only when all three seeds completed; incomplete configurations have a blank
mean. There are ten such complete configurations in this snapshot. All cells
use the same previously examined 4705-example held-out set, so selecting the
best of many cells or observations is not a fresh generalization test.

stop.json records the user request, saved checkpoint hashes and the clean
execution marker. checkpoint-manifests.json captures the protocol, checkpoint
hashes and clean execution record of all 47 observed cells, including the six
baseline cells. Full initial weights, histories and optimizer states remain
under the two runs/ directories; they are not included in Git.

source-commit.txt identifies implementation b8ff997. protocol.json freezes the
model, source, PTX and data identities. artifacts.sha256 is the original
runtime manifest, relative to runs/quartic-md-schedules; it verifies binaries,
the launch script, source commit and protocol. SHA256SUMS instead covers the
compact files in this results directory. The original 12-test GPU validation
log is preserved as gpu-validation.txt. Those tests ran before training; this
results-only commit reran the CPU audits and hash checks without restarting
training or executing Python. The original six-cell audit was rerun first.

Verification from the repository root, with the runtime archives available:
  (cd runs/quartic-muonmd && sha256sum --check artifacts.sha256)
  (cd runs/quartic-md-schedules && sha256sum --check artifacts.sha256)
  runs/quartic-muonmd/bin/quartic-muonmd audit runs/quartic-muonmd
  runs/quartic-md-schedules/bin/quartic-md-schedules combined runs/quartic-md-schedules runs/quartic-muonmd
  jq -e '.completed == 46 and (.integrity_failures | length == 0)' runs/quartic-md-schedules/combined-audit.json
  (cd research/quartic_md_schedules/results && sha256sum --check SHA256SUMS)

The CPU audit validates saved state/history hashes, source/PTX identities,
initialization and sampler pairing, observation order, exact classification
counts, finite states and recomputed trajectory summaries. It preserves the
stopped status and does not remove STOP or resume training. Baseline results
and their seven-test validation record are in research/quartic_muonmd/results.

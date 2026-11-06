# Reproducing the round-two reports

This package audits the published early99 evidence from preserved raw histories
without changing training, the frozen selection, or the original report files.
It lives outside the recursively hashed scientific source tree.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.early99_reporting.round2_audit
```

Run from the repository root. The command writes a separate accounting record
to `runs/early99/round2-report-audit.json`; `--output` selects another location.
It verifies the fixed selection SHA, all declared search/candidate plans and
archived candidate sources, then reconstructs all 14 screen decisions and
26 full-short paired comparisons from the raw histories. It checks every
JSON/CSV row, censoring marker, per-task statistic, eligible geometric mean
and the frozen winner. It also checks that each candidate has a matching
recorded deep checkpoint audit and verifies the six checked Lean fixtures and
their recorded proof artifacts.

The [short-stage audit](results/short-stage-audit.json) passed with all eight
promoted candidates and all rejected screens retained. At that snapshot,
21/30 selected long cells were complete. Consequently
`round2_results_ready_for_final_writeup` is **false**. Partial long outcomes are
not analyzed. Once all 30 cells, the automatic 105-cell checkpoint audit and
the final long reports exist, rerun the command to verify both long comparisons,
their CSVs, all censoring/regression records and the full long-summary metrics.

This accounting check does not reload every checkpoint or recompile Lean. Those
are separate recorded checks, identified in its output. It does not assert that
final documentation, commits, or the user's requested subsequent research are
complete. Three focused tests reject missing CSV rows, concealed budget misses,
invalid screen promotion and filtered/incomplete geometric means. The
[validation record](results/validation.json) also checks the long summarizers
against the already completed round-one report.

The [archive manifest](archive/manifest.json) preserves five orchestration and
report helpers previously held only under `/tmp`, with byte hashes. Their
historical paths and selection-time guards remain verbatim. The read-only
module above is the entry point for checking the frozen reports today.

For a fresh deep checkpoint audit of the selected candidate after all 105 cells
finish, run the archived auditor with its matching scientific source checkout:

```bash
cd /tmp/early99-attention-replay-7cff
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /home/misha/python_projects/gpt/.venv/bin/python /home/misha/python_projects/gpt/research/early99_reporting/archive/checkpoint-audit.py.txt --name peri-rms-scaled-ssmax-noqk-lr2 --full --long --output /tmp/early99-winner-checkpoint-audit.json
```

That checkout is pinned to `7cff292`; the run's archived source hashes are
verified against it. The current live driver already schedules this deep audit,
so the command is provided for later reproduction. Raw data and checkpoints
remain in the original `runs/early99` directories.

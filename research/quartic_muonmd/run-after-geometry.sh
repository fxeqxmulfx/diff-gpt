#!/usr/bin/env bash
# Run frozen native artifacts after the existing GPU cohort passes its final audit.
set -euo pipefail
cd /home/misha/python_projects/gpt
root="$PWD/runs/quartic-muonmd"
prior="$PWD/runs/complex-geometry-suite"
results="$PWD/research/quartic_muonmd/results"
predecessor_pid="${1:?predecessor PID required}"
predecessor_start="${2:?predecessor start ticks required}"
[[ "$predecessor_pid" =~ ^[0-9]+$ && "$predecessor_start" =~ ^[0-9]+$ ]]
exec 9>"$root/queue.lock"
flock -n 9
exec > >(tee -a "$root/cohort.log") 2>&1
phase=starting
state() {
    phase="$1"
    jq -n --arg phase "$phase" --arg time "$(date -u +%FT%TZ)" \
        --argjson pid "$$" --argjson predecessor_pid "$predecessor_pid" \
        --arg predecessor_start "$predecessor_start" \
        '{phase:$phase,updated_at_utc:$time,queue_pid:$pid,
          predecessor_pid:$predecessor_pid,predecessor_start_ticks:$predecessor_start,
          expected_cells:6,updates_per_cell:100000,checkpoint_epochs:500,python_invoked:false}' \
        > "$root/queue-state.pending"
    mv "$root/queue-state.pending" "$root/queue-state.json"
    printf '%s: %s\n' "$(date -u +%FT%TZ)" "$phase"
}
on_exit() {
    local code=$?
    if (( code != 0 )); then state "failed during $phase (exit $code)"; fi
}
trap on_exit EXIT
stop_requested() {
    if [[ -f "$root/STOP" ]]; then state stopped; exit 0; fi
}
same_predecessor() {
    local stat_line rest
    local -a fields
    [[ -r "/proc/$predecessor_pid/stat" ]] || return 1
    IFS= read -r stat_line < "/proc/$predecessor_pid/stat" || return 1
    rest="${stat_line##*) }"
    read -r -a fields <<< "$rest"
    [[ "${fields[19]}" == "$predecessor_start" && "${fields[0]}" != Z ]]
}
state waiting_for_geometry
while same_predecessor; do stop_requested; sleep 30; done
stop_requested
state checking_predecessor
jq -e '.phase == "complete"' "$prior/queue-state.json" > /dev/null
jq -e '.complete == true and .expected_cells == 420 and (.audit_failures | length == 0)' \
    "$PWD/research/complex_geometry_suite/results/audit.json" > /dev/null
# Hold the predecessor lock throughout this cohort so its dispatcher cannot restart.
exec 8>"$prior/queue.lock"
flock -n 8
if pgrep -f '^/home/misha/python_projects/gpt/runs/(complex-geometry-suite|complex-ffn-suite)/bin/.* (cell|suite) ' > /dev/null; then
    printf '%s\n' 'Existing cohort still owns the GPU; refusing concurrent training.' >&2
    exit 1
fi
(
    cd "$root"
    sha256sum --check artifacts.sha256
)
state validating_gpu
"$root/bin/quartic-muonmd-tests" --test-threads=1 > "$root/gpu-validation.log" 2>&1
stop_requested
state training
"$root/bin/quartic-muonmd" suite "$root"
stop_requested
state auditing
"$root/bin/quartic-muonmd" audit "$root" > "$root/final-audit.log"
jq -e '.completed == 6 and (.integrity_failures | length == 0)' "$root/audit.json" > /dev/null
mkdir -p "$results"
cp "$root/audit.json" "$root/protocol.json" "$root/artifacts.sha256" "$root/gpu-validation.log" "$results/"
jq -r '(["arm","seed","first99_epoch","best_accuracy","final_accuracy","final_loss","late_min_accuracy","late_max_accuracy"] | @tsv),
    (.results[] | [.arm,.seed,.first99.epochs,.best.validation.accuracy,.endpoint.validation.accuracy,.endpoint.validation.loss,.late_min_accuracy,.late_max_accuracy] | @tsv)' \
    "$root/audit.json" > "$results/summary.tsv"
state complete

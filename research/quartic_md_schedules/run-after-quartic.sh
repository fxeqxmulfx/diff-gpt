#!/usr/bin/env bash
# Frozen native experiment extension, serialized after the original six quartic cells.
set -euo pipefail
cd /home/misha/python_projects/gpt
root="$PWD/runs/quartic-md-schedules"
prior="$PWD/runs/quartic-muonmd"
geometry="$PWD/runs/complex-geometry-suite"
results="$PWD/research/quartic_md_schedules/results"
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
          expected_new_cells:48,expected_combined_cells:54,updates_per_cell:100000,
          checkpoint_epochs:500,python_invoked:false}' > "$root/queue-state.pending"
    mv "$root/queue-state.pending" "$root/queue-state.json"
    printf '%s: %s\n' "$(date -u +%FT%TZ)" "$phase"
}
on_exit() { local code=$?; if (( code != 0 )); then state "failed during $phase (exit $code)"; fi; }
trap on_exit EXIT
stop_requested() { if [[ -f "$root/STOP" ]]; then state stopped; exit 0; fi; }
same_predecessor() {
    local stat_line rest
    local -a fields
    [[ -r "/proc/$predecessor_pid/stat" ]] || return 1
    IFS= read -r stat_line < "/proc/$predecessor_pid/stat" || return 1
    rest="${stat_line##*) }"; read -r -a fields <<< "$rest"
    [[ "${fields[19]}" == "$predecessor_start" && "${fields[0]}" != Z ]]
}
state waiting_for_quartic
while same_predecessor; do stop_requested; sleep 30; done
stop_requested
state checking_predecessor
jq -e '.phase == "complete"' "$prior/queue-state.json" > /dev/null
exec 8>"$prior/queue.lock"; flock -n 8
exec 7>"$geometry/queue.lock"; flock -n 7
(
    cd "$prior"
    sha256sum --check artifacts.sha256
)
"$prior/bin/quartic-muonmd" audit "$prior" > "$root/baseline-recheck.json"
jq -e '.completed == 6 and (.integrity_failures | length == 0)' "$prior/audit.json" > /dev/null
(
    cd "$root"
    sha256sum --check artifacts.sha256
)
state validating_gpu
"$root/bin/quartic-md-schedules-tests" --test-threads=1 > "$root/gpu-validation.log" 2>&1
stop_requested
state training
training_status=0
"$root/bin/quartic-md-schedules" suite "$root" || training_status=$?
stop_requested
state auditing
"$root/bin/quartic-md-schedules" combined "$root" "$prior" > "$root/final-audit.log"
mkdir -p "$results"
cp "$root/combined-audit.json" "$root/audit.json" "$root/protocol.json" "$root/artifacts.sha256" "$results/"
cp "$root/gpu-validation.log" "$results/gpu-validation.txt"
jq -r '(["optimizer","schedule","warmup","seed","status","first99_epoch","final_accuracy","final_loss","late_min_accuracy","late_max_accuracy","reused_baseline"] | @tsv),
    (.results[] | [.optimizer,.schedule,.warmup,.seed,.status,.first99.epochs,.endpoint.validation.accuracy,.endpoint.validation.loss,.late_min_accuracy,.late_max_accuracy,.reused_baseline] | @tsv)' \
    "$root/combined-audit.json" > "$results/summary.tsv"
if (( training_status != 0 )); then exit "$training_status"; fi
jq -e '.completed == 54 and (.integrity_failures | length == 0)' "$root/combined-audit.json" > /dev/null
state complete

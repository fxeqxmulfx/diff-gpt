#!/usr/bin/env bash
# Frozen executables; no rebuild, Python, concurrent GPU work, or automatic replay.
set -euo pipefail
cd /home/misha/python_projects/gpt
root="$PWD/runs/complex-geometry-suite"
prior="$PWD/runs/complex-ffn-suite"
results="$PWD/research/complex_geometry_suite/results"
predecessor_pid="${1:?predecessor queue PID required}"
predecessor_start="${2:?predecessor /proc start ticks required}"
[[ "$predecessor_pid" =~ ^[0-9]+$ && "$predecessor_start" =~ ^[0-9]+$ ]]
exec 9>"$root/queue.lock"
flock -n 9
exec > >(tee -a "$root/queue.log") 2>&1

phase=starting
state() {
    phase="$1"
    jq -n --arg phase "$phase" --arg time "$(date -u +%FT%TZ)" \
        --argjson pid "$$" --argjson predecessor_pid "$predecessor_pid" \
        --arg predecessor_start "$predecessor_start" \
        '{phase:$phase,updated_at_utc:$time,queue_pid:$pid,
          predecessor_pid:$predecessor_pid,predecessor_start_ticks:$predecessor_start,
          expected_cells:420,checkpoint_epochs:500,python_invoked:false}' \
        > "$root/queue-state.pending"
    mv "$root/queue-state.pending" "$root/queue-state.json"
    printf '%s: %s\n' "$(date -u +%FT%TZ)" "$phase"
}
on_exit() {
    local code=$?
    if (( code != 0 )); then
        state "failed during $phase (exit $code)"
    fi
}
trap on_exit EXIT
stop_requested() {
    if [[ -f "$root/STOP" ]]; then
        state stopped
        exit 0
    fi
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
state waiting_for_ffn
while same_predecessor; do
    stop_requested
    sleep 60
done
stop_requested

# The previous dispatcher runs its independent audit before exiting.
state checking_predecessor
jq -e '.complete == true and .expected_cells == 210 and
       (.audit_failures | length == 0) and (.rows | length == 210) and
       all(.rows[]; .audit == "passed" and
           (.status == "passed" or .status == "budget_exhausted"))' \
    "$PWD/research/complex_ffn_suite/results/audit.json" > /dev/null
(
    cd "$root"
    sha256sum --check binaries.sha256
)

# Revalidate the exact frozen test executable once the GPU is free.
state validating_gpu
"$root/bin/complex-geometry-tests" --ignored --test-threads=1 > "$root/gpu-validation.log" 2>&1
stop_requested

state short
"$root/bin/complex-geometry" suite "$root" short
stop_requested
"$root/bin/complex-geometry-audit" "$root" "$results/audit.json"
jq -e '(.audit_failures | length == 0) and
       ([.rows[] | select(.tier == "short")] | length == 300 and
        all(.[]; .audit == "passed" and
            (.status == "passed" or .status == "budget_exhausted")))' \
    "$results/audit.json" > /dev/null
state long
"$root/bin/complex-geometry" suite "$root" long
stop_requested
state auditing
"$root/bin/complex-geometry-audit" "$root" "$results/audit.json"
jq -e '.complete == true and .expected_cells == 420 and
       (.audit_failures | length == 0)' "$results/audit.json" > /dev/null
state complete

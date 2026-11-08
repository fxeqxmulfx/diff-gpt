include "coverage_gate";

# Old results remain separate evidence. Only their verified task/seed passes
# impose requirements; no historical cell can fill a fresh-cohort result.
def historical_gate($baseline; $arm; $short_count; $all_count):
  . as $fresh |
  [.rows[] | select(.arm == $arm)] as $rows |
  [$baseline.passes[] as $old |
    ([$rows[] | select(.case == $old.case and .seed == $old.seed)] | first) as $new |
    {tier:$old.tier, task:$old.case, seed:$old.seed,
     old_epochs:$old.first99_epochs, new_epochs:$new.first99_epochs,
     status:($new.status // "missing"), audit:($new.audit // "missing")}] as $required |
  [$required[] | select(.status == "budget_exhausted" and .audit == "passed") |
    {tier,task,seed}] as $lost |
  [$required[] | select(.status != "passed" and .status != "budget_exhausted") |
    {tier,task,seed,status}] as $pending |
  [$required[] | select(.tier == "short" and .status == "passed" and .audit == "passed"
    and .old_epochs > 0 and .new_epochs > 0) | .new_epochs/.old_epochs] as $ratios |
  (.complete == true and .source_verified == true and (.audit_failures|length) == 0
   and $baseline.verified == true and ($rows|length) == $all_count
   and all($rows[]; .audit == "passed" and (.status == "passed" or .status == "budget_exhausted"))) as $complete |
  {
    arm:$arm, required_historical_passes:($required|length),
    lost_historical_passes:$lost, unresolved_historical_passes:$pending,
    coverage_satisfied:($complete and ($lost|length) == 0 and ($pending|length) == 0
      and all($required[];.audit == "passed")),
    full_short_geomean_vs_incumbent:(if ($ratios|length) == $short_count
      and ([$required[]|select(.tier=="short")]|length) == $short_count
      then ([$ratios[]|log]|add/$short_count|exp) else null end)
  };

def restart_decision($baseline; $short_count; $all_count):
  . as $audit |
  coverage_gate("main"; "mhc"; $all_count) as $paired |
  historical_gate($baseline; "main"; $short_count; $all_count) as $main |
  historical_gate($baseline; "mhc"; $short_count; $all_count) as $mhc |
  [$main,$mhc] | map(. + {
    eligible:(.coverage_satisfied and .full_short_geomean_vs_incumbent != null
      and .full_short_geomean_vs_incumbent < 1
      and (if .arm == "mhc" then $paired.coverage_requirement_satisfied else true end))
  }) as $arms |
  {
    historical_results_pooled_into_fresh_cohort:false,
    fresh_mhc_coverage_gate:$paired,
    arms:$arms,
    selected_arm:(if $audit.complete != true or $paired.status == "pending" or $paired.status == "blocked_by_invalid_evidence"
      then null
      else ([$arms[]|select(.eligible)]|sort_by(.full_short_geomean_vs_incumbent,.arm)|first|.arm) // "incumbent" end),
    rule:"Complete both fresh arms; preserve every verified historical pass and, for mHC, every fresh-main pass. Require fewer epochs over all75 short cells. Select the eligible arm with the lower short-suite geometric mean; keep incumbent on no improvement."
  };

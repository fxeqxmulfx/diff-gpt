# Input is the independently audited cohort, never the trainer's summary.
# A coverage pass is necessary for promotion; it does not establish an epoch gain.
def coverage_gate($control; $candidate; $expected_pairs):
  [.pairs[] | select(.control == $control and .candidate == $candidate)] as $pairs |
  [$pairs[] | select(.control_status == "passed" and .candidate_status == "budget_exhausted") |
    {tier, task:.case, seed}] as $lost |
  [.rows[] | select(.arm == $control)] as $main |
  [.rows[] | select(.status == "execution_error" or .status == "audit_failure") |
    {tier, task:.case, seed, arm, status}] as $invalid |
  (.complete == true and .source_verified == true and
   ($pairs|length) == $expected_pairs and ($main|length) == $expected_pairs and
   all(.rows[]; .audit == "passed" and (.status == "passed" or .status == "budget_exhausted"))) as $complete |
  {
    control:$control, candidate:$candidate, expected_pairs:$expected_pairs,
    status:(if ($lost|length)>0 then "rejected_lost_main_pass"
      elif (.audit_failures|length)>0 or ($invalid|length)>0 then "blocked_by_invalid_evidence"
      elif $complete then "passed_no_lost_main_pass" else "pending" end),
    coverage_requirement_satisfied:($complete and ($lost|length)==0 and (.audit_failures|length)==0 and ($invalid|length)==0),
    lost_main_passes:$lost,
    invalid_cells:$invalid,
    main_passes:([$main[]|select(.status=="passed" and .audit=="passed")]|length),
    main_budget_misses:([$main[]|select(.status=="budget_exhausted" and .audit=="passed")]|length),
    main_all_expected_passed:(if any($main[]; .status=="budget_exhausted" and .audit=="passed") then false
      elif ($main|length)==$expected_pairs and all($main[]; .status=="passed" and .audit=="passed") then true
      else null end),
    rule:"A single verified lost control task/seed pass rejects the candidate; aggregate epoch gains cannot compensate. A coverage pass alone does not establish an epoch improvement."
  };

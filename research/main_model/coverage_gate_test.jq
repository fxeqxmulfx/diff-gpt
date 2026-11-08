include "coverage_gate";
def baseline:
  [{tier:"short",case:"task-a",seed:0}, {tier:"short",case:"task-a",seed:1},
   {tier:"short",case:"task-b",seed:0}, {tier:"long",case:"task-c",seed:0}] as $cells |
  {complete:true,source_verified:true,audit_failures:[],
   rows:[$cells[] as $cell | ["main","mhc"][] as $arm |
     $cell+{arm:$arm,status:"passed",audit:"passed"}],
   pairs:[$cells[] | .+{control:"main",candidate:"mhc",control_status:"passed",
     candidate_status:"passed",candidate_over_control_epochs:0.1}]};
def miss($arm; $task; $seed):
  .rows |= map(if .arm==$arm and .case==$task and .seed==$seed then .status="budget_exhausted" else . end) |
  .pairs |= map(if .case==$task and .seed==$seed then
    (if $arm=="main" then .control_status="budget_exhausted" else .candidate_status="budget_exhausted" end) |
    .candidate_over_control_epochs=null else . end);
[
  {name:"complete_coverage", input:baseline, status:"passed_no_lost_main_pass", eligible:true, main_all:true, lost:0},
  {name:"one_lost_seed_despite_90_percent_epoch_gain", input:(baseline|miss("mhc";"task-a";1)), status:"rejected_lost_main_pass", eligible:false, main_all:true, lost:1},
  {name:"one_lost_long_pass", input:(baseline|miss("mhc";"task-c";0)), status:"rejected_lost_main_pass", eligible:false, main_all:true, lost:1},
  {name:"confirmed_regression_rejects_before_cohort_finishes", input:(baseline|miss("mhc";"task-a";1)|.complete=false), status:"rejected_lost_main_pass", eligible:false, main_all:true, lost:1},
  {name:"matching_budget_misses_are_not_lost_passes", input:(baseline|miss("main";"task-b";0)|miss("mhc";"task-b";0)), status:"passed_no_lost_main_pass", eligible:true, main_all:false, lost:0},
  {name:"incomplete_is_not_success", input:(baseline|.complete=false), status:"pending", eligible:false, main_all:true, lost:0},
  {name:"unrun_candidate_is_not_a_budget_miss", input:(baseline|.complete=false|.rows|=map(if .arm=="mhc" and .case=="task-c" then .status="pending"|.audit="pending" else . end)|.pairs|=map(select(.case!="task-c"))), status:"pending", eligible:false, main_all:true, lost:0},
  {name:"execution_error_requires_diagnosis", input:(baseline|.complete=false|.rows[1].status="execution_error"), status:"blocked_by_invalid_evidence", eligible:false, main_all:true, lost:0},
  {name:"failed_audit_blocks_acceptance", input:(baseline|.audit_failures=["checkpoint hash mismatch"]), status:"blocked_by_invalid_evidence", eligible:false, main_all:true, lost:0},
  {name:"missing_provenance_is_not_complete", input:(baseline|.source_verified=false), status:"pending", eligible:false, main_all:true, lost:0},
  {name:"missing_pair_is_not_complete", input:(baseline|.pairs|=.[0:3]), status:"pending", eligible:false, main_all:true, lost:0},
  {name:"main_coverage_is_unknown_until_measured", input:(baseline|.complete=false|.rows[0].status="pending"|.rows[0].audit="pending"|.pairs|=.[1:]), status:"pending", eligible:false, main_all:null, lost:0}
] |
map(. as $case | (.input|coverage_gate("main";"mhc";4)) as $actual |
  if $actual.status==$case.status and $actual.coverage_requirement_satisfied==$case.eligible and
     $actual.main_all_expected_passed==$case.main_all and ($actual.lost_main_passes|length)==$case.lost
  then {name:$case.name,passed:true}
  else error({test:$case.name,expected:$case,actual:$actual}|tojson) end) |
{passed:all(.[];.passed),test_count:length,tests:.}

include "restart_gate";
def cells: [{tier:"short",case:"a",seed:0}, {tier:"short",case:"a",seed:1},
  {tier:"short",case:"b",seed:0}, {tier:"long",case:"c",seed:0}];
def old: {verified:true,passes:[cells[]+{first99_epochs:100}]};
def fresh:
  {complete:true,source_verified:true,audit_failures:[],
   rows:[cells[] as $c | ["main","mhc"][] as $arm |
     $c+{arm:$arm,status:"passed",audit:"passed",first99_epochs:(if $arm=="main" then 80 else 60 end)}],
   pairs:[cells[]+{control:"main",candidate:"mhc",control_status:"passed",candidate_status:"passed",candidate_over_control_epochs:0.75}]};
def miss($arm;$task):
  .rows |= map(if .arm==$arm and .case==$task then .status="budget_exhausted"|.first99_epochs=null else . end) |
  .pairs |= map(if .case==$task then
    (if $arm=="main" then .control_status="budget_exhausted" else .candidate_status="budget_exhausted" end)
    | .candidate_over_control_epochs=null else . end);
[
  {name:"faster_mhc",input:fresh,want:"mhc"},
  {name:"mhc_loses_long_pass",input:(fresh|miss("mhc";"c")),want:"main"},
  {name:"both_lose_historical_long_pass",input:(fresh|miss("mhc";"c")|miss("main";"c")),want:"incumbent"},
  {name:"mhc_recovers_main_regression",input:(fresh|miss("main";"c")),want:"mhc"},
  {name:"incomplete_never_promotes",input:(fresh|.complete=false),want:null},
  {name:"early_loss_still_waits_for_full_comparison",input:(fresh|miss("mhc";"c")|.complete=false),want:null},
  {name:"fresh_main_additional_pass_is_required",input:(fresh|miss("mhc";"c")),want:"main",baseline:(old|.passes|=map(select(.case!="c")))},
  {name:"failed_audit_never_promotes",input:(fresh|.audit_failures=["bad hash"]),want:null},
  {name:"no_faster_arm",input:(fresh|.rows |= map(.first99_epochs=100)),want:"incumbent"},
  {name:"faster_main",input:(fresh|.rows |= map(if .arm=="main" then .first99_epochs=40 else . end)),want:"main"},
  {name:"one_lost_short_seed",input:(fresh|.rows|=map(if .arm=="mhc" and .case=="a" and .seed==1 then .status="budget_exhausted"|.first99_epochs=null else . end)|.pairs|=map(if .case=="a" and .seed==1 then .candidate_status="budget_exhausted"|.candidate_over_control_epochs=null else . end)),want:"main"},
  {name:"unverified_history_never_promotes",input:fresh,want:"incumbent",baseline:(old|.verified=false)}
] | map(. as $test | (.input|restart_decision(($test.baseline//old);3;4)) as $got |
  if $got.selected_arm==$test.want then {name:$test.name,passed:true}
  else error({test:$test.name,expected:$test.want,got:$got}|tojson) end) |
{passed:all(.[];.passed),test_count:length,tests:.}

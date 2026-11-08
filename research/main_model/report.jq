include "coverage_gate";
def median: sort | if length == 0 then null elif length % 2 == 1 then .[length/2|floor] else (.[length/2-1]+.[length/2])/2 end;
def mean: if length == 0 then null else add/length end;
. as $audit |
{
  complete: .complete,
  audit_failures: .audit_failures,
  source_sha256: .source_sha256,
  expected_cells: .expected_cells,
  totals: .totals,
  coverage_gate: coverage_gate("main"; "mhc"; 105),
  epoch_definition: "actual examples seen divided by the frozen number of training examples; tail batches counted exactly",
  primary_endpoint: "first observed held-out exact-sequence accuracy >=99%, then stop; no sustained-accuracy claim",
  comparisons: [ ["short","long"][] as $tier |
    [.pairs[]|select(.tier==$tier)] as $pairs |
    [$pairs[]|.candidate_over_control_epochs|select(. != null)] as $ratios |
    (if $tier=="short" then 75 else 30 end) as $expected |
    {tier:$tier,expected_pairs:$expected,paired_completed:($pairs|length),both_passed:($ratios|length),
     mhc_fewer_epochs:([$ratios[]|select(.<1)]|length),same_epochs:([$ratios[]|select(.==1)]|length),mhc_more_epochs:([$ratios[]|select(.>1)]|length),
     solved_pairs_geometric_mean_mhc_over_main:(if ($ratios|length)>0 then ([$ratios[]|log]|mean|exp) else null end),
     full_suite_geometric_mean_mhc_over_main:(if ($ratios|length)==$expected then ([$ratios[]|log]|mean|exp) else null end),
     aggregate_scope:(if ($ratios|length)==$expected then "all_declared_pairs_passed" else "solved_pairs_only_incomplete_or_censored" end),
     median_mhc_over_main:($ratios|median),
     mhc_only_passed:([$pairs[]|select(.candidate_status=="passed" and .control_status=="budget_exhausted")]|length),
     main_only_passed:([$pairs[]|select(.control_status=="passed" and .candidate_status=="budget_exhausted")]|length),
     both_budget_misses:([$pairs[]|select(.control_status=="budget_exhausted" and .candidate_status=="budget_exhausted")]|length)}],
  long_epochs: [ $audit.rows[]|select(.tier=="long" and .arm=="main") as $main |
    $audit.rows[]|select(.tier=="long" and .arm=="mhc" and .case==$main.case and .seed==$main.seed) as $mhc |
    {task:$main.case,seed:$main.seed,
     main_status:$main.status,main_first99_epochs:$main.first99_epochs,main_updates:$main.completed,main_final_epochs:$main.epochs,main_final_accuracy:$main.sequence_accuracy,
     mhc_status:$mhc.status,mhc_first99_epochs:$mhc.first99_epochs,mhc_updates:$mhc.completed,mhc_final_epochs:$mhc.epochs,mhc_final_accuracy:$mhc.sequence_accuracy,
     mhc_over_main_epochs:(if $main.status=="passed" and $mhc.status=="passed" and $main.first99_epochs>0 then $mhc.first99_epochs/$main.first99_epochs else null end)}],
  limitations:["A solved-pairs epoch mean is exploratory until every declared pair passes; it cannot compensate for a coverage regression.","Three frozen seeds per task; these tasks do not establish performance on larger unseen tasks.","mHC adds trainable routing parameters; this is not a matched-parameter-count comparison.","Finite Sinkhorn need not be exactly doubly stochastic; Lean forward bounds are exact-real and do not prove convergence.","Wall-clock results are secondary and not a controlled throughput benchmark."]
}

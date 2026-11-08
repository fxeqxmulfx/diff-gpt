# Input: independent phase_mhc_audit output; $control is the pinned main audit.
if .expected_cells != 105 or (.rows|length) != 105 then error("unexpected audit scope") else . end
| . as $audit
| [$audit.rows[] as $r
    | [$control[0].rows[] | select(.arm=="lerope" and .case==$r.case and .seed==$r.seed)] as $matches
    | if ($matches|length)!=1 or $matches[0].status!="passed" or $matches[0].audit!="passed"
      then error("missing or ambiguous verified control") else $matches[0] end as $base
    | {tier:$r.tier,case:$r.case,seed:$r.seed,status:$r.status,audit:$r.audit,
       main_epochs:$base.first99_epochs,candidate_epochs:$r.first99_epochs,
       candidate_over_main_epochs:(if $r.audit=="passed" and $r.status=="passed"
         then $r.first99_epochs/$base.first99_epochs else null end),
       main_updates:$base.completed,candidate_updates:$r.completed,
       previous_observed_epochs:$r.previous_observed_epochs,
       heldout_accuracy:$r.sequence_accuracy,train_accuracy:$r.train_accuracy,
       observations:$r.observations,training_wall_seconds:$r.seconds,
       checkpoint_sha256:$r.checkpoint_sha256}]
| {scope:"75 short and 30 long task/seed cells, original budgets and first99 observations",
   control:"selected main LeRoPE mHC, runs/main-model-lerope-01/lerope",
   candidate:"phase-conjugated mHC on LeRoPE; 6 additional real parameters per layer; unchanged ordinary AdamW/WSD",
   audit_passed:$audit.audit_passed,phase:$audit.phase,complete:$audit.complete,
   promotion:$audit.promotion,audit_failures:$audit.audit_failures,
   source_sha256:$audit.source_sha256,ptx_sha256:$audit.ptx_sha256,
   rows:.,
   caveats:["Epochs measure first observed99, not sustained99.",
     "Retain189extreme primitive failures,203strict mixed-precision AD misses,147near-zero FP32 router misses and inherited numerical misses.",
     "All1440normalized-state tests and540CPU finite differences passed; finite screens are not universal numerical guarantees.",
     "Retain every individual epoch regression; the long suite is a task-coverage check.",
     "Throughput gains do not qualify a model for promotion.",
     "Incomplete tiers have no full-tier geometric mean; unrun cells are not passes."]}

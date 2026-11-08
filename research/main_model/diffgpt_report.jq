if .audit_passed != true or .completed_cells != 36 or .unrun_cells != 0 then error("DiffGPT report requires36audited cells") else . end
| . as $audit
| [.rows|group_by([.cell.tokenizer,.cell.features,.cell.seed])[]
    | [.[]|select(.cell.arm=="sequence")] as $s
    | [.[]|select(.cell.arm=="axial")] as $a
    | if ($s|length)!=1 or ($a|length)!=1 then error("missing or duplicate paired arm") else
      {tokenizer:$s[0].cell.tokenizer,features:$s[0].cell.features,seed:$s[0].cell.seed,
       sequence_status:$s[0].status,axial_status:$a[0].status,
       sequence_first90_updates:$s[0].first90_updates,axial_first90_updates:$a[0].first90_updates,
       sequence_first90_epochs:$s[0].first90_epochs,axial_first90_epochs:$a[0].first90_epochs,
       axial_over_sequence_first90_epochs:(if $s[0].first90_epochs>0 and $a[0].first90_epochs!=null
         then $a[0].first90_epochs/$s[0].first90_epochs else null end),
       lost_sequence_first90_pass:($s[0].status=="reached90" and $a[0].status!="reached90"),
       sequence_endpoint:$s[0].endpoint,axial_endpoint:$a[0].endpoint,
       sequence_extrapolation:$s[0].extrapolation,axial_extrapolation:$a[0].extrapolation} end]
| {audit_passed:true,kind:"additional_diffgpt_matched_native_study",paired_rows:.,
   groups:(group_by([.tokenizer,.features])|map({tokenizer:.[0].tokenizer,features:.[0].features,
     sequence_reached90:([.[]|select(.sequence_status=="reached90")]|length),
     axial_reached90:([.[]|select(.axial_status=="reached90")]|length),
     axial_lost_sequence_passes:([.[]|select(.lost_sequence_first90_pass)]|length),
     individual_rows:.})),
   registration_sha256:$audit.registration_sha256,
   fixed_suite_promotion_eligible:false,
   caveats:["Every encoder/feature/seed is reported separately; no pooled pass or hidden regression.",
     "First90 is the first observed decoded whole-row90%at50-update cadence, not sustained90%.",
     "Every cell runs the same5000update budget; report final accuracy separately.",
     "The native cycle-disjoint split and WSD model differ from historical Torch/ScheduleFree experiments.",
     "The forecast_reanchored metric uses the last raw observed rows and is distinct from the training-window decode frame.",
     "Contexts4and8use the final checkpoint without updates; native positions and YaRN are separate columns.",
     "This additional study cannot erase a lost pass in the fixed75short/30long suite."]}

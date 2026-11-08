if (map(.peak_lr)|sort) != [0.001,0.002,0.01] then
  error("expected the three declared LR studies exactly once")
else . end
| . as $studies
| ($studies[]|select(.peak_lr == 0.002)) as $control
| [ $studies[] as $study
    | {
      peak_lr: $study.peak_lr,
      complete: ($study.completed == 3 and ($study.integrity_failures|length) == 0
        and ($study.results|length) == 3),
      passed: ([$study.results[]|select(.status == "passed")]|length),
      lost_control_passes: [ $study.results[] as $r
        | $control.results[]|select(.seed == $r.seed and .status == "passed")
        | select($r.status == "budget_exhausted")|.seed ],
      geometric_mean_first99_epochs:
        (if all($study.results[]; .status == "passed")
         then ([$study.results[].first99_epochs|log]|add/3|exp) else null end),
      results: $study.results
    }
  ] as $arms
| {
  task: "arithmetic-sum-pow4",
  scope: "User-requested LR diagnostic; recipe selection for a fresh full main/mHC retest, not model promotion.",
  complete: all($arms[]; .complete),
  arms: ($arms|sort_by(.peak_lr)),
  selected_peak_lr:
    (if all($arms[]; .complete) then
       ($arms|map(select(.passed == 3 and (.lost_control_passes|length) == 0))
        |sort_by([.geometric_mean_first99_epochs, (.peak_lr != 0.002), .peak_lr])
        |first|.peak_lr)
     else null end),
  selection_rule: "Require all3 seeds to pass; minimize geometric-mean first99 epochs; retain LR0.002 on an exact tie."
}

def result($data; $seed): $data.results[] | select(.seed == $seed);
[
  range(0;3) as $seed
  | result($base[0]; $seed) as $a
  | result($higher[0]; $seed) as $b
  | {
    seed: $seed,
    lr_002_status: $a.status,
    lr_002_epochs: $a.first99_epochs,
    lr_01_status: $b.status,
    lr_01_epochs: $b.first99_epochs,
    lr_01_best_accuracy: $b.best_accuracy,
    lr_01_final_epochs: $b.final_epochs,
    lr_01_over_lr_002_epochs:
      (if $a.status == "passed" and $b.status == "passed"
       then $b.first99_epochs / $a.first99_epochs else null end),
    lost_lr_002_pass: ($a.status == "passed" and $b.status == "budget_exhausted")
  }
] as $pairs
| {
  task: "arithmetic-sum-pow4",
  model: "complex-peri-rms-ordinary-v1",
  scope: "User-requested known-long-task diagnostic; no full-suite promotion.",
  complete: ($base[0].completed == 3 and $higher[0].completed == 3
    and ($base[0].integrity_failures|length) == 0
    and ($higher[0].integrity_failures|length) == 0),
  pairs: $pairs,
  lost_lr_002_passes: [$pairs[]|select(.lost_lr_002_pass)|.seed],
  geometric_mean_lr_01_over_lr_002_epochs:
    (if all($pairs[]; .lr_01_over_lr_002_epochs != null)
     then ([$pairs[].lr_01_over_lr_002_epochs|log]|add/3|exp) else null end)
}

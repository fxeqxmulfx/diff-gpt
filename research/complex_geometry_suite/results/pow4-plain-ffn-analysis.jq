# jq -n --slurpfile radial CONTROL/history.jsonl --slurpfile plain CANDIDATE/history.jsonl -f this_file
def brief:
  {step,epochs,train_accuracy:.train.sequence_accuracy,train_loss:.train.loss,
   heldout_correct:.validation.exact,heldout_accuracy:.validation.sequence_accuracy,
   heldout_loss:.validation.loss,heldout_diagonal:(.validation.diagonal//null),
   heldout_non_diagonal:(.validation.non_diagonal//null)};
def summarize($h):
  ($h|length) as $n |
  ($h[-([200,$n]|min):]) as $tail |
  ($tail|map(.validation.sequence_accuracy)) as $values |
  ($values|add/length) as $mean |
  ($h|map(.validation.exact)|max) as $peak |
  {observations:$n,initial:($h[0]|brief),final:($h[-1]|brief),
   earliest_peak:([$h[]|select(.validation.exact==$peak)][0]|brief),
   first99:([$h[]|select(.validation.exact*100>=99*.validation.examples)][0]//null|if .==null then null else brief end),
   tail_observations:($tail|length),tail_mean:$mean,
   tail_stddev:($values|map(pow(.-$mean;2))|add/length|sqrt),
   tail_min:($values|min),tail_max:($values|max),
   largest_one_observation_drop:([0]+[range(1;$n) as $i|$h[$i-1].validation.sequence_accuracy-$h[$i].validation.sequence_accuracy]|max)};
($plain|length) as $n |
{radial_full:summarize($radial),plain:summarize($plain),radial_matched:summarize($radial[:$n]),
 matched_observations:$n,steps_identical:([range(0;$n) as $i|$radial[$i].step==$plain[$i].step]|all),
 same_exposure_final_accuracy_change:($plain[-1].validation.sequence_accuracy-$radial[$n-1].validation.sequence_accuracy),
 censored_after_first99:($plain[-1].step<$radial[-1].step)}

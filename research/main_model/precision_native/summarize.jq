def formats: ["fp64", "fp32_partial_fp64_total", "fp16_inputs_fp32_partial", "bf16_inputs_fp32_partial", "tf32_inputs_fp32_partial", "fp16_partial", "bf16_partial", "tf32_partial", "compensated_fp32_partial"];
[inputs] as $rows
| if any($rows[]; .mode == 0 and ((.forward_pass|not) or (.backward_pass|not) or (.finite|not)))
  then error("reference fails; screen is invalid") else
  {observations: ($rows|length),
   unique_pairs: ($rows | map([.width,.label]) | unique | length),
   by_origin: ($rows | group_by([.mode,.origin]) | map({mode:.[0].mode, format:formats[.[0].mode],origin:.[0].origin,cases:length,
     forward_failures:([.[]|select(.forward_pass|not)]|length),
     backward_failures:([.[]|select(.backward_pass|not)]|length),
     nonfinite:([.[]|select(.finite|not)]|length)})),
   by_pattern: ($rows | group_by([.mode, (.label|split("/")[0])]) | map({mode:.[0].mode,format:formats[.[0].mode],pattern:(.[0].label|split("/")[0]),cases:length,
     forward_failures:([.[]|select(.forward_pass|not)]|length),
     backward_failures:([.[]|select(.backward_pass|not)]|length),
     nonfinite:([.[]|select(.finite|not)]|length)})),
   baseline_passed:true,
   full_model_certified:false,
   numerical_candidate_promoted:false}
  end

# Prepare a replacement manifest only after the independent full-suite gate.
# Inputs: current model on stdin; $audit, $reference, $checkpoint are slurpfiles.
($audit[0]) as $a | ($reference[0]) as $r | ($checkpoint[0]) as $c
| if $a.audit_passed!=true or $a.complete!=true or $a.solved_cells!=105
    or $a.promotion.coverage_requirement_satisfied!=true
    or $a.promotion.eligible_for_promotion!=true
    or ($a.audit_failures|length)!=0 or ($a.promotion.lost_main_passes|length)!=0
    or $a.source_sha256!="7fcb3c976c34b6cdc7d53510ca4a27a0484d2e161bc9d208580b2da5c3043737"
    or $a.ptx_sha256!="10b9c9a47e8ae9cfbe37212efef8bd7f361f0185f4ba63c02cf98cd8ce3e5c3b"
  then error("full independent coverage/epoch gate not satisfied") else . end
| ([$a.rows[]|select(.case=="arithmetic-sum-pow4" and .seed==1)]|first) as $cell
| if $cell.audit!="passed" or $r.status!="passed" or $r.case!="arithmetic-sum-pow4" or $r.seed!=1
    or $r.first99_epochs!=$cell.first99_epochs or $r.completed!=$cell.completed
    or $c.state_sha256!=$cell.checkpoint_sha256 or $c.source_sha256!=$a.source_sha256 or $c.ptx_sha256!=$a.ptx_sha256
  then error("selected reference differs from independently audited cell") else . end
| .parent=.id
| .historical_mhc_manifest="research/main_model/results/gpu-suite-01/promotion/model-before.json"
| .id="complex-peri-rms-ordinary-mhc4-fp16tf32-v1"
| .name="Complex Peri-RMS + mHC (FP16/TF32)"
| .selected_by="standing_user_authorization_and_independent_epoch_coverage_gate"
| .architecture.dtype="FP16_forward_GEMM_TF32_backward_GEMM_FP32_accumulation_other_math_weights_and_moments"
| .precision_policy={selection:"static_per_operation",forward_affine_operands:"FP16",backward_affine_operands:"TF32",
    accumulation:"FP32",norms_and_routing:"FP32",master_weights_and_moments:"FP32",runtime_dtype_fallback:false,
    GPU_FP64_model_optimizer_operations:false,router_log_cycle:"two_FP32_words_for_offset_corrections"}
| .implementation |= (.cargo_manifest="research/main_model/gpu_opt_native/Cargo.toml"
    | .engine="research/main_model/gpu_opt_native/src/engine.rs"
    | .attention_kernel="research/main_model/gpu_opt_native/src/ordinary_forward.rs"
    | .attention_backward_kernel="research/main_model/gpu_opt_native/src/ordinary_attention.rs"
    | .qk_normalization_kernel="research/main_model/gpu_opt_native/src/attention.rs"
    | .residual_engine="research/main_model/gpu_opt_native/src/engine_mhc.rs"
    | .routing_kernel="research/main_model/gpu_opt_native/src/mhc.rs"
    | .frozen_executable="research/main_model/results/gpu-opt-05/combined"
    | .artifact_checksums="research/main_model/results/gpu-suite-01/promotion/runtime-artifacts.sha256"
    | .optimization_evidence="research/main_model/results/gpu-opt-05/REPORT.txt"
    | .optimized_ptx_sha256=$a.ptx_sha256 | .compiled_source_sha256=$a.source_sha256
    | .build_features=["tf32-backward","fused-attention","tiled-transpose"]
    | .optimization="static_tensor_precision_tiled_transpose_packed_warps_ordered_embedding_scan_and_GEMM_finish_fusion")
| .research_default_optimizer.cargo_manifest=.implementation.cargo_manifest
| .research_default_optimizer.frozen_executable=.implementation.frozen_executable
| .reference_configuration.role="current_static_tensor_mhc_quartic_seed1_reference"
| .reference_run |= (.path="runs/main-model-gpu-suite-01/mhc/arithmetic-sum-pow4/seed1"
    | .checkpoint_sha256=$c.state_sha256 | .source_sha256=$a.source_sha256 | .ptx_sha256=$a.ptx_sha256
    | .initial_parameters_sha256=$r.metadata.initial_parameters_sha256
    | .first99_epochs=$r.first99_epochs | .first99_updates=$r.completed | .audit_passed=true)
| .validation_scope |= (.paired_comparison="runs/main-model-gpu-suite-01"
    | .audit="research/main_model/results/gpu-suite-01/final-audit.json"
    | .control_cohort="runs/main-model-mhc-wsd-01/mhc"
    | .short_epoch_ratio_vs_parent=$a.promotion.short.geometric_mean_epoch_ratio
    | .long_epoch_ratio_vs_parent=$a.promotion.long.geometric_mean_epoch_ratio
    | .strict_scalar_AD={passed:922,missed:386,total:1308,all_directions_passed:false}
    | .FP64_optimizer_reference_comparison={passed:false,missed_coordinates:15337,max_absolute_error:6.315531209111214e-8}
    | .numerical_evidence="research/main_model/results/gpu-opt-05/REPORT.txt")
| .latest_promotion_decision={selected_arm:"mhc",architecture_changed:false,numerical_precision_changed:true,
    selected_by:"standing_user_authorization_and_independent_epoch_coverage_gate",
    evidence:"research/main_model/results/gpu-suite-01/promotion/decision.json",
    criterion:"All105verified main passes retained and fewer geometric-mean short epochs; runtime alone does not qualify."}

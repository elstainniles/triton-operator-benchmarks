| ID | kernel | 实现 | 类型 | 历史耗时/状态 |
| --- | --- | --- | --- | --- |
| LK01 | [_block_rms_norm_backward_kernel](_block_rms_norm_backward_kernel/) | generic | VV | 16.011 µs |
| LK02 | [_block_rms_norm_forward_kernel](_block_rms_norm_forward_kernel/) | generic | VV | 30.678 µs |
| LK03 | [_dyt_bwd_kernel](_dyt_bwd_kernel/) | generic | VV | 3.593 µs |
| LK04 | [_dyt_fwd_kernel](_dyt_fwd_kernel/) | generic | VV | 1.678 µs |
| LK05 | [_fused_add_rms_norm_backward_kernel](_fused_add_rms_norm_backward_kernel/) | generic | VV | 4.341 µs |
| LK06 | [_fused_add_rms_norm_forward_kernel](_fused_add_rms_norm_forward_kernel/) | generic | VV | 7.693 µs |
| LK07 | [_fused_neighborhood_attention_qk_kernel](_fused_neighborhood_attention_qk_kernel/) | generic | CV | 4.165 µs |
| LK08 | [_fused_neighborhood_attention_av_kernel](_fused_neighborhood_attention_av_kernel/) | generic | CV | 3.773 µs |
| LK09 | [_fused_neighborhood_attention_grad_qk_kernel](_fused_neighborhood_attention_grad_qk_kernel/) | generic | CV | 4.186 µs |
| LK10 | [_fused_neighborhood_attention_grad_k_kernel](_fused_neighborhood_attention_grad_k_kernel/) | generic | CV | 4.09 µs |
| LK11 | [_fused_neighborhood_attention_grad_v_kernel](_fused_neighborhood_attention_grad_v_kernel/) | generic | CV | 4.169 µs |
| LK12 | [_fused_neighborhood_attention_grad_attn_kernel](_fused_neighborhood_attention_grad_attn_kernel/) | generic | CV | 3.857 µs |
| LK13 | [_neighborhood_mask_kernel](_neighborhood_mask_kernel/) | generic | VV | 1.149 µs |
| LK14 | [_geglu_backward_kernel_flat](_geglu_backward_kernel_flat/) | ascend | VV | 233.167 µs |
| LK15 | [_geglu_forward_kernel_flat](_geglu_forward_kernel_flat/) | ascend | VV | 123.106 µs |
| LK16 | [_group_norm_backward_kernel](_group_norm_backward_kernel/) | generic | VV | 2.357 µs |
| LK17 | [_group_norm_forward_kernel](_group_norm_forward_kernel/) | generic | VV | 5.329 µs |
| LK18 | [_grpo_loss_bwd_kernel](_grpo_loss_bwd_kernel/) | generic | VV | 591.395 µs |
| LK19 | [_grpo_loss_fwd_kernel](_grpo_loss_fwd_kernel/) | generic | VV | 1143.945 µs |
| LK20 | [_jsd_kernel](_jsd_kernel/) | generic | VV | 353.591 µs |
| LK21 | [_kldiv_kernel_backward](_kldiv_kernel_backward/) | generic | VV | 725.458 µs |
| LK22 | [_kldiv_kernel_forward](_kldiv_kernel_forward/) | generic | VV | 641.0 µs |
| LK23 | [_layer_norm_backward_kernel](_layer_norm_backward_kernel/) | generic | VV | 96.133 µs |
| LK24 | [_llama4_rope_kernel](_llama4_rope_kernel/) | generic | VV | 6613.474 µs |
| LK25 | [_mask_bwd_kernel](_mask_bwd_kernel/) | generic | VV | 1.131 µs |
| LK26 | [_mask_fwd_kernel](_mask_fwd_kernel/) | generic | VV | 1.799 µs |
| LK27 | [_poly_norm_backward_kernel](_poly_norm_backward_kernel/) | generic | VV | 13.276 µs |
| LK28 | [_poly_norm_forward_kernel](_poly_norm_forward_kernel/) | generic | VV | 13.158 µs |
| LK29 | [_rms_norm_backward_kernel](_rms_norm_backward_kernel/) | generic | VV | 3.479 µs |
| LK30 | [_rms_norm_forward_kernel](_rms_norm_forward_kernel/) | generic | VV | 6.329 µs |
| LK31 | [_selective_log_softmax_kernel](_selective_log_softmax_kernel/) | generic | VV | 498.963 µs |
| LK32 | [_softmax_multi_block_backward_kernel](_softmax_multi_block_backward_kernel/) | generic | VV | compile_failed |
| LK33 | [_softmax_multi_block_forward_kernel](_softmax_multi_block_forward_kernel/) | generic | VV | compile_failed |
| LK34 | [_softmax_single_block_backward_kernel](_softmax_single_block_backward_kernel/) | generic | VV | 1.326 µs |
| LK35 | [_softmax_single_block_forward_kernel](_softmax_single_block_forward_kernel/) | generic | VV | 1.592 µs |
| LK36 | [_sparsemax_backward_kernel](_sparsemax_backward_kernel/) | generic | VV | blocked_by_forward_compile |
| LK37 | [_sparsemax_forward_kernel](_sparsemax_forward_kernel/) | generic | VV | compile_failed |
| LK38 | [_swiglu_backward_kernel_flat](_swiglu_backward_kernel_flat/) | ascend | VV | 1.578 µs |
| LK39 | [_swiglu_forward_kernel_flat](_swiglu_forward_kernel_flat/) | ascend | VV | 1.092 µs |
| LK40 | [_triton_qwen2vl_mrope_npu](_triton_qwen2vl_mrope_npu/) | ascend | VV | 7.694 µs |
| LK41 | [_tv_distance_kernel](_tv_distance_kernel/) | generic | VV | 4344.844 µs |
| LK42 | [element_mul_kernel](element_mul_kernel/) | generic | VV | 136.458 µs |
| LK43 | [embedding_backward_kernel](embedding_backward_kernel/) | ascend | VV | 65.462 µs |
| LK44 | [embedding_forward_kernel](embedding_forward_kernel/) | ascend | VV | 8.425 µs |
| LK45 | [liger_cross_entropy_kernel](liger_cross_entropy_kernel/) | generic | VV | 3697.92 µs |
| LK46 | [matmul_kernel](matmul_kernel/) | experimental | CV | 54.796 µs |

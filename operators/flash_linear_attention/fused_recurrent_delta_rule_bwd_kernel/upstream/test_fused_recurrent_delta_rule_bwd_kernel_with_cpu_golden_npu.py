import torch
import torch_npu
import triton
import triton.language as tl
import os


def cal_precision(input: torch.tensor, golden: torch.tensor):
    # 最大相对误差
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    # 平均相对误差
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    # 均方根误差
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    # 小值域绝对误差
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    
    return relative_max, relative_mean, rmse


def print_comparison(name, npu_val, gpu_val, threshold_max=3.0, threshold_mean=1.2, threshold_rmse=1.2):
    """打印比较结果并检查阈值"""
    print(f"\n{'='*60}")
    print(f"Comparing {name} outputs...")
    print(f"{'='*60}")
    
    # 计算比值
    if gpu_val[0] > 1e-10:  # 避免除以0
        ratio_max = npu_val[0] / gpu_val[0]
        ratio_mean = npu_val[1] / gpu_val[1]
        ratio_rmse = npu_val[2] / gpu_val[2]
    else:
        print(f"Warning: GPU {name} has very small error, using absolute check")
        ratio_max = npu_val[0]
        ratio_mean = npu_val[1]
        ratio_rmse = npu_val[2]
    
    print(f"\nNPU vs CPU_FP64:")
    print(f"  Max relative error:  {npu_val[0]:.6e}")
    print(f"  Mean relative error: {npu_val[1]:.6e}")
    print(f"  RMSE:                {npu_val[2]:.6e}")
    print(f"  Percentiles: 50%={npu_val[3]['50%']:.6e}, 90%={npu_val[3]['90%']:.6e}, "
          f"95%={npu_val[3]['95%']:.6e}, 99%={npu_val[3]['99%']:.6e}")
    
    print(f"\nGPU vs CPU_FP64:")
    print(f"  Max relative error:  {gpu_val[0]:.6e}")
    print(f"  Mean relative error: {gpu_val[1]:.6e}")
    print(f"  RMSE:                {gpu_val[2]:.6e}")
    print(f"  Percentiles: 50%={gpu_val[3]['50%']:.6e}, 90%={gpu_val[3]['90%']:.6e}, "
          f"95%={gpu_val[3]['95%']:.6e}, 99%={gpu_val[3]['99%']:.6e}")
    
    print(f"\nNPU/GPU Error Ratios:")
    print(f"  Max relative error ratio:  {ratio_max:.6f}")
    print(f"  Mean relative error ratio: {ratio_mean:.6f}")
    print(f"  RMSE ratio:                {ratio_rmse:.6f}")
    
    # 检查阈值
    if gpu_val[0] > 1e-10:
        assert ratio_max <= threshold_max, f"{name} 最大相对误差比 {ratio_max:.6f} > {threshold_max} check failed"
        assert ratio_mean <= threshold_mean, f"{name} 平均相对误差比 {ratio_mean:.6f} > {threshold_mean} check failed"
        assert ratio_rmse <= threshold_rmse, f"{name} 均方根误差比 {ratio_rmse:.6f} > {threshold_rmse} check failed"
    else:
        # GPU误差极小，检查NPU的绝对误差
        assert npu_val[0] < 0.5, f"{name} NPU absolute error check failed"
    
    print(f"\n✅ {name} precision check PASSED!")
    return ratio_max, ratio_mean, ratio_rmse


# Copy kernel definition from original file
@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_delta_rule_bwd_kernel(
    q,
    k,
    v,
    beta,
    h0,
    dh0,
    dht,
    do,
    dq,
    dk,
    dv,
    db,
    cu_seqlens,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V

    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    p_dk = dk + ((i_v * all + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_dv = dv + ((i_k * all + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos + T - 1) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dbeta = db + ((i_v * NK + i_k) * all + bos + T - 1) * H*V + i_h * V + tl.arange(0, BV)
    else:
        p_beta = beta + (bos + T - 1) * H + i_h
        p_dbeta = db + (i_v * all + bos + T - 1) * H + i_h

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_ht = dht + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_ht, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * (b_v * b_beta)[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)

        b_db = b_dv * b_v if IS_BETA_HEADWISE else tl.sum(b_dv * b_v)
        b_dv = b_dv * b_beta

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        if IS_BETA_HEADWISE:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty), mask=mask_v)
        else:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty))

        b_dh -= b_k[:, None] * b_dv[None, :]

        p_q -= H*K
        p_k -= H*K
        p_v -= H*V
        p_do -= H*V
        p_dk -= H*K
        p_dv -= H*V
        p_dbeta -= H * (V if IS_BETA_HEADWISE else 1)
        p_beta -= H * (V if IS_BETA_HEADWISE else 1)

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_k[:, None] & mask_v[None, :])

    tl.debug_barrier()

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + bos * H + i_h
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_dq = dq + ((i_v * all + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_dk = dk + ((i_v * all + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_dv = dv + ((i_k * all + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    if USE_INITIAL_STATE:
        mask_h = mask_k[:, None] & mask_v[None, :]
        p_h0 = h0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_dk = tl.load(p_dk, mask=mask_k, other=0).to(tl.float32)
        b_dv = tl.load(p_dv, mask=mask_v, other=0).to(tl.float32)
        b_dk -= tl.sum(b_dv[None, :] * b_h, axis=1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)

        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta

        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        d_q = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += H*K
        p_v += H*V
        p_do += H*V
        p_dq += H*K
        p_dk += H*K
        p_dv += H*V
        p_beta += H * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    dht: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton kernel wrapper for GPU/NPU"""
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 2

    beta_vector = beta.ndim == v.ndim

    dq = q.new_empty(NV, *q.shape)
    dk = q.new_empty(NV, *k.shape)
    dv = q.new_empty(NK, *v.shape)
    if beta_vector:
        db = q.new_empty(NV, NK, B, T, H, V)
    else:
        db = q.new_empty(NV, B, T, H)
    grid = (NV, NK, N * H)

    if initial_state is not None and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None

    fused_recurrent_delta_rule_bwd_kernel[grid](
        q,
        k,
        v,
        beta,
        initial_state,
        dh0,
        dht,
        do,
        dq,
        dk,
        dv,
        db,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NK=NK,
        IS_BETA_HEADWISE=beta_vector,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    db = db.sum((0, 1)) if beta_vector else db.sum(0)

    return dq, dk, dv, db, dh0


def fused_recurrent_delta_rule_bwd_cpu_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    dht: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch CPU reference implementation for backward pass.
    This is used as the golden reference with FP64 precision.
    
    Args:
        q: [B, T, H, K] queries
        k: [B, T, H, K] keys
        v: [B, T, H, V] values
        beta: [B, T, H] or [B, T, H, V] beta values
        dht: [B, H, K, V] gradient of final state
        do: [B, T, H, V] gradient of output
        scale: scaling factor
        initial_state: [B, H, K, V] initial state
        cu_seqlens: cumulative sequence lengths (not used in this case)
    
    Returns:
        dq: [B, T, H, K] gradient of queries
        dk: [B, T, H, K] gradient of keys
        dv: [B, T, H, V] gradient of values
        db: [B, T, H] or [B, T, H, V] gradient of beta
        dh0: [B, H, K, V] gradient of initial state
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    beta_vector = beta.ndim == v.ndim
    
    # Initialize outputs
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    if beta_vector:
        db = torch.zeros_like(beta)
    else:
        db = torch.zeros(B, T, H, dtype=beta.dtype, device=beta.device)
    
    # Process each batch and head
    for b in range(B):
        for h in range(H):
            # Initialize hidden state gradient
            if dht is not None:
                dh = dht[b, h].clone()  # [K, V]
            else:
                dh = torch.zeros(K, V, dtype=torch.float64, device=q.device)
            
            # Initialize hidden state for forward pass
            if initial_state is not None:
                h_state = initial_state[b, h].clone()  # [K, V]
            else:
                h_state = torch.zeros(K, V, dtype=torch.float64, device=q.device)
            
            # ========== Phase 1: Backward pass (reverse order) ==========
            # Compute dk, dv, db from T-1 to 0
            for t in range(T - 1, -1, -1):
                # Load current values
                q_t = q[b, t, h] * scale  # [K]
                k_t = k[b, t, h]  # [K]
                v_t = v[b, t, h]  # [V]
                do_t = do[b, t, h]  # [V]
                
                if beta_vector:
                    beta_t = beta[b, t, h]  # [V]
                else:
                    beta_t = beta[b, t, h]  # scalar
                
                # Compute gradients
                # dh += q_t[:, None] * do_t[None, :]
                dh = dh + q_t.unsqueeze(1) * do_t.unsqueeze(0)  # [K, V]
                
                # dk = sum(dh * (v_t * beta_t), dim=1)
                if beta_vector:
                    dk_t = torch.sum(dh * (v_t * beta_t).unsqueeze(0), dim=1)  # [K]
                else:
                    dk_t = torch.sum(dh * (v_t * beta_t).unsqueeze(0), dim=1)  # [K]
                
                # dv = sum(dh * k_t, dim=0)
                dv_t = torch.sum(dh * k_t.unsqueeze(1), dim=0)  # [V]
                
                # db
                if beta_vector:
                    db_t = dv_t * v_t  # [V]
                else:
                    db_t = torch.sum(dv_t * v_t)  # scalar
                
                # dv = dv * beta_t
                if beta_vector:
                    dv_t = dv_t * beta_t
                else:
                    dv_t = dv_t * beta_t
                
                # Store gradients
                dk[b, t, h] = dk_t
                dv[b, t, h] = dv_t
                if beta_vector:
                    db[b, t, h] = db_t
                else:
                    db[b, t, h] = db_t
                
                # Update dh for next iteration
                # dh -= k_t[:, None] * dv_t[None, :]
                dh = dh - k_t.unsqueeze(1) * dv_t.unsqueeze(0)  # [K, V]
            
            # Store dh0
            if initial_state is not None:
                dh0_val = dh
            else:
                dh0_val = None
            
            # ========== Phase 2: Forward pass to compute dq ==========
            # Reset hidden state
            if initial_state is not None:
                h_state = initial_state[b, h].clone()
            else:
                h_state = torch.zeros(K, V, dtype=torch.float64, device=q.device)
            
            # Compute dq from 0 to T-1
            for t in range(T):
                # Load current values
                k_t = k[b, t, h]  # [K]
                v_t = v[b, t, h]  # [V]
                do_t = do[b, t, h]  # [V]
                
                if beta_vector:
                    beta_t = beta[b, t, h]  # [V]
                else:
                    beta_t = beta[b, t, h]  # scalar
                
                # Load dk, dv from phase 1
                dk_t = dk[b, t, h]  # [K]
                dv_t = dv[b, t, h]  # [V]
                
                # Update dk: dk -= sum(dv_t * h_state, dim=1)
                dk_t = dk_t - torch.sum(dv_t.unsqueeze(0) * h_state, dim=1)  # [K]
                dk[b, t, h] = dk_t
                
                # Update h_state: h_state += k_t[:, None] * (v_t * beta_t)[None, :]
                if beta_vector:
                    h_state = h_state + k_t.unsqueeze(1) * (v_t * beta_t).unsqueeze(0)
                else:
                    h_state = h_state + k_t.unsqueeze(1) * (v_t * beta_t).unsqueeze(0)
                
                # Compute dq: dq = sum(h_state * do_t, dim=1) * scale
                dq_t = torch.sum(h_state * do_t.unsqueeze(0), dim=1) * scale  # [K]
                dq[b, t, h] = dq_t
    
    # Prepare dh0 output
    if initial_state is not None:
        dh0_out = torch.zeros_like(initial_state)
        for b in range(B):
            for h in range(H):
                # Recompute dh0 by running backward pass again
                if dht is not None:
                    dh = dht[b, h].clone()
                else:
                    dh = torch.zeros(K, V, dtype=torch.float64, device=q.device)
                
                for t in range(T - 1, -1, -1):
                    q_t = q[b, t, h] * scale
                    k_t = k[b, t, h]
                    v_t = v[b, t, h]
                    do_t = do[b, t, h]
                    
                    if beta_vector:
                        beta_t = beta[b, t, h]
                    else:
                        beta_t = beta[b, t, h]
                    
                    dh = dh + q_t.unsqueeze(1) * do_t.unsqueeze(0)
                    dv_t = torch.sum(dh * k_t.unsqueeze(1), dim=0)
                    
                    if beta_vector:
                        dv_t = dv_t * beta_t
                    else:
                        dv_t = dv_t * beta_t
                    
                    dh = dh - k_t.unsqueeze(1) * dv_t.unsqueeze(0)
                
                dh0_out[b, h] = dh
    else:
        dh0_out = None
    
    return dq, dk, dv, db, dh0_out


def test_with_cpu_fp64_as_golden():
    """
    使用CPU FP64作为标杆，比较NPU和GPU的精度
    """
    print("="*80)
    print("Testing fused_recurrent_delta_rule_bwd_kernel with CPU FP64 as golden")
    print("="*80)
    
    # Load inputs from saved pt file
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/fused_recurrent_delta_rule_bwd_kernel'
    inputs_dict = torch.load(os.path.join(load_dir, 'inputs.pt'))
    
    # 准备CPU FP64输入（作为golden reference）
    q_cpu_fp64 = inputs_dict['q'].to('cpu').to(torch.float64)
    k_cpu_fp64 = inputs_dict['k'].to('cpu').to(torch.float64)
    v_cpu_fp64 = inputs_dict['v'].to('cpu').to(torch.float64)
    beta_cpu_fp64 = inputs_dict['beta'].to('cpu').to(torch.float64)
    do_cpu_fp64 = inputs_dict['do'].to('cpu').to(torch.float64)
    dht_cpu_fp64 = inputs_dict['dht'].to('cpu').to(torch.float64)
    initial_state_cpu_fp64 = inputs_dict['initial_state'].to('cpu').to(torch.float64)
    initial_state_cpu_fp64.requires_grad = True
    scale = inputs_dict['scale']
    
    print(f"\nInput shapes:")
    print(f"  q: {q_cpu_fp64.shape}, dtype: {q_cpu_fp64.dtype}")
    print(f"  k: {k_cpu_fp64.shape}, dtype: {k_cpu_fp64.dtype}")
    print(f"  v: {v_cpu_fp64.shape}, dtype: {v_cpu_fp64.dtype}")
    print(f"  beta: {beta_cpu_fp64.shape}, dtype: {beta_cpu_fp64.dtype}")
    print(f"  scale: {scale}")
    
    # ========== Step 1: 在CPU上用FP64计算golden reference ==========
    print(f"\n{'='*80}")
    print("Step 1: Computing golden reference on CPU with FP64...")
    print("        Using pure PyTorch implementation (no Triton)")
    print(f"{'='*80}")
    
    dq_cpu_fp64, dk_cpu_fp64, dv_cpu_fp64, db_cpu_fp64, dh0_cpu_fp64 = fused_recurrent_delta_rule_bwd_cpu_ref(
        q=q_cpu_fp64,
        k=k_cpu_fp64,
        v=v_cpu_fp64,
        beta=beta_cpu_fp64,
        dht=dht_cpu_fp64,
        do=do_cpu_fp64,
        scale=scale,
        initial_state=initial_state_cpu_fp64,
    )
    device_npu = torch.device('npu:0')
    dq_cpu_fp64 = dq_cpu_fp64.to(device_npu)
    dk_cpu_fp64 = dk_cpu_fp64.to(device_npu)
    dv_cpu_fp64 = dv_cpu_fp64.to(device_npu)
    db_cpu_fp64 = db_cpu_fp64.to(device_npu)
    dh0_cpu_fp64  = dh0_cpu_fp64.to(device_npu)
    
    print(f"✅ CPU FP64 computation completed!")
    print(f"  dq shape: {dq_cpu_fp64.shape}, dtype: {dq_cpu_fp64.dtype}")
    print(f"  dk shape: {dk_cpu_fp64.shape}, dtype: {dk_cpu_fp64.dtype}")
    print(f"  dv shape: {dv_cpu_fp64.shape}, dtype: {dv_cpu_fp64.dtype}")
    print(f"  db shape: {db_cpu_fp64.shape}, dtype: {db_cpu_fp64.dtype}")
    if dh0_cpu_fp64 is not None:
        print(f"  dh0 shape: {dh0_cpu_fp64.shape}, dtype: {dh0_cpu_fp64.dtype}")
    
    # ========== Step 2: 在NPU上计算 ==========
    print(f"\n{'='*80}")
    print("Step 2: Computing on NPU with Triton kernel...")
    print(f"{'='*80}")
    
    q_npu = inputs_dict['q'].to(device_npu)
    k_npu = inputs_dict['k'].to(device_npu)
    v_npu = inputs_dict['v'].to(device_npu)
    beta_npu = inputs_dict['beta'].to(device_npu)
    do_npu = inputs_dict['do'].to(device_npu)
    dht_npu = inputs_dict['dht'].to(device_npu)
    initial_state_npu = inputs_dict['initial_state'].to(device_npu)
    initial_state_npu.requires_grad = True
    
    dq_npu, dk_npu, dv_npu, db_npu, dh0_npu = fused_recurrent_delta_rule_bwd(
        q=q_npu,
        k=k_npu,
        v=v_npu,
        beta=beta_npu,
        dht=dht_npu,
        do=do_npu,
        scale=scale,
        initial_state=initial_state_npu,
    )
    torch.npu.synchronize()
    
    print(f"✅ NPU computation completed!")
    print(f"  dq shape: {dq_npu.shape}, dtype: {dq_npu.dtype}")
    print(f"  dk shape: {dk_npu.shape}, dtype: {dk_npu.dtype}")
    print(f"  dv shape: {dv_npu.shape}, dtype: {dv_npu.dtype}")
    print(f"  db shape: {db_npu.shape}, dtype: {db_npu.dtype}")
    if dh0_npu is not None:
        print(f"  dh0 shape: {dh0_npu.shape}, dtype: {dh0_npu.dtype}")
    
    # ========== Step 3: 在GPU上计算 ==========
    # Load expected outputs and compare
    gpu_outputs = torch.load(os.path.join(load_dir, 'outputs.pt'))
    
    # Convert expected outputs to npu for comparison
    dq_gpu = gpu_outputs['dq'].to(device_npu)
    dk_gpu = gpu_outputs['dk'].to(device_npu)
    dv_gpu = gpu_outputs['dv'].to(device_npu)
    db_gpu = gpu_outputs['db'].to(device_npu)
    dh0_gpu = gpu_outputs['dh0'].to(device_npu) if gpu_outputs['dh0'] is not None else None
    
    print(f"✅ GPU computation completed!")
    print(f"  dq shape: {dq_gpu.shape}, dtype: {dq_gpu.dtype}")
    print(f"  dk shape: {dk_gpu.shape}, dtype: {dk_gpu.dtype}")
    print(f"  dv shape: {dv_gpu.shape}, dtype: {dv_gpu.dtype}")
    print(f"  db shape: {db_gpu.shape}, dtype: {db_gpu.dtype}")
    if dh0_gpu is not None:
        print(f"  dh0 shape: {dh0_gpu.shape}, dtype: {dh0_gpu.dtype}")
    
    # ========== Step 4: 比较精度 ==========
    print("\nComparing dq outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dq_npu, dq_cpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dq_gpu, dq_cpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")

    assert relative_max_npu / relative_max_gpu <= 2.0, "dq 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dq 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dq 均方根误差比 check failed"
    
    print("\nComparing dk outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk_npu, dk_cpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dk_gpu, dk_cpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2.0, "dk 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dk 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dk 均方根误差比 check failed"
    
    print("\nComparing dv outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dv_npu, dv_cpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dv_gpu, dv_cpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    # Note: dv has higher error ratio due to gradient computation complexity
    assert relative_max_npu / relative_max_gpu <= 2.0, "dv 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dv 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dv 均方根误差比 check failed"
    
    print("\nComparing db outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(db_npu, db_cpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(db_gpu, db_cpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2.0, "db 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "db 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "db 均方根误差比 check failed"
    
    if dh0_gpu is not None and dh0_cpu_fp64 is not None:
        print("\nComparing dh0 outputs...")
        relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dh0_npu, dh0_cpu_fp64)
        relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dh0_gpu, dh0_cpu_fp64)
        print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
        
        assert relative_max_npu / relative_max_gpu <= 2, "dh0 最大相对误差比 check failed"
        assert relative_mean_npu / relative_mean_gpu <= 1.2, "dh0 平均相对误差比 check failed"
        assert rmse_npu / rmse_gpu <= 1.2, "dh0 均方根误差比 check failed"
    
    print("All outputs match within tolerance!")


if __name__ == "__main__":
    test_with_cpu_fp64_as_golden()

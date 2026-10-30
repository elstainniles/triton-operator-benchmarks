# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
import os
import torch_npu


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


# episold
@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def intra_chunk_preprocess_bwd_kernel(
    q, k, w, w2, beta,
    AT,
    dA_local, dq, dq_new, dk, dk_new, dw, dbeta, dw1, dw2, T,
    offsets, indices,
    HQ: tl.constexpr, G: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw_beta = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dT = tl.zeros([BT, BT], dtype=tl.float32)

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta + (bos * H + i_h), (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    p_T = tl.make_block_ptr(AT + (bos * H + i_h) * BT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    b_Twb = tl.load(p_w2, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_T = tl.load(p_T, boundary_check=(0, 1))
    b_w_beta = (b_w * b_beta[:, None]).to(b_w.dtype)

    o_i = tl.arange(0, BT)
    b_qw = tl.where(o_i[:, None] >= o_i[None, :], tl.dot(b_q, tl.trans(b_w)), 0).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w_beta, tl.trans(b_k)), 0).to(b_k.dtype)
    b_Twbk = tl.dot(b_T, b_wbk).to(b_w.dtype)

    p_dA_local = tl.make_block_ptr(dA_local + (bos * HQ + i_hq) * BT, (T, BT), (BT*HQ, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA_local = tl.load(p_dA_local, boundary_check=(0, 1))

    # # Twb part qw part.
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dq = tl.load(p_dq, boundary_check=(0, 1))

    p_dw1 = tl.make_block_ptr(dw1 + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dw += tl.load(p_dw1, boundary_check=(0, 1))

    b_dqw = -tl.dot(b_dA_local, tl.trans(b_Twbk)) - tl.dot(b_dq.to(b_Twb.dtype), tl.trans(b_Twb))
    p_dw2 = tl.make_block_ptr(dw2 + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dTwb = -tl.dot(tl.trans(b_qw), b_dq) + tl.load(p_dw2, boundary_check=(0, 1))
    b_dT += tl.dot(b_dTwb.to(b_w_beta.dtype), tl.trans(b_w_beta))
    b_dw_beta += tl.dot(tl.trans(b_T), b_dTwb.to(b_T.dtype))

    b_dqw = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dqw, 0)
    b_dq += tl.dot(b_dA_local.to(b_k.dtype), b_k)
    b_dq += tl.dot(b_dqw.to(b_w.dtype), b_w)
    b_dw += tl.dot(tl.trans(b_dqw.to(b_q.dtype)), b_q)
    p_q_new = tl.make_block_ptr(dq_new + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, b_dq.to(dq_new.dtype.element_ty), boundary_check=(0, 1))

    # Twbk part
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dk = tl.load(p_dk, boundary_check=(0, 1))
    b_dTwbk = -tl.dot(tl.trans(b_qw), b_dA_local.to(b_qw.dtype)) - tl.dot(b_w, tl.trans(b_dk.to(b_w.dtype)))
    b_dw -= tl.dot(b_Twbk, b_dk.to(b_w.dtype))
    b_dT += tl.dot(b_dTwbk.to(b_wbk.dtype), tl.trans(b_wbk))
    b_dwbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(tl.trans(b_T), b_dTwbk.to(b_T.dtype)), 0).to(b_w.dtype)
    b_dw_beta += tl.dot(b_dwbk, b_k)

    b_dk += tl.dot(tl.trans(b_dwbk), b_w_beta)
    b_dk += tl.dot(tl.trans(b_dA_local), b_q)
    p_dk_new = tl.make_block_ptr(dk_new + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dk_new, b_dk.to(dk_new.dtype.element_ty), boundary_check=(0, 1))

    # matrix inverse's gradient
    p_T = tl.make_block_ptr(AT + (bos * H + i_h) * BT, (BT, T), (1, BT*H), (0, i_t * BT), (BT, BT), (0, 1))
    b_Tt = tl.load(p_T, boundary_check=(0, 1))
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dT, 0).to(b_w.dtype)
    b_dT = tl.dot(b_Tt, b_dT).to(b_w.dtype)
    b_dT = tl.dot(b_dT, b_Tt)
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dT, 0).to(b_k.dtype)

    b_dw_beta += tl.dot(b_dT, b_w)
    b_dw += tl.dot(tl.trans(b_dT), b_w_beta)
    b_dw += b_dw_beta * b_beta[:, None]
    b_dbeta = tl.sum(b_dw_beta * b_w, axis=1)

    p_dw = tl.make_block_ptr(dw + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dw, b_dw.to(dw.dtype.element_ty), boundary_check=(0, 1))
    p_dbeta = tl.make_block_ptr(dbeta + (bos * HQ + i_hq), (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    tl.store(p_dbeta, b_dbeta.to(dbeta.dtype.element_ty), boundary_check=(0, ))


def intra_chunk_preprocess_bwd_fn(q, k, w, w2, beta,
                                  dq, dk, dA_local,
                                  dw1, dw2,
                                  A, L, D, do, scale, cu_seqlens=None,
                                  chunk_indices: torch.LongTensor | None = None):
    BT = A.shape[-1]
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    G = HQ//H
    
    # Simplified version for testing - in real code this uses prepare_chunk_indices
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = None
    indices = chunk_indices
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    
    # better precision because h would be of norm smaller than 1 anyways
    dbeta = torch.empty(B, T, HQ, device=q.device, dtype=k.dtype if G == 1 else torch.float32)
    dw = torch.empty(B, T, HQ, K, device=q.device, dtype=k.dtype if G == 1 else torch.float32)
    dk_new = torch.empty_like(dk, dtype=k.dtype if G == 1 else torch.float32)  # float32 reduction
    dq_new = torch.empty_like(dq, dtype=q.dtype)

    # Simplified check_shared_mem for testing
    check_shared_mem_hopper = False  # Assume not hopper for testing

    intra_chunk_preprocess_bwd_kernel[grid](
        q=q, k=k, w=w, w2=w2, beta=beta,
        AT=A,
        dA_local=dA_local, dq=dq, dq_new=dq_new, dk=dk, dk_new=dk_new, dw=dw, dbeta=dbeta, dw1=dw1, dw2=dw2, T=T,
        offsets=cu_seqlens, indices=indices,
        HQ=HQ, G=G, H=H,
        K=K, BT=BT, BK=triton.next_power_of_2(K),
        num_stages=3 if check_shared_mem_hopper else 1,
    )
    return dq_new, dk_new, dbeta, dw


def test_intra_chunk_preprocess_bwd_kernel():
    """Test function for intra_chunk_preprocess_bwd_kernel with profiling"""
    
    # Load input tensors from pt files
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/intra_chunk_preprocess_bwd_kernel"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    device = torch.device('npu')
    
    # Convert loaded tensors to NPU
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    w = inputs_dict['w'].to(device)
    w2 = inputs_dict['w2'].to(device)
    beta = inputs_dict['beta'].to(device)
    dq = inputs_dict['dq'].to(device)
    dk = inputs_dict['dk'].to(device)
    dA_local = inputs_dict['dA_local'].to(device)
    dw1 = inputs_dict['dw1'].to(device)
    dw2 = inputs_dict['dw2'].to(device)
    A = inputs_dict['A'].to(device)
    L = inputs_dict['L'].to(device)
    D = inputs_dict['D'].to(device)
    do = inputs_dict['do'].to(device)
    scale = inputs_dict['scale']
    cu_seqlens = inputs_dict['cu_seqlens']
    
    # Calculate derived shapes
    B, T, HQ, K = dq.shape
    H = k.shape[2]
    BT = A.shape[-1]
    
    print(f"Testing intra_chunk_preprocess_bwd_kernel with config:")
    print(f"  B={B}, T={T}, HQ={HQ}, H={H}, K={K}")
    print(f"  BT={BT}")
    print(f"  Device: {device}")
    
    # Execute the kernel once
    print("\nExecuting kernel...")
    dq_new_npu, dk_new_npu, dbeta_npu, dw_npu = intra_chunk_preprocess_bwd_fn(
        q=q, k=k, w=w, w2=w2, beta=beta,
        dq=dq, dk=dk, dA_local=dA_local,
        dw1=dw1, dw2=dw2,
        A=A, L=L, D=D, do=do, scale=scale,
        cu_seqlens=cu_seqlens,
    )
    torch.npu.synchronize()
    
    # Load expected output tensors from pt file
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    dq_new_gpu_fp64 = outputs_dict['dq_new_fp64'].to(device)
    dk_new_gpu_fp64 = outputs_dict['dk_new_fp64'].to(device)
    dbeta_gpu_fp64 = outputs_dict['dbeta_fp64'].to(device)
    dw_gpu_fp64 = outputs_dict['dw_fp64'].to(device)
    dq_new_gpu = outputs_dict['dq_new'].to(device)
    dk_new_gpu = outputs_dict['dk_new'].to(device)
    dbeta_gpu = outputs_dict['dbeta'].to(device)
    dw_gpu = outputs_dict['dw'].to(device)
    
    # Compare outputs with expected values
    print("\nComparing dq_new outputs with expected values...")
    relative_max_npu_dq, relative_mean_npu_dq, rmse_npu_dq = cal_precision(dq_new_npu, dq_new_gpu_fp64)
    relative_max_gpu_dq, relative_mean_gpu_dq, rmse_gpu_dq = cal_precision(dq_new_gpu, dq_new_gpu_fp64)
    print(f"dq_new 最大相对误差比 :{relative_max_npu_dq / relative_max_gpu_dq}, 平均相对误差比 :{relative_mean_npu_dq / relative_mean_gpu_dq}, 均方根误差比 :{rmse_npu_dq / rmse_gpu_dq}")
    assert relative_max_npu_dq / relative_max_gpu_dq <= 2, "dq_new 最大相对误差比 check failed"
    assert relative_mean_npu_dq / relative_mean_gpu_dq <= 1.2, "dq_new 平均相对误差比 check failed"
    assert rmse_npu_dq / rmse_gpu_dq <= 1.2, "dq_new 均方根误差比 check failed"
    
    print("\nComparing dk_new outputs with expected values...")
    relative_max_npu_dk, relative_mean_npu_dk, rmse_npu_dk = cal_precision(dk_new_npu, dk_new_gpu_fp64)
    relative_max_gpu_dk, relative_mean_gpu_dk, rmse_gpu_dk = cal_precision(dk_new_gpu, dk_new_gpu_fp64)
    print(f"dk_new 最大相对误差比 :{relative_max_npu_dk / relative_max_gpu_dk}, 平均相对误差比 :{relative_mean_npu_dk / relative_mean_gpu_dk}, 均方根误差比 :{rmse_npu_dk / rmse_gpu_dk}")
    assert relative_max_npu_dk / relative_max_gpu_dk <= 2, "dk_new 最大相对误差比 check failed"
    assert relative_mean_npu_dk / relative_mean_gpu_dk <= 1.2, "dk_new 平均相对误差比 check failed"
    assert rmse_npu_dk / rmse_gpu_dk <= 1.2, "dk_new 均方根误差比 check failed"
    
    print("\nComparing dbeta outputs with expected values...")
    relative_max_npu_beta, relative_mean_npu_beta, rmse_npu_beta = cal_precision(dbeta_npu, dbeta_gpu_fp64)
    relative_max_gpu_beta, relative_mean_gpu_beta, rmse_gpu_beta = cal_precision(dbeta_gpu, dbeta_gpu_fp64)
    print(f"dbeta 最大相对误差比 :{relative_max_npu_beta / relative_max_gpu_beta}, 平均相对误差比 :{relative_mean_npu_beta / relative_mean_gpu_beta}, 均方根误差比 :{rmse_npu_beta / rmse_gpu_beta}")
    assert relative_max_npu_beta / relative_max_gpu_beta <= 2, "dbeta 最大相对误差比 check failed"
    assert relative_mean_npu_beta / relative_mean_gpu_beta <= 1.2, "dbeta 平均相对误差比 check failed"
    assert rmse_npu_beta / rmse_gpu_beta <= 1.2, "dbeta 均方根误差比 check failed"
    
    print("\nComparing dw outputs with expected values...")
    relative_max_npu_dw, relative_mean_npu_dw, rmse_npu_dw = cal_precision(dw_npu, dw_gpu_fp64)
    relative_max_gpu_dw, relative_mean_gpu_dw, rmse_gpu_dw = cal_precision(dw_gpu, dw_gpu_fp64)
    print(f"dw 最大相对误差比 :{relative_max_npu_dw / relative_max_gpu_dw}, 平均相对误差比 :{relative_mean_npu_dw / relative_mean_gpu_dw}, 均方根误差比 :{rmse_npu_dw / rmse_gpu_dw}")
    assert relative_max_npu_dw / relative_max_gpu_dw <= 2, "dw 最大相对误差比 check failed"
    assert relative_mean_npu_dw / relative_mean_gpu_dw <= 1.2, "dw 平均相对误差比 check failed"
    assert rmse_npu_dw / rmse_gpu_dw <= 1.2, "dw 均方根误差比 check failed"
    print("Output comparison passed!")
    
    # Profiling
    print("\nStarting profiling...")
    
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False
    )
    
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=torch_npu.profiler.schedule(
            wait=1,
            warmup=1,
            active=30,
            repeat=1,
            skip_first=1
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir_original")
    ) as prof:
        prof.start()
        for i in range(35):
            _ = intra_chunk_preprocess_bwd_fn(
                q=q, k=k, w=w, w2=w2, beta=beta,
                dq=dq, dk=dk, dA_local=dA_local,
                dw1=dw1, dw2=dw2,
                A=A, L=L, D=D, do=do, scale=scale,
                cu_seqlens=cu_seqlens,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("\nProfiling completed.")
    print(f"Profiling results saved to: ./result_dir_original")
    
    print("\nTest passed!")
    print(f"Output shapes: dq_new={dq_new_npu.shape}, dk_new={dk_new_npu.shape}, dbeta={dbeta_npu.shape}, dw={dw_npu.shape}")


if __name__ == "__main__":
    test_intra_chunk_preprocess_bwd_kernel()

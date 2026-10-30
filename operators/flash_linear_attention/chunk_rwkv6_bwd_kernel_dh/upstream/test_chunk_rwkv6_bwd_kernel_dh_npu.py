import sys
import os
import triton
import pytest
import torch
import triton.language as tl
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_bwd_kernel_dh(
    q, gi, ge, do, dh, dht, dh0, cu_seqlens, chunk_offsets, scale, T,
    HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NG: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr, USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + ((boh + i_t) * H + i_h) * K * V, (K, V), (V, 1),
                                 (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (K, T), (1, HQ * K),
                                (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1),
                                 (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        p_gk = tl.make_block_ptr(ge + (bos * H + i_h) * K, (K, T), (1, H * K),
                                 (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gk_last = gi + (bos + last_idx) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)

        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_q = (b_q * exp(b_gk) * scale).to(b_q.dtype)
        b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
        b_dh *= exp(b_gk_last)[:, None]
        b_dh += tl.dot(b_q, b_do)

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K * V, (K, V), (V, 1),
                                  (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def naive_chunk_rwkv6_bwd_dh(q, gi, ge, do_tensor, dht, scale, T, HQ, H, K, V, BT, NG,
                              STORE_INITIAL_STATE_GRADIENT, USE_FINAL_STATE_GRADIENT, IS_VARLEN,
                              cu_seqlens=None, chunk_offsets=None):
    q_f = q.to(torch.float32)
    do_f = do_tensor.to(torch.float32)
    dht_f = dht.to(torch.float32)

    B = q.shape[0]
    NT = T // BT

    ref_dh = torch.zeros(B, NT, HQ, K, V, dtype=torch.float32)
    ref_dh0 = torch.zeros(B, H, K, V, dtype=torch.float32)

    for i_n in range(B):
        for i_hq in range(HQ):
            i_h = i_hq // NG
            b_dh = dht_f[i_n, i_hq].clone()

            for i_t in range(NT - 1, -1, -1):
                ref_dh[i_n, i_t, i_hq] = b_dh

                b_q = q_f[i_n, i_t * BT:(i_t + 1) * BT, i_hq].T
                b_do = do_f[i_n, i_t * BT:(i_t + 1) * BT, i_hq]
                b_gk = ge[i_n, i_t * BT:(i_t + 1) * BT, i_h].T
                last_idx = min(i_t * BT + BT, T) - 1
                b_gk_last = gi[i_n, last_idx, i_h]

                b_q_modified = b_q * torch.exp(b_gk) * scale
                b_dh = b_dh * torch.exp(b_gk_last)[:, None]
                b_dh = b_dh + torch.matmul(b_q_modified, b_do)

            if STORE_INITIAL_STATE_GRADIENT:
                ref_dh0[i_n, i_h] = b_dh

    return ref_dh, ref_dh0


def cal_precision(input, golden):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    return relative_max, relative_mean, rmse


def third_part_cmp(y_npu, y_gpu, ref, name):
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(y_npu.to("cpu"), ref)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(y_gpu.to("cpu"), ref)
    print(
        f"{name} 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}, "
        f"平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}, "
        f"均方根误差比: {rmse_npu / rmse_gpu:.4f}")


@pytest.mark.functiontest
def test_chunk_rwkv6_bwd_kernel_dh_function():
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    torch.manual_seed(42)

    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_rwkv6_bwd_kernel_dh/test_chunk_rwkv6_bwd_kernel_dh_v2.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data = data["input_data"]
    grid = data["grid"]
    gpu_output_dh = data["gpu_output"]["dh"]
    gpu_output_dh0 = data["gpu_output"]["dh0"]

    T = input_data['T']
    HQ = input_data['HQ']
    H = input_data['H']
    K = input_data['K']
    V = input_data['V']
    BT = input_data['BT']
    NG = input_data['NG']
    NT = T // BT
    B = input_data['q'].shape[0]

    # 1> CPU fp32 高精度计算
    ref_dh, ref_dh0 = naive_chunk_rwkv6_bwd_dh(
        q=input_data['q'], gi=input_data['gi'], ge=input_data['ge'],
        do_tensor=input_data['do'], dht=input_data['dht'],
        scale=input_data['scale'], T=T, HQ=HQ, H=H, K=K, V=V, BT=BT, NG=NG,
        STORE_INITIAL_STATE_GRADIENT=input_data['STORE_INITIAL_STATE_GRADIENT'],
        USE_FINAL_STATE_GRADIENT=input_data['USE_FINAL_STATE_GRADIENT'],
        IS_VARLEN=input_data['IS_VARLEN'],
        cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None,
        chunk_offsets=input_data['chunk_offsets'] if input_data['chunk_offsets'].numel() > 0 else None,
    )

    # 2> NPU fp16 计算
    npu_input = input_data.copy()
    npu_input['dh'] = torch.empty(B, NT, HQ, K, V, dtype=torch.float16)
    npu_input['dh0'] = torch.empty(B, H, K, V, dtype=torch.float32)
    for k, v in npu_input.items():
        if isinstance(v, torch.Tensor):
            npu_input[k] = v.to('npu')

    chunk_rwkv6_bwd_kernel_dh[grid](
        q=npu_input['q'], gi=npu_input['gi'], ge=npu_input['ge'], do=npu_input['do'],
        dh=npu_input['dh'], dht=npu_input['dht'], dh0=npu_input['dh0'],
        cu_seqlens=npu_input['cu_seqlens'] if npu_input['cu_seqlens'].numel() > 0 else None,
        chunk_offsets=npu_input['chunk_offsets'] if npu_input['chunk_offsets'].numel() > 0 else None,
        scale=npu_input['scale'], T=npu_input['T'],
        HQ=npu_input['HQ'], H=npu_input['H'], K=npu_input['K'], V=npu_input['V'],
        BT=npu_input['BT'], BK=npu_input['BK'], BV=npu_input['BV'], NG=npu_input['NG'],
    )

    npu_dh = npu_input["dh"].cpu()
    npu_dh0 = npu_input["dh0"].cpu()

    # 3> 三方精度对比
    third_part_cmp(npu_dh, gpu_output_dh, ref_dh, "dh")
    third_part_cmp(npu_dh0, gpu_output_dh0, ref_dh0, "dh0")

    # 4> allclose 验证
    assert torch.allclose(npu_dh.float(), ref_dh.to(torch.float16).float(), atol=1e-2, rtol=1e-2), \
        "[dh] NPU 和 CPU fp32 结果不匹配"
    assert torch.allclose(npu_dh0.float(), ref_dh0, atol=1e-1, rtol=1e-2), \
        "[dh0] NPU 和 CPU fp32 结果不匹配"
    print(f"\n=== ALL TESTS PASSED ===")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
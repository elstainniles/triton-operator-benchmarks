import os
import torch
import torch_npu
import triton
import triton.language as tl
import functools
import inspect
from packaging import version
from typing import Optional


FLA_CACHE_RESULTS = False
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}
USE_CUDA_GRAPH = False
IS_GATHER_SUPPORTED = hasattr(triton.language, 'gather')


if not IS_GATHER_SUPPORTED:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        return None
else:
    gather = tl.gather


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2)
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def wu_fwd_kernel(
    w,
    u,
    ag,
    v,
    A_ab_inv,
    A_ak,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_s = tl.arange(0, BT)

    p_A_ab_inv = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_A_ak = tl.make_block_ptr(A_ak + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    b_Aab_inv = tl.load(p_A_ab_inv, boundary_check=(0, 1))
    b_Aak = tl.load(p_A_ak, boundary_check=(0, 1))
    b_Aab_inv = tl.where(o_s[:, None] >= o_s[None, :], b_Aab_inv, 0)
    b_Aak = tl.where(o_s[:, None] > o_s[None, :], b_Aak, 0)
    b_Aak = tl.dot(b_Aab_inv, b_Aak)
    b_Aak = b_Aak.to(v.dtype.element_ty, fp_downcast_rounding="rtne")
    b_Aab_inv = b_Aab_inv.to(ag.dtype.element_ty, fp_downcast_rounding="rtne")

    for i_k in range(tl.cdiv(K, BK)):
        p_ag = tl.make_block_ptr(ag + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_ag = tl.load(p_ag, boundary_check=(0, 1))
        b_w = tl.dot(b_Aab_inv, b_ag)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = tl.dot(b_Aak, b_v)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


def prepare_chunk_indices(
    cu_seqlens,
    chunk_size,
):
    if cu_seqlens is None:
        return None
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def wu_fwd(
    ag,
    v,
    A_ak,
    A_ab_inv,
    cu_seqlens,
    chunk_size,
    chunk_indices=None,
):
    B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64)

    w = torch.empty_like(ag)
    u = torch.empty_like(v)
    wu_fwd_kernel[(NT, B * H)](
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        w=w,
        u=u,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u


def cal_precision(input: torch.tensor, golden: torch.tensor):
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def test_wu_fwd_kernel():
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/wu_fwd_kernel"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"), map_location='cpu', weights_only=False)
    
    device = torch.device('npu')
    ag = inputs_dict['ag'].to(device)
    v = inputs_dict['v'].to(device)
    A_ak = inputs_dict['A_ak'].to(device)
    A_ab_inv = inputs_dict['A_ab_inv'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_size = inputs_dict['chunk_size']
    
    w, u = wu_fwd(
        ag,
        v,
        A_ak,
        A_ab_inv,
        cu_seqlens,
        chunk_size=chunk_size
    )
    
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"), map_location='cpu', weights_only=False)
    w_gpu_fp64 = outputs_dict['w_fp64'].to(device)
    u_gpu_fp64 = outputs_dict['u_fp64'].to(device)
    w_gpu = outputs_dict['w'].to(device)
    u_gpu = outputs_dict['u'].to(device)
    
    print("\nComparing w with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(w, w_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(w_gpu, w_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "w 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "w 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "w 均方根误差比 check failed"
    
    print("\nComparing u with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(u, u_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(u_gpu, u_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "u 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "u 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "u 均方根误差比 check failed"
    
    print("Test wu_fwd_kernel passed!")


if __name__ == "__main__":
    test_wu_fwd_kernel()

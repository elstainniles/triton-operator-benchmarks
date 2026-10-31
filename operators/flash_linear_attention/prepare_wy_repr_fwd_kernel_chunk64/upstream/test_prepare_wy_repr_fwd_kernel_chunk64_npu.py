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
        triton.Config({})
    ],
    key=['BC'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk64(
        A_ab,
        A_ab_inv,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        GATHER_SUPPORTED: tl.constexpr = IS_GATHER_SUPPORTED,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A1 = tl.make_block_ptr(A_ab + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    p_A2 = tl.make_block_ptr(A_ab + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
    p_A3 = tl.make_block_ptr(A_ab + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
    p_A_inv1 = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    p_A_inv2 = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + BC, BC), (BC, BC),
                                 (1, 0))
    p_A_inv3 = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + BC, 0), (BC, BC),
                                 (1, 0))
    p_A_inv4 = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, BC), (BC, BC),
                                 (1, 0))

    b_A = tl.load(p_A1, boundary_check=(0, 1))
    b_A2 = tl.load(p_A2, boundary_check=(0, 1))
    b_A3 = tl.load(p_A3, boundary_check=(0, 1))
    b_A = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0)
    b_A2 = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0)

    for i in range(1, BC):
        if GATHER_SUPPORTED:
            row_idx = tl.full([1, BC], i, dtype=tl.int16)
            b_a = tl.sum(gather(b_A, row_idx, axis=0), 0)
            b_a2 = tl.sum(gather(b_A2, row_idx, axis=0), 0)
        else:
            mask = tl.arange(0, BC) == i
            b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
            b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        mask = tl.arange(0, BC) == i
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)

    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = tl.dot(tl.dot(b_A2, b_A3), b_A)
    tl.store(p_A_inv1, b_A.to(p_A_inv1.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_inv2, b_A2.to(p_A_inv2.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_inv3, b_A3.to(p_A_inv3.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_inv4, tl.zeros([BC, BC], dtype=tl.float32).to(p_A_inv4.dtype.element_ty), boundary_check=(0, 1))


def prepare_chunk_indices(
        cu_seqlens,
        chunk_size,
):
    if cu_seqlens is None:
        return None
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_wy_repr_fwd(
        A_ab,
        cu_seqlens,
        chunk_size,
        chunk_indices=None,
):
    B, T, H, BT = A_ab.shape[0], A_ab.shape[1], A_ab.shape[2], chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(BT, 32)

    A_ab_inv = torch.empty_like(A_ab)
    prepare_wy_repr_fwd_kernel_chunk64[(NT, B * H)](
        A_ab=A_ab,
        A_ab_inv=A_ab_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        BC=BC,
    )
    return A_ab_inv


def cal_precision(input: torch.tensor, golden: torch.tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def test_prepare_wy_repr_fwd_kernel_chunk64():
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/prepare_wy_repr_fwd_kernel_chunk64"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"), map_location='cpu', weights_only=False)

    device = torch.device('npu')
    A_ab = inputs_dict['A_ab'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_size = inputs_dict['chunk_size']

    A_ab_inv = prepare_wy_repr_fwd(
        A_ab,
        cu_seqlens,
        chunk_size=chunk_size
    )

    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"), map_location='cpu', weights_only=False)
    A_ab_inv_gpu_fp64 = outputs_dict['A_ab_inv_fp64'].to(device)
    A_ab_inv_gpu = outputs_dict['A_ab_inv'].to(device)

    print("\nComparing A_ab_inv with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(A_ab_inv, A_ab_inv_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(A_ab_inv_gpu, A_ab_inv_gpu_fp64)
    print(
        f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "A_ab_inv 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "A_ab_inv 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "A_ab_inv 均方根误差比 check failed"

    print("Test prepare_wy_repr_fwd_kernel_chunk64 passed!")


if __name__ == "__main__":
    test_prepare_wy_repr_fwd_kernel_chunk64()

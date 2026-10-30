import os
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from typing import Optional
from enum import Enum
from typing import TYPE_CHECKING
import inspect
import functools
from packaging import version

if TYPE_CHECKING:
    from fla import __version__

FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

device_platform = 'cuda'
IS_AMD = (device_platform == 'hip')
IS_NVIDIA = (device_platform == 'cuda')
USE_CUDA_GRAPH = (IS_NVIDIA and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(torch.cuda.device_count())
        ]
    except BaseException:
        return [-1]


class Backend(Enum):
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


BK_LIST = [32, 64, 128] if check_shared_mem() else [16, 32]


@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({})
    ],
    key=['BK', 'NC', 'BT'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_bwd_kernel_intra(
    q,
    k,
    gi,
    ge,
    dA,
    dq,
    dk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    p_ge = tl.make_block_ptr(ge + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_ge = tl.load(p_ge, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = gi + (bos + i_t * BT + i_i * BC - 1) * H*K + i_h*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(gi+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA+(bos*H+i_h)*BT, (T, BT), (H*BT, 1), (i_t*BT+i_i*BC, i_j * BC), (BC, BC), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp2(b_gn[None, :] - b_gk)
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= exp2(b_ge - b_gn[None, :])

    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    o_dA = bos*H*BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_i * BC
    p_kj = k + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_gkj = gi + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_dq = tl.make_block_ptr(dq + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        m_i = o_i[:, None] > j
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * exp2(b_ge - b_gkj[None, :]), 0.)
        p_kj += H*K
        p_gkj += H*K
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gk = tl.make_block_ptr(gi + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = gi + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H*K + i_h*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            m_j = (i_t * BT + i_j * BC + tl.arange(0, BC)) < T
            p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
            p_gq = tl.make_block_ptr(ge + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + (bos*H+i_h)*BT, (BT, T), (1, H*BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.where(m_j[:, None] & m_k, tl.load(p_gq, boundary_check=(0, 1)), float('-inf'))
            b_qg = b_q * exp2(b_gq - b_gn[None, :])
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= exp2(b_gn[None, :] - b_gk)
    o_dA = bos*H*BT + (i_t * BT + i_i * BC) * H*BT + i_h * BT + i_i * BC + tl.arange(0, BC)
    p_qj = q + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_gqj = ge + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_dk = tl.make_block_ptr(dk + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dA = tl.load(dA + o_dA + j * H*BT)
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0).to(tl.float32)
        m_i = o_i[:, None] < j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * exp2(b_gqj[None, :] - b_gk), 0.)
        p_qj += H*K
        p_gqj += H*K
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def chunk_rwkv6_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    dA: torch.Tensor,
    chunk_size: int = 64,
):
    B, T, H, K = q.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))

    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    grid = (NK, NT * NC, B * H)
    chunk_rwkv6_bwd_kernel_intra[grid](
        q,
        k,
        gi,
        ge,
        dA,
        dq,
        dk,
        None,
        None,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    return dq, dk


def cal_precision(input: torch.Tensor, golden: torch.Tensor):
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


def test_chunk_rwkv6_bwd_kernel_intra():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/test_chunk_rwkv6_bwd_kernel_intra/'
    inputs_dict = torch.load(os.path.join(load_dir, "test_chunk_rwkv6_bwd_kernel_intra_cuda.pt"), map_location='cpu', weights_only=False)

    device = torch.device('npu')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    gi = inputs_dict['gi'].to(device)
    ge = inputs_dict['ge'].to(device)
    dA = inputs_dict['dA'].to(device)
    chunk_size = inputs_dict['chunk_size']

    dq, dk = chunk_rwkv6_bwd_intra(
        q=q,
        k=k,
        gi=gi,
        ge=ge,
        dA=dA,
        chunk_size=chunk_size,
    )

    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"), map_location='cpu', weights_only=False)
    dq_fp64 = outputs_dict['dq_fp64_actual'].to(device)
    dk_fp64 = outputs_dict['dk_fp64_actual'].to(device)
    dq_gpu = outputs_dict['dq'].to(device)
    dk_gpu = outputs_dict['dk'].to(device)

    print("\nComparing dq with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dq, dq_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dq_gpu, dq_fp64)
    print(f"dq - 最大相对误差比: {relative_max_npu / relative_max_gpu}, "
          f"平均相对误差比: {relative_mean_npu / relative_mean_gpu}, "
          f"均方根误差比: {rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "dq 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dq 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dq 均方根误差比 check failed"

    print("\nComparing dk with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk, dk_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dk_gpu, dk_fp64)
    print(f"dk - 最大相对误差比: {relative_max_npu / relative_max_gpu}, "
          f"平均相对误差比: {relative_mean_npu / relative_mean_gpu}, "
          f"均方根误差比: {rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "dk 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dk 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dk 均方根误差比 check failed"

    print("Test chunk_rwkv6_bwd_kernel_intra passed!")


if __name__ == "__main__":
    test_chunk_rwkv6_bwd_kernel_intra()

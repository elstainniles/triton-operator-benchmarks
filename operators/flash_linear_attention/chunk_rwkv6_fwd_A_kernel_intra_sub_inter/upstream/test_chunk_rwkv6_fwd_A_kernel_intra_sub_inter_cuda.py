import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp2
from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BC'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    gi,
    ge,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    m_i = i_t * BT + i_i * BC + tl.arange(0, BC) < T

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gq = tl.make_block_ptr(ge + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(gi + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = gi + (bos + i_t * BT + i_i * BC - 1) * H*K + i_h * K + o_k

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gq = tl.where(m_i[:, None] & m_k, tl.load(p_gq, boundary_check=(0, 1)), float('-inf'))
        b_qg = b_q * exp2(b_gq - b_gn[None, :]) * scale
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * exp2(b_gn[:, None] - b_gk)
        # [BC, BC] using tf32 to improve precision here.
        b_A += tl.dot(b_qg, b_kg)

    p_A = tl.make_block_ptr(A + (bos*H + i_h)*BT, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


def chunk_rwkv6_fwd_intra_sub_inter(
    q: torch.Tensor,
    k: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
):
    B, T, H, K = k.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)

    A = q.new_empty(B, T, H, BT, dtype=torch.float)
    grid = (NT, NC * NC, B * H)
    chunk_rwkv6_fwd_A_kernel_intra_sub_inter[grid](
        q,
        k,
        gi,
        ge,
        A,
        None,
        None,
        scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
    )
    return A


def test_chunk_rwkv6_fwd_A_kernel_intra_sub_inter():
    B, T, H, K = 4, 2048, 4, 256
    dtype = torch.float16
    device = torch.device('cuda')

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    gi = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    ge = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    chunk_size = 64
    scale = K ** -0.5

    save_dir = './'

    data = {}
    data['input_data'] = {}
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['gi'] = gi.cpu()
    data['input_data']['ge'] = ge.cpu()
    data['input_data']['chunk_size'] = chunk_size
    data['input_data']['scale'] = scale
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['K'] = K

    A = chunk_rwkv6_fwd_intra_sub_inter(
        q, k, gi, ge,
        scale=scale,
        chunk_size=chunk_size
    )

    data['gpu_output'] = {}
    data['gpu_output']['A'] = A.cpu()

    torch.save(data, f'{save_dir}/test_chunk_rwkv6_fwd_A_kernel_intra_sub_inter_cuda.pt')

    print("Test completed for chunk_rwkv6_fwd_A_kernel_intra_sub_inter, data saved.")


if __name__ == "__main__":
    test_chunk_rwkv6_fwd_A_kernel_intra_sub_inter()

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
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['BC', 'BK'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    gi,
    ge,
    u,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    o_A = (i_k * all + bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BC + i_h * BC
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_qj = q + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k
    p_kj = k + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k
    p_gk = gi + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (i_k * BK), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_kj[None, :] * exp2(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.)
        b_A = tl.where(o_i != j, b_A, tl.sum(b_qj * b_kj * b_u * scale))
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_qj += H*K
        p_kj += H*K
        p_gk += H*K


def chunk_rwkv6_fwd_intra_sub_intra_split(
    q: torch.Tensor,
    k: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    u: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
):
    B, T, H, K = k.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    BC = min(16, BT)
    BK = min(128, triton.next_power_of_2(K))
    NK = triton.cdiv(K, BK)
    NC = triton.cdiv(BT, BC)

    A_intra = q.new_empty(NK, B, T, H, BC, dtype=torch.float)
    grid = (NK, NT * NC, B * H)
    chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split[grid](
        q,
        k,
        gi,
        ge,
        u,
        A_intra,
        None,
        None,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    return A_intra


def test_chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split():
    B, T, H, K = 4, 2048, 4, 512
    dtype = torch.float32
    device = torch.device('cuda')

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    gi = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    ge = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    u = torch.randn(H, K, dtype=dtype, device=device)
    chunk_size = 64
    scale = K ** -0.5

    save_dir = './'

    data = {}
    data['input_data'] = {}
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['gi'] = gi.cpu()
    data['input_data']['ge'] = ge.cpu()
    data['input_data']['u'] = u.cpu()
    data['input_data']['chunk_size'] = chunk_size
    data['input_data']['scale'] = scale
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['K'] = K

    A_intra = chunk_rwkv6_fwd_intra_sub_intra_split(
        q, k, gi, ge, u,
        scale=scale,
        chunk_size=chunk_size
    )

    data['gpu_output'] = {}
    data['gpu_output']['A_intra'] = A_intra.cpu()

    torch.save(data, f'{save_dir}/test_chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split_cuda.pt')

    print("Test completed for chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split, data saved.")


if __name__ == "__main__":
    test_chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split()

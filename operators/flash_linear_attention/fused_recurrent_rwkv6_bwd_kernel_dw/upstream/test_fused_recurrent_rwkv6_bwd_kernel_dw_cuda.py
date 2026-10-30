import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT, 'BK': BK}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['K'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv6_bwd_kernel_dw(
    q,
    k,
    dq,
    dk,
    dw,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos
    NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.) if not REVERSE else tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BK], dtype=tl.float32)

    i_t = 0 if not REVERSE else NT - 1
    for _ in range(NT):
        p_q = tl.make_block_ptr(q + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T-1, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T-1, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_dq = tl.load(p_dq, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dw = (b_q * b_dq * scale) - b_k * b_dk
        b_c = b_z[None, :] + tl.dot(m_i, b_dw, allow_tf32=False)
        tl.store(p_dw, b_c.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_dw, 0)

        i_t += (1 if not REVERSE else -1)


def fused_recurrent_rwkv6_bwd_dw(
    q: torch.Tensor,
    k: torch.Tensor,
    dq1: torch.Tensor,
    dk1: torch.Tensor,
    scale: float | None = None,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    B, T, H, K = q.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    dw = torch.empty_like(q)
    def grid(meta): return (triton.cdiv(meta['K'], meta['BK']), N * H)
    fused_recurrent_rwkv6_bwd_kernel_dw[grid](
        q,
        k,
        dq1,
        dk1,
        dw,
        cu_seqlens,
        scale,
        T=T,
        H=H,
        K=K,
        REVERSE=not reverse,
    )
    return dw


def test_fused_recurrent_rwkv6_bwd_kernel_dw():
    B, T, H, K = 4, 2048, 4, 256
    dtype = torch.float16
    device = torch.device('cuda')

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    dq1 = torch.randn(B, T, H, K, dtype=dtype, device=device)
    dk1 = torch.randn(B, T, H, K, dtype=dtype, device=device)

    scale = K ** -0.5
    reverse = False

    save_dir = './'

    data = {}
    data['input_data'] = {}
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['dq1'] = dq1.cpu()
    data['input_data']['dk1'] = dk1.cpu()
    data['input_data']['scale'] = scale
    data['input_data']['reverse'] = reverse
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['K'] = K

    dw = fused_recurrent_rwkv6_bwd_dw(
        q, k, dq1, dk1,
        scale=scale,
        reverse=reverse
    )

    data['gpu_output'] = {}
    data['gpu_output']['dw'] = dw.cpu()

    torch.save(data, f'{save_dir}/test_fused_recurrent_rwkv6_bwd_kernel_dw_cuda.pt')

    print("Test completed for fused_recurrent_rwkv6_bwd_kernel_dw, data saved.")


if __name__ == "__main__":
    test_fused_recurrent_rwkv6_bwd_kernel_dw()

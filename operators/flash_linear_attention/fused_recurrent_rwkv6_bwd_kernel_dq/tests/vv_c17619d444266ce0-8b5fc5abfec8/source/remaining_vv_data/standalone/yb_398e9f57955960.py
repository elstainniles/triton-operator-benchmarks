import torch

import triton

from yb_ec8ad59510d3ad import fused_recurrent_rwkv6_bwd_kernel_dq
from yb_ec8ad59510d3ad import fused_recurrent_rwkv6_bwd_kernel_dq_fp64

def fused_recurrent_rwkv6_bwd_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    do: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    reverse: bool = False,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B

    BK, BV = min(triton.next_power_of_2(K), 16), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dq = q.new_empty(NV, *q.shape, dtype=torch.float)
    dq1 = torch.empty_like(dq)

    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_bwd_kernel_dq[grid](
        k,
        v,
        w,
        u,
        do,
        dq,
        dq1,
        initial_state,
        None,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
    )
    dq = dq.sum(0)
    dq1 = dq1.sum(0)
    return dq, dq1

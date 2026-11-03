import torch

import triton

from yb_ec8ad59510d3ad import fused_recurrent_rwkv6_fwd_kernel

def fused_recurrent_rwkv6_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B
    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float)

    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_fwd_kernel[grid](
        q,
        k,
        v,
        w,
        u,
        o,
        h0,
        ht,
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
    o = o.sum(0)
    return o, ht

import torch

import triton

from yb_ec8ad59510d3ad import fused_recurrent_rwkv6_bwd_kernel_dkv

def fused_recurrent_rwkv6_bwd_dkv(
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

    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dk = q.new_empty(NV, *k.shape, dtype=torch.float)
    dk1 = q.new_empty(NV, *k.shape, dtype=torch.float)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float)

    dh0 = torch.empty_like(initial_state) if initial_state is not None else None
    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_bwd_kernel_dkv[grid](
        q,
        k,
        v,
        w,
        u,
        do,
        dk,
        dk1,
        dv,
        dh0,
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
    dk = dk.sum(0)
    dk1 = dk1.sum(0)
    dv = dv.sum(0)
    return dk, dk1, dv, dh0

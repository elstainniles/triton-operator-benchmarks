import torch

import triton

from yb_2558c710edc3b7 import fused_recurrent_gsa_inference_kernel

def fused_recurrent_gsa_inference_wrapped(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    scale: float = 1.,
):
    B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    HQ = q.shape[2]
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NG = HQ // H

    if initial_state is not None:
        hk0, hv0 = initial_state
    else:
        hk0 = q.new_zeros(B, H, K, M, dtype=torch.float)
        hv0 = q.new_zeros(B, H, M, V, dtype=torch.float)

    hkt, hvt = None, None
    if output_final_state:
        if NG == 1:
            hkt, hvt = hk0, hv0
        else:
            hkt = q.new_empty(B, H, K, M, dtype=torch.float)
            hvt = q.new_empty(B, H, M, V, dtype=torch.float)

    o = v.new_empty(B, T, HQ, V)
    grid = (B * HQ,)
    fused_recurrent_gsa_inference_kernel[grid](
        q,
        k,
        v,
        s,
        g,
        o,
        hk0,
        hv0,
        hkt,
        hvt,
        scale=scale,
        K=K,
        V=V,
        M=M,
        BK=BK,
        BV=BV,
        NG=NG,
    )
    return o, (hkt, hvt)

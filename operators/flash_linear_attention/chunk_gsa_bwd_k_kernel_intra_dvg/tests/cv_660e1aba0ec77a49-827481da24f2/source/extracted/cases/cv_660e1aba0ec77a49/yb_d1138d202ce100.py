import torch

from yb_249d3453092fd3 import chunk_fwd_h_torch
from yb_249d3453092fd3 import chunk_bwd_dh_torch
from yb_249d3453092fd3 import chunk_local_cumsum_torch
from yb_249d3453092fd3 import chunk_gsa_bwd_k_kernel_dA_torch
from yb_249d3453092fd3 import chunk_gsa_bwd_k_kernel_dqkvg_torch
from yb_249d3453092fd3 import chunk_gsa_bwd_k_kernel_intra_dvg_torch
from yb_249d3453092fd3 import compare_precision

def chunk_gsa_bwd_k_full_torch(
    q, k, v, g, h, h0, o, do, dht=None, dg_input=None, scale=1., chunk_size=64
):
    """
    chunk_gsa_bwd_k 完整入口 - Torch Native实现

    流程:
      1. chunk_fwd_h:  计算 h (若h为None)
      2. chunk_bwd_dh:  计算 dh, dh0
      3. kernel_dA:     计算 dA
      4. kernel_dqkvg:  计算 A, dq, dk, dv, dgv
      5. kernel_intra_dvg: 计算 dv(intra), dg(intra)
      6. cumsum:        计算 dg_final = dgv + cumsum(dg_intra, reverse=True)

    输入:
        q:        [B, T, HQ, K]
        k:        [B, T, H, K]
        v:        [B, T, H, V]
        g:        [B, T, H, V]  (作为gv使用)
        h:        [B, NS, H, K, V] or None
        h0:       [N, H, K, V] or None
        o:        [B, T, HQ, V]  (forward输出)
        do:       [B, T, HQ, V]  (梯度)
        dht:      [N, HQ, K, V] or None
        dg_input: [B, T, HQ, V] or None
        scale:    float
        chunk_size: int (BT)

    输出:
        dq:  [B, T, HQ, K]
        dk:  [B, T, HQ, K]
        dv:  [B, T, HQ, V]
        dg:  [B, T, HQ, V]
        dh0: [N, H, K, V] or None
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size
    BC = min(16, BT)
    BK = 64
    BV = 64
    NS = (T + BT - 1) // BT

    if dg_input is None:
        dg_input = torch.zeros(B, T, HQ, V, dtype=torch.float32, device=q.device)

    # Step 1: chunk_fwd_h (若h为None则计算)
    if h is None:
        h, _ = chunk_fwd_h_torch(k, v, g, h0, BT=BT)

    # Step 2: chunk_bwd_dh
    dh, dh0 = chunk_bwd_dh_torch(q, g, do, dht, h0, BT=BT)

    # Step 3: kernel_dA
    dA = chunk_gsa_bwd_k_kernel_dA_torch(v, g, do, BT=BT, BC=BC, BV=BV, scale=scale)

    # Step 4: kernel_dqkvg
    A, dq, dk, dv, dgv = chunk_gsa_bwd_k_kernel_dqkvg_torch(
        q, k, v, h, g, do, dh, dA, dg_input, BT=BT, BK=BK, BV=BV, scale=scale
    )

    # Step 5: kernel_intra_dvg
    dv, dg = chunk_gsa_bwd_k_kernel_intra_dvg_torch(
        v, g, o, A, do, dv, dg_input, BT=BT, BC=BC, BV=BV
    )

    # Step 6: cumsum + final dg
    dg = dgv + chunk_local_cumsum_torch(dg, BT=BT, reverse=True)

    return dq, dk, dv, dg, dh0, dA, A

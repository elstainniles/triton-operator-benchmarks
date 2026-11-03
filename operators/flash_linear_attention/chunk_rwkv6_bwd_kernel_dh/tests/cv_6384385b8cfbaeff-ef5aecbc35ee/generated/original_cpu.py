"""Existing independent CPU equations; file-generation main omitted."""
import torch

def naive_chunk_rwkv6_bwd_dh(q, gi, ge, do_tensor, dht, scale, T, HQ, H, K, V, BT, NG, STORE_INITIAL_STATE_GRADIENT, USE_FINAL_STATE_GRADIENT, IS_VARLEN, cu_seqlens=None, chunk_offsets=None):
    q_f = q.to(torch.float32)
    do_f = do_tensor.to(torch.float32)
    dht_f = dht.to(torch.float32)
    B = q.shape[0]
    NT = T // BT
    ref_dh = torch.zeros(B, NT, HQ, K, V, dtype=torch.float32)
    ref_dh0 = torch.zeros(B, H, K, V, dtype=torch.float32)
    for i_n in range(B):
        for i_hq in range(HQ):
            i_h = i_hq // NG
            b_dh = dht_f[i_n, i_hq].clone()
            for i_t in range(NT - 1, -1, -1):
                ref_dh[i_n, i_t, i_hq] = b_dh
                b_q = q_f[i_n, i_t * BT:(i_t + 1) * BT, i_hq].T
                b_do = do_f[i_n, i_t * BT:(i_t + 1) * BT, i_hq]
                b_gk = ge[i_n, i_t * BT:(i_t + 1) * BT, i_h].T
                last_idx = min(i_t * BT + BT, T) - 1
                b_gk_last = gi[i_n, last_idx, i_h]
                b_q_modified = b_q * torch.exp(b_gk) * scale
                b_dh = b_dh * torch.exp(b_gk_last)[:, None]
                b_dh = b_dh + torch.matmul(b_q_modified, b_do)
            if STORE_INITIAL_STATE_GRADIENT:
                ref_dh0[i_n, i_h] = b_dh
    return (ref_dh, ref_dh0)

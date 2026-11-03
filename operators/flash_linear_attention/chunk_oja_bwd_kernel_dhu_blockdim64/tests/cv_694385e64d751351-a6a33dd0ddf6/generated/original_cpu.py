"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def naive_chunk_oja_bwd_dhu(q, vg, w, do, dk, gv, h0, dht, scale, chunk_size, dtype=torch.float64):
    B, T, H, K = q.shape
    V = vg.shape[-1]
    BT = 64
    NT = math.ceil(T / BT)
    dh = torch.zeros(B, NT, H, K, V, dtype=dtype)
    dh0 = torch.zeros(B, H, K, V, dtype=dtype)
    dk2 = torch.zeros(B, T, H, K, dtype=dtype)
    for i_b in range(B):
        for i_h in range(H):
            b_dh = dht[i_b, i_h].clone()
            for i_t in range(NT - 1, -1, -1):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                last_idx = t_end - 1
                dh[i_b, i_t, i_h] = b_dh
                vg_chunk = vg[i_b, t_start:t_end, i_h]
                dk_dot = vg_chunk @ b_dh.T
                b_dk2 = dk_dot + dk[i_b, t_start:t_end, i_h]
                dk2[i_b, t_start:t_end, i_h] = b_dk2
                gv_last = gv[i_b, last_idx, i_h]
                b_dh = b_dh * torch.exp(gv_last)[None, :]
                q_chunk = q[i_b, t_start:t_end, i_h]
                do_chunk = do[i_b, t_start:t_end, i_h]
                gv_chunk = gv[i_b, t_start:t_end, i_h]
                w_chunk = w[i_b, t_start:t_end, i_h]
                do_gated = do_chunk * torch.exp(gv_chunk)
                qdo = q_chunk.T @ do_gated * scale
                dkw = b_dk2.T @ w_chunk
                b_dh = b_dh + qdo - dkw
            dh0[i_b, i_h] = b_dh
    return (dh, dh0, dk2)

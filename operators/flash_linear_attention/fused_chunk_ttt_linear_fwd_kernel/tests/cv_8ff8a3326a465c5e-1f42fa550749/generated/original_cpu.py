"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def naive_fused_chunk_ttt_linear_fwd(q, k, v, w, b, eta, scale, eps, h0, hb0, output_final_state, BT, dtype=torch.float64):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = math.ceil(T / BT)
    o = torch.zeros(B, T, H, V, dtype=dtype)
    ht = torch.zeros(B, H, K, V, dtype=dtype) if output_final_state else None
    hbt = torch.zeros(B, H, 1, V, dtype=dtype) if output_final_state else None
    for i_b in range(B):
        for i_h in range(H):
            b_h = h0[i_b, i_h].clone().to(dtype)
            b_hb = hb0[i_b, i_h].clone().to(dtype).squeeze(0)
            b_w = w[i_h].to(dtype)
            b_b = b[i_h].to(dtype)
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                cur_BT = t_end - t_start
                k_chunk = k[i_b, t_start:t_end, i_h].to(dtype)
                v_chunk = v[i_b, t_start:t_end, i_h].to(dtype)
                b_kh = k_chunk @ b_h + b_hb.unsqueeze(0)
                mean = b_kh.mean(dim=1, keepdim=True)
                xbar = b_kh - mean
                var = (xbar ** 2).mean(dim=1, keepdim=True)
                rstd = 1.0 / torch.sqrt(var + eps)
                b_kh_hat = (b_kh - mean) * rstd
                b_v_new = b_kh_hat * b_w.unsqueeze(0) + b_b.unsqueeze(0) - v_chunk + k_chunk
                b_v_new = b_v_new * b_w.unsqueeze(0)
                b_v2 = rstd * (V * b_v_new - b_v_new.sum(dim=1, keepdim=True) - b_kh_hat * (b_v_new * b_kh_hat).sum(dim=1, keepdim=True)) / V
                q_chunk = q[i_b, t_start:t_end, i_h].to(dtype)
                e_chunk = eta[i_b, t_start:t_end, i_h, 0].to(dtype)
                q_scaled = q_chunk * scale
                b_A = q_scaled @ k_chunk.T
                causal_mask = torch.tril(torch.ones(cur_BT, cur_BT, dtype=dtype))
                b_A = b_A * causal_mask
                b_Ae = e_chunk.unsqueeze(1) * causal_mask
                b_o = -(e_chunk.unsqueeze(1) * b_A) @ b_v2
                b_o = b_o + b_hb.unsqueeze(0) - b_Ae @ b_v2
                b_o = b_o + q_scaled @ b_h
                o[i_b, t_start:t_end, i_h] = b_o
                if i_t == NT - 1:
                    e_last = eta[i_b, T - 1, i_h, 0].to(dtype)
                else:
                    e_last = eta[i_b, i_t * BT + BT - 1, i_h, 0].to(dtype)
                b_h = b_h - e_last * k_chunk.T @ b_v2
                b_hb = b_hb - e_last * b_v2.sum(dim=0)
            if output_final_state:
                ht[i_b, i_h] = b_h
                hbt[i_b, i_h, 0, :] = b_hb
    return (o, ht, hbt)

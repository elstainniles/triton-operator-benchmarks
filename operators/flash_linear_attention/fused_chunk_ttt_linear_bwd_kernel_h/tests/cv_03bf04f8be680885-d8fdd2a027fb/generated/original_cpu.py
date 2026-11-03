"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def naive_chunk_ttt_linear_bwd_h(k, v, w, b, eta, eps, h0, hb0, BT, dtype=torch.float64):
    B, T, H, K = k.shape
    V = v.shape[-1]
    NT = math.ceil(T / BT)
    h_out = torch.zeros(B, NT, H, K, V, dtype=dtype)
    v_new_out = torch.zeros(B, T, H, V, dtype=dtype)
    x_out = torch.zeros(B, T, H, V, dtype=dtype)
    y_out = torch.zeros(B, T, H, V, dtype=dtype)
    rstd_out = torch.zeros(B, T, H, 1, dtype=dtype)
    for i_b in range(B):
        for i_h in range(H):
            b_h = h0[i_b, i_h].clone().to(dtype) if h0 is not None else torch.zeros(K, V, dtype=dtype)
            b_hb = hb0[i_b, i_h].clone().to(dtype).squeeze(0) if hb0 is not None else torch.zeros(V, dtype=dtype)
            b_w = w[i_h].to(dtype)
            b_b = b[i_h].to(dtype)
            for i_t in range(NT):
                h_out[i_b, i_t, i_h] = b_h
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
                b_v_new_raw = (b_kh_hat * b_w.unsqueeze(0) + b_b.unsqueeze(0) - v_chunk + k_chunk) * b_w.unsqueeze(0)
                v_mean = b_v_new_raw.mean(dim=1, keepdim=True)
                v_khhat_dot = (b_v_new_raw * b_kh_hat).mean(dim=1, keepdim=True)
                b_v2 = rstd * (b_v_new_raw - v_mean - b_kh_hat * v_khhat_dot)
                x_out[i_b, t_start:t_end, i_h] = b_kh_hat
                y_out[i_b, t_start:t_end, i_h] = b_v_new_raw
                rstd_out[i_b, t_start:t_end, i_h, 0] = rstd.squeeze(-1)
                v_new_out[i_b, t_start:t_end, i_h] = b_v2
                if i_t == NT - 1:
                    e_last = eta[i_b, T - 1, i_h, 0].to(dtype)
                else:
                    e_last = eta[i_b, i_t * BT + BT - 1, i_h, 0].to(dtype)
                b_h = b_h - e_last * k_chunk.T @ b_v2
                b_hb = b_hb - e_last * b_v2.sum(dim=0)
    return (h_out, v_new_out, x_out, y_out, rstd_out)

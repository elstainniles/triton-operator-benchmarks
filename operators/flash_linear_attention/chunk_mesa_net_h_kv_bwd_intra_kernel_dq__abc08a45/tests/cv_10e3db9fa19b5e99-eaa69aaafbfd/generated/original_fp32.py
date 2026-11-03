"""Existing independent CPU equations; file-generation main omitted."""
import torch

def naive_chunk_mesa_net_h_kv_bwd_intra_dkv(q_star, k, v, beta, h_kv, g, do, dh_kv, B, T, H, K, V, BT):
    NT = (T + BT - 1) // BT
    dk = torch.zeros(B, T, H, K, dtype=torch.float32)
    dv = torch.zeros(B, T, H, V, dtype=torch.float32)
    dg = torch.zeros(B, T, H, dtype=torch.float32)
    for i_b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                i_tg = i_b * NT + i_t
                t_start = i_t * BT
                t_end = min(t_start + BT, T)
                bt = t_end - t_start
                if bt <= 0:
                    continue
                q = q_star[i_b, t_start:t_end, i_h].float()
                kk = k[i_b, t_start:t_end, i_h].float()
                vv = v[i_b, t_start:t_end, i_h].float()
                bbeta = beta[i_b, t_start:t_end, i_h].float()
                gg = g[i_b, t_start:t_end, i_h].float()
                ddo = do[i_b, t_start:t_end, i_h].float()
                h = h_kv[i_tg, i_h].float()
                dh = dh_kv[i_tg, i_h].float()
                g_last = gg[-1]
                dg_last = (h * dh).sum() * torch.exp(g_last)
                g_diff = gg[:, None] - gg[None, :]
                m = torch.tril(torch.exp(g_diff))
                k_beta = kk * bbeta[:, None]
                s = q @ k_beta.T * m
                ds_full = ddo @ vv.T
                dm = torch.tril(s * ds_full)
                b_dg = dm.sum(dim=1) - dm.sum(dim=0)
                g_exp_k = torch.exp(-gg + g_last)
                ds = ds_full * m
                dk_inter = vv @ dh.T * g_exp_k[:, None]
                b_dk = dk_inter
                dg_last = dg_last + (dk_inter * k_beta).sum()
                b_dg = b_dg - (dk_inter * k_beta).sum(dim=1)
                b_dv = k_beta @ dh * g_exp_k[:, None] + s.T @ ddo
                b_dk = b_dk + ds.T @ q
                b_dg[-1] = b_dg[-1] + dg_last
                dk[i_b, t_start:t_end, i_h] = b_dk
                dv[i_b, t_start:t_end, i_h] = b_dv
                dg[i_b, t_start:t_end, i_h] = b_dg
    return (dk, dv, dg)

def naive_chunk_mesa_net_h_kv_bwd_intra_dq(q_star, k, v, beta, h_kv, g, do, dg_prev, B, T, H, K, V, BT):
    NT = (T + BT - 1) // BT
    dq = torch.zeros(B, T, H, K, dtype=torch.float32)
    dg_final = torch.zeros(B, T, H, dtype=torch.float32)
    for i_b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                i_tg = i_b * NT + i_t
                t_start = i_t * BT
                t_end = min(t_start + BT, T)
                bt = t_end - t_start
                if bt <= 0:
                    continue
                q = q_star[i_b, t_start:t_end, i_h].float()
                kk = k[i_b, t_start:t_end, i_h].float()
                vv = v[i_b, t_start:t_end, i_h].float()
                bbeta = beta[i_b, t_start:t_end, i_h].float()
                gg = g[i_b, t_start:t_end, i_h].float()
                ddo = do[i_b, t_start:t_end, i_h].float()
                h = h_kv[i_tg, i_h].float()
                dg_p = dg_prev[i_b, t_start:t_end, i_h].float()
                g_diff = gg[:, None] - gg[None, :]
                m = torch.tril(torch.exp(g_diff))
                k_beta = kk * bbeta[:, None]
                ds = ddo @ vv.T * m
                g_exp_q = torch.exp(gg)
                b_dq = ddo @ h.T * g_exp_q[:, None]
                b_dg = (b_dq * q).sum(dim=1) + dg_p
                b_dq = b_dq + ds @ k_beta
                dq[i_b, t_start:t_end, i_h] = b_dq
                dg_final[i_b, t_start:t_end, i_h] = b_dg
    return (dq, dg_final)

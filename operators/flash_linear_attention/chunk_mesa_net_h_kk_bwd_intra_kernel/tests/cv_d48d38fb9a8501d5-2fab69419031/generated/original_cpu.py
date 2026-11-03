"""Existing independent CPU equations; file-generation main omitted."""
import torch
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import math

def naive_chunk_mesa_net_h_kk_bwd_intra(k, beta, g, h, dh, q_star, dq, dk_beta, chunk_size=64):
    B, T, H, D = k.shape
    k = k.double().clone()
    beta = beta.double().clone()
    g = g.double().clone()
    h = h.double().clone()
    dh = dh.double().clone()
    q_star = q_star.double().clone()
    dq = dq.double().clone()
    dk_beta = dk_beta.double().clone()
    NT = triton.cdiv(T, chunk_size)
    dk = torch.zeros(B, T, H, D, dtype=torch.float64)
    dg = torch.zeros(B, T, H, dtype=torch.float64)
    dbeta = torch.zeros(B, T, H, dtype=torch.float64)
    dlamb = torch.zeros(B, NT, H, D, dtype=torch.float64)
    for b in range(B):
        for h_idx in range(H):
            for t in range(NT):
                t_start = t * chunk_size
                t_end = min(t_start + chunk_size, T)
                BT = t_end - t_start
                g_chunk = g[b, t_start:t_end, h_idx]
                g_last = g_chunk[-1]
                gk = torch.where(torch.arange(BT) < BT, torch.exp(g_last - g_chunk), 0.0)
                q_star_chunk = q_star[b, t_start:t_end, h_idx, :]
                dq_chunk = dq[b, t_start:t_end, h_idx, :]
                k_chunk = k[b, t_start:t_end, h_idx, :]
                beta_chunk = beta[b, t_start:t_end, h_idx]
                dlamb[b, t, h_idx, :] = -(q_star_chunk * dq_chunk).sum(dim=0)
                v_chunk = k_chunk
                k_beta_chunk = k_chunk * beta_chunk.unsqueeze(-1)
                h_mat = h[t + b * NT, h_idx, :, :]
                dh_mat = dh[t + b * NT, h_idx, :, :]
                m = torch.zeros(BT, BT, dtype=torch.float64)
                for i in range(BT):
                    for j in range(i + 1):
                        m[i, j] = math.exp(g_chunk[i].item() - g_chunk[j].item())
                s = q_star_chunk @ k_beta_chunk.T * m
                ds = dq_chunk @ v_chunk.T
                dv = s.T @ dq_chunk
                dm = s * ds
                dm = torch.tril(dm)
                dg_chunk = dm.sum(dim=1) - dm.sum(dim=0)
                ds = ds * m
                dk_chunk = ds.T @ q_star_chunk
                dg_chunk += (dq_chunk @ h_mat * torch.exp(g_chunk).unsqueeze(-1) * q_star_chunk).sum(dim=1)
                dk2 = v_chunk @ dh_mat.T * gk.unsqueeze(-1)
                dg_chunk -= (dk2 * k_beta_chunk).sum(dim=1)
                dg_last = (dk2 * k_beta_chunk).sum()
                dk_chunk += dk2
                dv += k_beta_chunk @ dh_mat * gk.unsqueeze(-1)
                dh_prod = dh_mat * h_mat
                dg_last += dh_prod.sum() * math.exp(g_last.item())
                dk_chunk -= dk_beta[b, t_start:t_end, h_idx, :]
                dbeta_chunk = (dk_chunk * v_chunk).sum(dim=1)
                dk_chunk = dk_chunk * beta_chunk.unsqueeze(-1) + dv
                dk_chunk = -dk_chunk
                if BT > 0:
                    dg_chunk[-1] += dg_last
                dk[b, t_start:t_end, h_idx, :] = dk_chunk
                dg[b, t_start:t_end, h_idx] = -dg_chunk
                dbeta[b, t_start:t_end, h_idx] = -dbeta_chunk
    return (dk, dg, dbeta, dlamb)

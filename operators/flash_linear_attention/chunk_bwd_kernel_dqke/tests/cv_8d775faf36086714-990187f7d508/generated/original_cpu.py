"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def naive_chunk_bwd_kernel_dqke(q, k, v, eta, h, do, dh, dhb, scale, chunk_size, dtype=torch.float64):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = math.ceil(T / BT)
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    eta = eta.to(dtype)
    h = h.to(dtype)
    do = do.to(dtype)
    dh = dh.to(dtype)
    dhb = dhb.to(dtype)
    dq = torch.zeros(B, T, H, K, dtype=dtype)
    dk = torch.zeros(B, T, H, K, dtype=dtype)
    de = torch.zeros(B, T, H, 1, dtype=dtype)
    for i_b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                cur_BT = t_end - t_start
                q_c = q[i_b, t_start:t_end, i_h, :]
                k_c = k[i_b, t_start:t_end, i_h, :]
                v_c = v[i_b, t_start:t_end, i_h, :]
                e_c = eta[i_b, t_start:t_end, i_h, 0]
                h_c = h[i_b, i_t, i_h, :, :]
                do_c = do[i_b, t_start:t_end, i_h, :]
                dh_c = dh[i_b, i_t, i_h, :, :]
                dhb_c = dhb[i_b, i_t, i_h, 0, :]
                e_last = e_c[-1]
                ds = do_c @ v_c.T
                causal_mask = torch.tril(torch.ones(cur_BT, cur_BT, dtype=dtype))
                ds = ds * causal_mask
                dq_c = (do_c @ h_c.T - ds @ k_c * e_c[:, None]) * scale
                dk_c = -e_last * (v_c @ dh_c.T) - ds.T @ (q_c * e_c[:, None]) * scale
                de_c = torch.zeros(cur_BT, dtype=dtype)
                de_c[-1] -= torch.sum(dh_c * (k_c.T @ v_c))
                de_c[-1] -= torch.sum(dhb_c * v_c.sum(dim=0))
                de_c -= torch.sum(scale * ds @ k_c * q_c, dim=1)
                de_c -= torch.sum(ds, dim=1)
                dq[i_b, t_start:t_end, i_h, :] = dq_c
                dk[i_b, t_start:t_end, i_h, :] = dk_c
                de[i_b, t_start:t_end, i_h, 0] = de_c
    return (dq, dk, de)

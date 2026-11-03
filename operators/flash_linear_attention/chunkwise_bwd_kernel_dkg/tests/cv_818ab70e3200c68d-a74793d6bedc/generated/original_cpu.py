"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def diag_reference(data):
    input_data = data['input_data']
    do = input_data['do'].double()
    q = input_data['q'].double()
    k = input_data['k'].double()
    v = input_data['v'].double()
    g = input_data['g'].double()
    l = input_data['l'].double()
    llut = input_data['llut'].long()
    mask = input_data['mask']
    dq = input_data['dq'].double().clone()
    dk = input_data['dk'].double().clone()
    dv = input_data['dv'].double().clone()
    dg = input_data['dg'].double().clone()
    dl = input_data['dl'].double().clone()
    B = input_data['B']
    T = input_data['T']
    H = input_data['H']
    K = input_data['K']
    V = input_data['V']
    L = input_data['L']
    BT = input_data['BT']
    G = input_data['G']
    NT = input_data['NT']
    num_intra_levels = int(math.log2(BT)) + 1
    causal = torch.tril(torch.ones(BT, BT, dtype=torch.bool))
    row_idx = torch.arange(BT)[:, None]
    for i_n in range(B):
        for i_h in range(H):
            g_idx = i_h // (H // G)
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min(t_start + BT, T)
                chunk_size = t_end - t_start
                if chunk_size <= 0:
                    continue
                b_q = torch.zeros(K, BT, dtype=torch.float64)
                b_q[:, :chunk_size] = q[i_n, t_start:t_end, g_idx, :].T
                b_k = torch.zeros(BT, K, dtype=torch.float64)
                b_k[:chunk_size, :] = k[i_n, t_start:t_end, g_idx, :]
                b_v = torch.zeros(V, BT, dtype=torch.float64)
                b_v[:, :chunk_size] = v[i_n, t_start:t_end, i_h, :].T
                b_do = torch.zeros(BT, V, dtype=torch.float64)
                b_do[:chunk_size, :] = do[i_n, t_start:t_end, i_h, :]
                b_g = torch.zeros(BT, dtype=torch.float64)
                b_g[:chunk_size] = g[i_n, t_start:t_end, i_h]
                l_chunk = l[i_n, t_start:t_end, i_h, :]
                gathered = l_chunk[row_idx[:chunk_size].expand(chunk_size, BT), llut[:chunk_size, :]]
                b_h = torch.zeros(BT, BT, dtype=torch.float64)
                b_h[:chunk_size, :] = torch.where(causal[:chunk_size, :], gathered, torch.zeros_like(gathered, dtype=torch.float64))
                b_dq = torch.zeros(BT, K, dtype=torch.float64)
                b_dq[:chunk_size, :] = dq[i_n, t_start:t_end, i_h, :]
                b_dk = torch.zeros(BT, K, dtype=torch.float64)
                b_dk[:chunk_size, :] = dk[i_n, t_start:t_end, i_h, :]
                b_dv = torch.zeros(BT, V, dtype=torch.float64)
                b_dv[:chunk_size, :] = dv[i_n, t_start:t_end, i_h, :]
                b_dg = torch.zeros(BT, dtype=torch.float64)
                b_dg[:chunk_size] = dg[i_n, t_start:t_end, i_h]
                m_t = torch.zeros(BT, dtype=torch.bool)
                m_t[:chunk_size] = True
                b_s = torch.matmul(b_k, b_q)
                b_a = torch.where(causal & m_t[:, None] & m_t[None, :], torch.exp(b_g[:, None] - b_g[None, :]), torch.zeros(BT, BT, dtype=torch.float64))
                b_dv += torch.matmul(b_s * (b_a * b_h).T, b_do)
                b_ds = torch.matmul(b_do, b_v) * b_a
                b_dl = b_ds * b_s.T
                b_dg += torch.sum(b_dl * b_h, dim=1) - torch.sum(b_dl * b_h, dim=0)
                b_ds = b_ds * b_h
                b_dq += torch.matmul(b_ds, b_k)
                b_dk += torch.matmul(b_q, b_ds).T
                dq[i_n, t_start:t_end, i_h, :] = b_dq[:chunk_size, :]
                dk[i_n, t_start:t_end, i_h, :] = b_dk[:chunk_size, :]
                dv[i_n, t_start:t_end, i_h, :] = b_dv[:chunk_size, :]
                dg[i_n, t_start:t_end, i_h] = b_dg[:chunk_size]
                for i_level in range(num_intra_levels):
                    b_mask_i = mask[i_level]
                    dl_i = torch.sum(torch.where(b_mask_i == 1, b_dl, torch.zeros_like(b_dl)), dim=1)
                    dl[i_n, t_start:t_end, i_h, i_level] = dl_i[:chunk_size]
    output = {'dq': dq, 'dk': dk, 'dv': dv, 'dg': dg, 'dl': dl}
    return output

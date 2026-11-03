"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def tiled(x, bt):
    b, t, h, d = x.shape
    assert t % bt == 0
    return x.double().reshape(b, t // bt, bt, h, d).transpose(2, 3)

def untiled(x):
    b, n, h, t, d = x.shape
    return x.transpose(2, 3).reshape(b, n * t, h, d).contiguous()

def original_A(data):
    q, k = (data['q'], data['k'])
    B, T, HQ, K = q.shape
    BT = data['BT']
    NG = data['NG']
    scale = data['scale']
    A = torch.zeros(B, T, HQ, BT, dtype=q.dtype)
    for i_b in range(B):
        for i_t in range(T // BT):
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            for i_hq in range(HQ):
                q_chunk = q[i_b, t_start:t_end, i_hq, :].float()
                k_chunk = k[i_b, t_start:t_end, i_hq // NG, :].float()
                A_chunk = torch.matmul(q_chunk, k_chunk.T) * scale
                mask = torch.tril(torch.ones(t_end - t_start, t_end - t_start, dtype=torch.bool))
                A[i_b, t_start:t_end, i_hq, :t_end - t_start] = A_chunk * mask
    return A

def intra_reference(data):
    bt = data['BT']
    A = original_A(data)
    a = tiled(A, bt)
    v, g, o, do = (tiled(data[key], bt) for key in ('v', 'g', 'o', 'do'))
    dv = a.transpose(-1, -2) @ (do * g.exp()) * (-g).exp()
    dg = o * do - v * dv
    for i, l in ((0, 0), (19, 7), (63, 63)):
        direct = math.fsum((float(a[0, 0, 0, j, i]) * float(do[0, 0, 0, j, l]) * math.exp(float(g[0, 0, 0, j, l] - g[0, 0, 0, i, l])) for j in range(i, bt)))
        assert abs(float(dv[0, 0, 0, i, l]) - direct) < 1e-09
    return ({'dv': untiled(dv), 'dg': untiled(dg)}, {'restored_original_A_preparation': True, 'actual_A_nonzero_elements': int(A.count_nonzero()), 'scalar_dv_checks': 3, 'FP64_gold_final_rounding': False})

reference = intra_reference

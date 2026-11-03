"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def naive_chunk_ttt_linear_bwd_norm(q, k, v, v_new, x, y, rstd, w, b, eta, h0, hb0, h, dht, dhbt, dv_new, do, scale, chunk_size, dtype=torch.float64):
    B, T, H, K = q.shape
    V = do.shape[-1]
    BT = chunk_size
    NT = math.ceil(T / BT)
    dh = torch.zeros(B, NT, H, K, V, dtype=dtype)
    dhb = torch.zeros(B, NT, H, 1, V, dtype=dtype)
    dv = torch.zeros(B, T, H, V, dtype=dtype)
    dk = torch.zeros(B, T, H, K, dtype=dtype)
    dw_per_bh = torch.zeros(B, H, V, dtype=dtype)
    db_per_bh = torch.zeros(B, H, V, dtype=dtype)
    dh0 = torch.zeros(B, H, K, V, dtype=dtype) if h0 is not None else None
    dhb0 = torch.zeros(B, H, 1, V, dtype=dtype) if hb0 is not None else None
    for i_b in range(B):
        for i_h in range(H):
            b_dh = torch.zeros(K, V, dtype=dtype)
            b_dhb = torch.zeros(V, dtype=dtype)
            if dht is not None:
                b_dh = b_dh + dht[i_b, i_h].to(dtype)
            if dhbt is not None:
                b_dhb = b_dhb + dhbt[i_b, i_h].reshape(V).to(dtype)
            _w = w[i_h].to(dtype)
            _b = b[i_h].to(dtype)
            b_dw = torch.zeros(V, dtype=dtype)
            b_db = torch.zeros(V, dtype=dtype)
            for i_t in range(NT - 1, -1, -1):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                dh[i_b, i_t, i_h] = b_dh
                dhb[i_b, i_t, i_h, 0, :] = b_dhb
                _k = k[i_b, t_start:t_end, i_h].to(dtype)
                _dv_new = dv_new[i_b, t_start:t_end, i_h].to(dtype).clone()
                eta_last = eta[i_b, t_end - 1, i_h, 0].to(dtype)
                _dv_new = _dv_new - eta_last * (_k @ b_dh) - eta_last * b_dhb.unsqueeze(0)
                _v_new = v_new[i_b, t_start:t_end, i_h].to(dtype)
                _x = x[i_b, t_start:t_end, i_h].to(dtype)
                _y = y[i_b, t_start:t_end, i_h].to(dtype)
                _rstd = rstd[i_b, t_start:t_end, i_h, 0:1].to(dtype)
                _dy = _rstd * (_dv_new * V - _dv_new.sum(dim=-1, keepdim=True) - _x * (_dv_new * _x).sum(dim=-1, keepdim=True)) / V
                _dx = -_rstd * (_dv_new * (_x * _y).sum(dim=-1, keepdim=True) + _y * (_dv_new * _x).sum(dim=-1, keepdim=True)) / V
                _drstd = (_dv_new * _v_new / _rstd).sum(dim=-1, keepdim=True)
                _v = v[i_b, t_start:t_end, i_h].to(dtype)
                _dv = -_w.unsqueeze(0) * _dy
                _dk = _w.unsqueeze(0) * _dy
                b_dw = b_dw + (2 * _w.unsqueeze(0) * _x * _dy + (_b.unsqueeze(0) - _v + _k) * _dy).sum(dim=0)
                b_db = b_db + (_w.unsqueeze(0) * _dy).sum(dim=0)
                _dx = _dx + _w.unsqueeze(0) * _w.unsqueeze(0) * _dy
                _q = q[i_b, t_start:t_end, i_h].to(dtype)
                _h = h[i_b, i_t, i_h].to(dtype)
                _do = do[i_b, t_start:t_end, i_h].to(dtype)
                _dkh = _rstd * (V * _dx - _dx.sum(dim=-1, keepdim=True) - _x * (_x * _dx).sum(dim=-1, keepdim=True)) / V
                _dkh = _dkh - _rstd * _rstd * _drstd * _x / V
                _dk = _dk + _dkh @ _h.T
                b_dh = b_dh + (scale * _q).T @ _do + _k.T @ _dkh
                b_dhb = b_dhb + (_do + _dkh).sum(dim=0)
                dv[i_b, t_start:t_end, i_h] = _dv
                dk[i_b, t_start:t_end, i_h] = _dk
            dw_per_bh[i_b, i_h] = b_dw
            db_per_bh[i_b, i_h] = b_db
            if h0 is not None:
                dh0[i_b, i_h] = b_dh
            if hb0 is not None:
                dhb0[i_b, i_h, 0, :] = b_dhb
    dw_final = dw_per_bh.sum(dim=0)
    db_final = db_per_bh.sum(dim=0)
    return (dh, dhb, dh0, dhb0, dv, dk, dw_final, db_final)

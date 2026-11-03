import torch
import torch.nn.functional as F
import triton


def chunk_abc_pytorch_fwd(q, k, v, s, initial_state=None, output_final_state=False):
    B, H, T, K = q.shape
    V_dim = v.shape[-1]
    M = s.shape[-1]

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    s_f = s.float()

    z = torch.logcumsumexp(s_f, dim=2)
    scale = K ** -0.5

    qk = torch.einsum('bhtk,bhsk->bhts', q_f * scale, k_f)
    causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    qk = qk * causal_mask.float()

    exp_s = torch.exp(s_f)
    weighted_qk = torch.einsum('bhti,bhim->bhtm', qk, exp_s)
    ok = weighted_qk / torch.exp(z)

    p = torch.softmax(ok, dim=-1)

    exp_s_v = torch.einsum('bhtm,bhtv->bhtmv', exp_s, v_f)
    cum_sv = torch.cumsum(exp_s_v, dim=2)
    ov = torch.einsum('bhtm,bhtmv->bhtv', p, cum_sv / torch.exp(z)[..., None])

    return ov.to(v.dtype), None


def chunk_abc_bwd_kernel_dh_pytorch(q, z, do_, dh_dtype, scale, T, K, V, BT, BK, BV, NT, NORMK):
    B, H = q.shape[0], q.shape[1]
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    dh = torch.zeros(B, H, NT * K, V, dtype=torch.float32, device=q.device)

    for i_k in range(NK):
        for i_v in range(NV):
            b_dh = torch.zeros(B, H, BK, BV, dtype=torch.float32, device=q.device)
            zp_dim = BK if NORMK else BV
            b_zp = torch.full((B, H, zp_dim), float('inf'), dtype=torch.float32, device=q.device)

            for i_t in range(NT - 1, -1, -1):
                i_p = max(i_t * BT - 1, 0)
                t_start = i_t * BT
                t_end = min(t_start + BT, T)
                actual_BT = t_end - t_start

                q_chunk = q[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK].float().transpose(-1, -2)
                q_chunk = q_chunk * scale
                do_chunk = do_[:, :, t_start:t_end, i_v * BV:(i_v + 1) * BV].float()

                if actual_BT < BT:
                    q_chunk = F.pad(q_chunk, (0, BT - actual_BT))
                    do_chunk = F.pad(do_chunk, (0, 0, 0, BT - actual_BT))

                dh[:, :, i_t * K + i_k * BK:i_t * K + i_k * BK + BK, i_v * BV:i_v * BV + BV] = b_dh

                if NORMK:
                    z_chunk = z[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK].float().transpose(-1, -2)
                    if actual_BT < BT:
                        z_chunk = F.pad(z_chunk, (0, BT - actual_BT))
                    z_prev_end = z[:, :, i_p, i_k * BK:(i_k + 1) * BK].float()
                    z_start = z[:, :, t_start, i_k * BK:(i_k + 1) * BK].float()

                    decay = torch.exp(z_prev_end - b_zp)
                    b_zp = z_prev_end.clone()

                    q_adjusted = q_chunk * torch.exp(z_start.unsqueeze(-1) - z_chunk)
                    b_dh = b_dh * decay.unsqueeze(-1) + torch.matmul(q_adjusted, do_chunk) * torch.exp(z_prev_end - z_start).unsqueeze(-1)
                else:
                    z_chunk = z[:, :, t_start:t_end, i_v * BV:(i_v + 1) * BV].float()
                    if actual_BT < BT:
                        z_chunk = F.pad(z_chunk, (0, 0, 0, BT - actual_BT))
                    z_prev_end = z[:, :, i_p, i_v * BV:(i_v + 1) * BV].float()
                    z_start = z[:, :, t_start, i_v * BV:(i_v + 1) * BV].float()

                    decay = torch.exp(z_prev_end - b_zp)
                    b_zp = z_prev_end.clone()

                    do_adjusted = do_chunk * torch.exp(z_start.unsqueeze(-2) - z_chunk)
                    b_dh = b_dh * decay.unsqueeze(-2) + torch.matmul(q_chunk, do_adjusted) * torch.exp(z_prev_end - z_start).unsqueeze(-2)

    return dh.to(dh_dtype)


def chunk_abc_bwd_kernel_V_pytorch(k, v, z, h, A, do_, dh, scale, T, K, V, BT, BK, BV, NT, dp0=None, dsv0=None):
    """
    PyTorch equivalent of chunk_abc_bwd_kernel_V.
    In the V-path backward call: k=s, h=hv, A=Av, do=dov, dh=dhv, dq=dp1, dk=dsv1.
    dp0 and dsv0 are from chunk_abc_bwd_kernel_intra_V; if provided, they are added to dp1 and dsv1
    to match the in-place add_ behavior in prepare_bwd_data.
    """
    B, H = k.shape[0], k.shape[1]
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    dq = torch.zeros(B, H, T, K, dtype=torch.float32, device=k.device)
    dk = torch.zeros(B, H, T, K, dtype=torch.float32, device=k.device)
    dv_out = torch.zeros(NK, B, H, T, V, dtype=k.dtype, device=k.device)
    dA_out = torch.zeros(B, H, T, BT, dtype=torch.float32, device=k.device)

    for i_k in range(NK):
        for i_t in range(NT):
            i_p = max(i_t * BT - 1, 0)
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            actual_BT = t_end - t_start

            k_chunk = k[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK].float()
            z_end = z[:, :, t_start + BT - 1, i_k * BK:(i_k + 1) * BK].float()
            if actual_BT < BT:
                k_chunk = F.pad(k_chunk, (0, 0, 0, BT - actual_BT))

            k_norm = torch.exp(k_chunk - z_end.unsqueeze(2))

            A_T = A[:, :, t_start:t_end, :].transpose(-1, -2).float()
            if actual_BT < BT:
                A_T = F.pad(A_T, (0, 0, 0, BT - actual_BT))

            b_dq = torch.zeros(B, H, BT, BK, dtype=torch.float32, device=k.device)
            b_dk = torch.zeros(B, H, BT, BK, dtype=torch.float32, device=k.device)
            b_dA = torch.zeros(B, H, BT, BT, dtype=torch.float32, device=k.device)

            for i_v in range(NV):
                v_chunk = v[:, :, t_start:t_end, i_v * BV:(i_v + 1) * BV].float()
                h_slice = h[:, :, i_t * K + i_k * BK:i_t * K + i_k * BK + BK, i_v * BV:(i_v + 1) * BV]
                h_T = h_slice.transpose(-1, -2).float()
                do_chunk = do_[:, :, t_start:t_end, i_v * BV:(i_v + 1) * BV].float()
                dh_chunk = dh[:, :, i_t * K + i_k * BK:i_t * K + i_k * BK + BK, i_v * BV:(i_v + 1) * BV].float()

                if actual_BT < BT:
                    v_chunk = F.pad(v_chunk, (0, 0, 0, BT - actual_BT))
                    do_chunk = F.pad(do_chunk, (0, 0, 0, BT - actual_BT))

                b_dv = torch.matmul(k_norm, dh_chunk)
                if i_k == 0:
                    b_dv = b_dv + torch.matmul(A_T, do_chunk)

                do_scaled = do_chunk * scale

                dv_out[i_k, :, :, t_start:t_end, i_v * BV:(i_v + 1) * BV] = b_dv[:, :, :actual_BT, :].to(k.dtype)

                b_dA = b_dA + torch.matmul(do_scaled, v_chunk.transpose(-1, -2))
                b_dq = b_dq + torch.matmul(do_scaled, h_T)
                b_dk = b_dk + torch.matmul(v_chunk, dh_chunk.transpose(-1, -2))

            z_chunk = z[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK].float()
            z_prev = z[:, :, i_p, i_k * BK:(i_k + 1) * BK].float()
            z_start = z[:, :, t_start, i_k * BK:(i_k + 1) * BK].float()
            if actual_BT < BT:
                z_chunk = F.pad(z_chunk, (0, 0, 0, BT - actual_BT))

            z_factor = torch.exp(z_start.unsqueeze(2) - z_chunk) * torch.exp(z_prev - z_start).unsqueeze(2)
            b_dq = b_dq * z_factor
            b_dk = b_dk * k_norm

            dq[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK] = b_dq[:, :, :actual_BT, :]
            dk[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK] = b_dk[:, :, :actual_BT, :]

            causal_mask = torch.tril(torch.ones(BT, BT, device=k.device, dtype=torch.float32))
            b_dA = b_dA * causal_mask
            if i_k == 0:
                dA_out[:, :, t_start:t_end, :actual_BT] = b_dA[:, :, :actual_BT, :actual_BT]

    dv_out = dv_out.sum(0)
    if dp0 is not None:
        dq = dq + dp0.float()
    if dsv0 is not None:
        dk = dk + dsv0.float()
    return dq, dk, dv_out, dA_out


def chunk_abc_bwd_kernel_intra_V_pytorch(q, k, z, dA, T, K, BT, BC, BK, NC):
    """
    PyTorch equivalent of chunk_abc_bwd_kernel_intra_V.
    In the V-path backward: q=qv, k=s, dq=dp0, dk=dsv0.
    """
    B, H = q.shape[0], q.shape[1]
    NK = triton.cdiv(K, BK)
    NT = triton.cdiv(T, BT)

    dq = torch.zeros(B, H, T, K, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, H, T, K, dtype=torch.float32, device=q.device)

    for i_k in range(NK):
        for i_c in range(NT * NC):
            i_t = i_c // NC
            i_i = i_c % NC
            row_start = i_t * BT + i_i * BC
            if row_start >= T:
                continue
            actual_BC = min(BC, T - row_start)

            # ---- Part 1: Compute dq (dp0) ----
            z_norm = z[:, :, row_start, i_k * BK:(i_k + 1) * BK].float()
            z_sub = z[:, :, row_start:row_start + actual_BC, i_k * BK:(i_k + 1) * BK].float()
            b_zq = torch.exp(z_norm.unsqueeze(2) - z_sub)

            b_dq = torch.zeros(B, H, BC, BK, dtype=torch.float32, device=q.device)

            for i_j in range(i_i):
                j_start = i_t * BT + i_j * BC
                k_j = k[:, :, j_start:j_start + BC, i_k * BK:(i_k + 1) * BK].float()
                k_norm = torch.exp(k_j - z_norm.unsqueeze(2)).to(k.dtype).float()
                dA_block = dA[:, :, row_start:row_start + actual_BC, i_j * BC:i_j * BC + BC].to(dA.dtype).float()
                if actual_BC < BC:
                    k_norm = k_norm[:, :, :actual_BC, :]
                b_dq[:, :, :actual_BC, :] += torch.matmul(dA_block, k_norm[:, :, :actual_BC, :])

            b_dq[:, :, :actual_BC, :] *= b_zq

            o_i = torch.arange(BC, device=q.device)
            for j in range(actual_BC):
                dA_col = dA[:, :, row_start:row_start + actual_BC, i_i * BC + j].float()
                k_j = k[:, :, row_start + j, i_k * BK:(i_k + 1) * BK].float()
                m_i = o_i[:actual_BC] >= j
                contrib = dA_col.unsqueeze(-1) * torch.exp(k_j.unsqueeze(2) - z_sub)
                b_dq[:, :, :actual_BC, :] += torch.where(m_i.view(1, 1, -1, 1), contrib, 0.)

            dq[:, :, row_start:row_start + actual_BC, i_k * BK:(i_k + 1) * BK] = b_dq[:, :, :actual_BC, :]

            # ---- Part 2: Compute dk (dsv0) ----
            z_end_idx = min(row_start + BC - 1, T - 1)
            z_end = z[:, :, z_end_idx, i_k * BK:(i_k + 1) * BK].float()
            k_sub = k[:, :, row_start:row_start + actual_BC, i_k * BK:(i_k + 1) * BK].float()
            k_norm = torch.exp(k_sub - z_end.unsqueeze(2))

            b_dk = torch.zeros(B, H, BC, BK, dtype=torch.float32, device=q.device)

            for i_j in range(i_i + 1, NC):
                j_start = i_t * BT + i_j * BC
                if j_start >= T:
                    continue
                actual_BC_j = min(BC, T - j_start)
                q_j = q[:, :, j_start:j_start + actual_BC_j, i_k * BK:(i_k + 1) * BK].float()
                z_j = z[:, :, j_start:j_start + actual_BC_j, i_k * BK:(i_k + 1) * BK].float()
                q_norm = (q_j * torch.exp(z_end.unsqueeze(2) - z_j)).to(q.dtype).float()
                dA_block = dA[:, :, j_start:j_start + actual_BC_j, i_i * BC:i_i * BC + actual_BC].to(dA.dtype).float()
                b_dk[:, :, :actual_BC, :] += torch.matmul(dA_block.transpose(-1, -2), q_norm)[:, :, :actual_BC, :]

            b_dk[:, :, :actual_BC, :] *= k_norm

            for j in range(actual_BC):
                row_j = row_start + j
                dA_row = dA[:, :, row_j, i_i * BC:i_i * BC + actual_BC].float()
                q_j = q[:, :, row_j, i_k * BK:(i_k + 1) * BK].float()
                z_j = z[:, :, row_j, i_k * BK:(i_k + 1) * BK].float()
                m_i = o_i[:actual_BC] <= j
                contrib = dA_row.unsqueeze(-1) * q_j.unsqueeze(2) * torch.exp(k_sub - z_j.unsqueeze(2))
                b_dk[:, :, :actual_BC, :] += torch.where(m_i.view(1, 1, -1, 1), contrib, 0.)

            dk[:, :, row_start:row_start + actual_BC, i_k * BK:(i_k + 1) * BK] = b_dk[:, :, :actual_BC, :]

    return dq, dk


def softmax_bwd_kernel_pytorch(p, dp, dtype):
    """
    PyTorch equivalent of softmax_bwd_kernel.
    ds = p * dp - p * sum(p * dp)
    """
    p_f = p.float()
    dp_f = dp.float()
    pp = torch.sum(p_f * dp_f, dim=-1, keepdim=True)
    ds = p_f * dp_f - p_f * pp
    return ds.to(dtype)


def chunk_abc_bwd_kernel_intra_K_pytorch(v, z, do_, dA, scale, T, V, BT, BC, BV, NC):
    """
    PyTorch equivalent of chunk_abc_bwd_kernel_intra_K.
    In the K-path backward: v=s, do=dok, dA=dAk.
    """
    B, H = v.shape[0], v.shape[1]
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    n_bh = B * H

    dA_out = torch.zeros(NV, B, H, T, BT, dtype=v.dtype, device=v.device)

    for i_v in range(NV):
        for i_c in range(NT * NC * NC):
            i_t = i_c // (NC * NC)
            i_i = (i_c % (NC * NC)) // NC
            i_j = (i_c % (NC * NC)) % NC

            if i_i > i_j:
                v_j = v[:, :, i_t * BT + i_j * BC:i_t * BT + i_j * BC + BC, i_v * BV:(i_v + 1) * BV].float()
                z_i = z[:, :, i_t * BT + i_i * BC:i_t * BT + i_i * BC + BC, i_v * BV:(i_v + 1) * BV].float()
                z_norm = z[:, :, i_t * BT + i_i * BC, i_v * BV:(i_v + 1) * BV].float()
                do_i = do_[:, :, i_t * BT + i_i * BC:i_t * BT + i_i * BC + BC, i_v * BV:(i_v + 1) * BV].float()
                do_i = do_i * torch.exp(z_norm.unsqueeze(2) - z_i) * scale
                v_j_T = v_j.transpose(-1, -2)
                v_j_norm = torch.exp(v_j - z_norm.unsqueeze(2))
                b_dA = torch.matmul(do_i, v_j_norm.transpose(-1, -2))
                dA_out[i_v, :, :, i_t * BT + i_i * BC:i_t * BT + i_i * BC + BC, i_j * BC:i_j * BC + BC] = b_dA.to(v.dtype)

            elif i_i == i_j:
                row_start = i_t * BT + i_i * BC
                actual_BC = min(BC, T - row_start)
                if actual_BC <= 0:
                    continue

                v_ptr = v[:, :, row_start:, i_v * BV:(i_v + 1) * BV].float()
                z_i = z[:, :, row_start:row_start + actual_BC, i_v * BV:(i_v + 1) * BV].float()
                do_i = do_[:, :, row_start:row_start + actual_BC, i_v * BV:(i_v + 1) * BV].float() * scale

                o_i = torch.arange(BC, device=v.device)
                for j in range(actual_BC):
                    v_j = v_ptr[:, :, j, :].unsqueeze(2)  # (B, H, 1, BV)
                    b_dA = torch.sum(do_i * torch.exp(v_j - z_i), dim=-1)  # (B, H, actual_BC)
                    m_i = o_i[:actual_BC] >= j
                    b_dA = torch.where(m_i.unsqueeze(0).unsqueeze(0), b_dA, torch.zeros_like(b_dA))
                    dA_out[i_v, :, :, row_start:row_start + actual_BC, i_i * BC + j] = b_dA.to(v.dtype)

    return dA_out


def chunk_abc_bwd_kernel_K_pytorch(q, k, v, z, h, do_, dh, dA, scale, T, K, V, BT, BK, BV, NT):
    """
    PyTorch equivalent of chunk_abc_bwd_kernel_K.
    In the K-path backward: v=s, h=hk, do=dok, dh=dhk, dA=dAk.
    Outputs: dq, dk, dv(=dsk1).
    """
    B, H = q.shape[0], q.shape[1]
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    dq = torch.zeros(B, H, T, K, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, H, T, K, dtype=torch.float32, device=q.device)
    dv_out = torch.zeros(NK, B, H, T, V, dtype=torch.float32, device=q.device)

    for i_k in range(NK):
        for i_t in range(NT):
            i_p = max(i_t * BT - 1, 0)
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            actual_BT = t_end - t_start

            q_chunk = q[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK].float()
            k_chunk = k[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK].float()

            b_dq = torch.zeros(B, H, BT, BK, dtype=torch.float32, device=q.device)
            b_dk = torch.zeros(B, H, BT, BK, dtype=torch.float32, device=q.device)

            for i_v in range(NV):
                v_chunk = v[:, :, t_start:t_end, i_v * BV:(i_v + 1) * BV].float()
                z_chunk = z[:, :, t_start:t_end, i_v * BV:(i_v + 1) * BV].float()
                z_prev = z[:, :, i_p, i_v * BV:(i_v + 1) * BV].float()
                z_end = z[:, :, t_start + BT - 1, i_v * BV:(i_v + 1) * BV].float()
                z_start = z[:, :, t_start, i_v * BV:(i_v + 1) * BV].float()
                h_slice = h[:, :, i_t * K + i_k * BK:i_t * K + i_k * BK + BK, i_v * BV:(i_v + 1) * BV]
                h_T = h_slice.transpose(-1, -2).float()
                do_chunk = do_[:, :, t_start:t_end, i_v * BV:(i_v + 1) * BV].float()
                dh_chunk = dh[:, :, i_t * K + i_k * BK:i_t * K + i_k * BK + BK, i_v * BV:(i_v + 1) * BV].float()

                if actual_BT < BT:
                    v_chunk = F.pad(v_chunk, (0, 0, 0, BT - actual_BT))
                    z_chunk = F.pad(z_chunk, (0, 0, 0, BT - actual_BT))
                    do_chunk = F.pad(do_chunk, (0, 0, 0, BT - actual_BT))

                v_norm = torch.exp(v_chunk - z_end.unsqueeze(2))
                z_factor = torch.exp(z_start.unsqueeze(2) - z_chunk) * torch.exp(z_prev - z_start).unsqueeze(2)
                do_scaled = do_chunk * z_factor * scale

                b_dq = b_dq + torch.matmul(do_scaled, h_T)
                b_dk = b_dk + torch.matmul(v_norm, dh_chunk.transpose(-1, -2))

                k_dh = torch.matmul(k_chunk, dh_chunk)
                b_dv = v_norm * k_dh

                dv_out[i_k, :, :, t_start:t_end, i_v * BV:(i_v + 1) * BV] = b_dv[:, :, :actual_BT, :]

            dA_chunk = dA[:, :, t_start:t_end, :actual_BT].float()
            b_dq = b_dq + torch.matmul(dA_chunk, k_chunk)
            b_dk = b_dk + torch.matmul(dA_chunk.transpose(-1, -2), q_chunk)

            dq[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK] = b_dq[:, :, :actual_BT, :]
            dk[:, :, t_start:t_end, i_k * BK:(i_k + 1) * BK] = b_dk[:, :, :actual_BT, :]

    dv_out = dv_out.sum(0)
    return dq, dk, dv_out


def chunk_abc_bwd_kernel_intra_KV_pytorch(v, z, A, do_, T, V, BT, BC, BV, NC):
    """
    PyTorch equivalent of chunk_abc_bwd_kernel_intra_KV.
    In the K-path backward: v=s, A=Ak, do=dok, dv=dsk0.
    """
    B, H = v.shape[0], v.shape[1]
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)

    dv = torch.zeros(B, H, T, V, dtype=torch.float32, device=v.device)

    for i_v in range(NV):
        for i_c in range(NT * NC):
            i_t = i_c // NC
            i_i = i_c % NC
            row_start = i_t * BT + i_i * BC
            if row_start >= T:
                continue
            actual_BC = min(BC, T - row_start)

            v_sub = v[:, :, row_start:row_start + actual_BC, i_v * BV:(i_v + 1) * BV].float()
            z_end = z[:, :, min(row_start + BC - 1, T - 1), i_v * BV:(i_v + 1) * BV].float()

            b_dv = torch.zeros(B, H, BC, BV, dtype=torch.float32, device=v.device)

            for i_j in range(i_i + 1, NC):
                j_start = i_t * BT + i_j * BC
                if j_start >= T:
                    continue
                actual_BC_j = min(BC, T - j_start)
                z_j = z[:, :, j_start:j_start + actual_BC_j, i_v * BV:(i_v + 1) * BV].float()
                do_j = do_[:, :, j_start:j_start + actual_BC_j, i_v * BV:(i_v + 1) * BV].float()
                do_j = do_j * torch.exp(z_end.unsqueeze(2) - z_j)
                A_slice = A[:, :, j_start:j_start + actual_BC_j, i_i * BC:i_i * BC + actual_BC]
                A_block = A_slice.transpose(-1, -2).float()
                b_dv[:, :, :actual_BC, :] += torch.matmul(A_block, do_j)[:, :, :actual_BC, :]

            b_dv[:, :, :actual_BC, :] *= torch.exp(v_sub - z_end.unsqueeze(2))

            o_i = torch.arange(BC, device=v.device)
            for j in range(actual_BC):
                row_j = row_start + j
                z_j = z[:, :, row_j, i_v * BV:(i_v + 1) * BV].float()
                A_col = A[:, :, row_j, i_i * BC:i_i * BC + actual_BC].float()
                do_j = do_[:, :, row_j, i_v * BV:(i_v + 1) * BV].float()
                m_i = o_i[:actual_BC] <= j
                contrib = torch.exp(v_sub - z_j.unsqueeze(2)) * A_col.unsqueeze(-1) * do_j.unsqueeze(2)
                b_dv[:, :, :actual_BC, :] += torch.where(m_i.view(1, 1, -1, 1), contrib, 0.)

            dv[:, :, row_start:row_start + actual_BC, i_v * BV:(i_v + 1) * BV] = b_dv[:, :, :actual_BC, :]

    return dv


def chunk_abc_bwd_kernel_rcum_inter_pytorch(s, z, ss, T, S, BT, BS, NT):
    """
    PyTorch equivalent of chunk_abc_bwd_kernel_rcum_inter.
    """
    B, H = s.shape[0], s.shape[1]
    NS = triton.cdiv(S, BS)

    doo = torch.zeros(B, H, T, S, dtype=s.dtype, device=s.device)

    for i_m in range(NS):
        b_sp = torch.zeros(B, H, BS, dtype=torch.float32, device=s.device)
        b_zp = torch.full((B, H, BS), float('inf'), dtype=torch.float32, device=s.device)

        for i_t in range(NT - 1, -1, -1):
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            actual_BT = t_end - t_start

            s_chunk = s[:, :, t_start:t_end, i_m * BS:(i_m + 1) * BS].float()
            z_chunk = z[:, :, t_start:t_end, i_m * BS:(i_m + 1) * BS].float()
            z_start = z[:, :, t_start, i_m * BS:(i_m + 1) * BS].float()
            z_c = z[:, :, t_start, i_m * BS:(i_m + 1) * BS].float()
            ss_chunk = ss[:, :, t_start:t_end, i_m * BS:(i_m + 1) * BS].float()

            if actual_BT < BT:
                s_chunk = F.pad(s_chunk, (0, 0, 0, BT - actual_BT))
                z_chunk = F.pad(z_chunk, (0, 0, 0, BT - actual_BT))
                ss_chunk = F.pad(ss_chunk, (0, 0, 0, BT - actual_BT))

            b_doo = torch.exp(s_chunk - z_start.unsqueeze(2)) * b_sp.unsqueeze(2) * torch.exp(z_start.unsqueeze(2) - b_zp.unsqueeze(2))
            doo[:, :, t_start:t_end, i_m * BS:(i_m + 1) * BS] = b_doo[:, :, :actual_BT, :].to(s.dtype)

            b_sp = b_sp * torch.exp(z_c - b_zp) + torch.exp(z_c - z_start) * torch.sum(ss_chunk * torch.exp(z_start.unsqueeze(2) - z_chunk), dim=2)
            b_zp = z_c.clone()

    return doo


def chunk_abc_bwd_kernel_rcum_intra_pytorch(s, z, ss, doo, T, S, BT, BC, BS, NC):
    """
    PyTorch equivalent of chunk_abc_bwd_kernel_rcum_intra.
    doo is modified in-place (adds intra-chunk contributions).
    """
    B, H = s.shape[0], s.shape[1]
    NS = triton.cdiv(S, BS)
    NT = triton.cdiv(T, BT)

    for i_s in range(NS):
        for i_c in range(NT * NC):
            i_t = i_c // NC
            i_i = i_c % NC
            row_start = i_t * BT + i_i * BC
            if row_start >= T:
                continue
            actual_BC = min(BC, T - row_start)

            s_sub = s[:, :, row_start:row_start + actual_BC, i_s * BS:(i_s + 1) * BS].float()
            z_end = z[:, :, min(row_start + BC - 1, T - 1), i_s * BS:(i_s + 1) * BS].float()

            b_doo = torch.zeros(B, H, BC, BS, dtype=torch.float32, device=s.device)

            for i_j in range(i_i + 1, NC):
                j_start = i_t * BT + i_j * BC
                if j_start >= T:
                    continue
                actual_BC_j = min(BC, T - j_start)
                z_j = z[:, :, j_start:j_start + actual_BC_j, i_s * BS:(i_s + 1) * BS].float()
                ss_j = ss[:, :, j_start:j_start + actual_BC_j, i_s * BS:(i_s + 1) * BS].float()
                b_doo[:, :, :actual_BC, :] += ss_j[:, :, :actual_BC, :] * torch.exp(z_end.unsqueeze(2) - z_j[:, :, :actual_BC, :])

            ones_mat = torch.ones(BC, BC, dtype=torch.float32, device=s.device)
            b_doo = torch.exp(s_sub - z_end.unsqueeze(2)) * torch.matmul(ones_mat, b_doo)

            o_i = torch.arange(BC, device=s.device)
            for j in range(actual_BC):
                row_j = row_start + j
                z_j = z[:, :, row_j, i_s * BS:(i_s + 1) * BS].float()
                ss_j = ss[:, :, row_j, i_s * BS:(i_s + 1) * BS].float()
                m_i = o_i[:actual_BC] <= j
                contrib = torch.exp(s_sub - z_j.unsqueeze(2)) * ss_j.unsqueeze(2)
                b_doo[:, :, :actual_BC, :] += torch.where(m_i.view(1, 1, -1, 1), contrib, 0.)

            doo_existing = doo[:, :, row_start:row_start + actual_BC, i_s * BS:(i_s + 1) * BS].float()
            doo[:, :, row_start:row_start + actual_BC, i_s * BS:(i_s + 1) * BS] = (b_doo[:, :, :actual_BC, :] + doo_existing).to(s.dtype)

    return doo




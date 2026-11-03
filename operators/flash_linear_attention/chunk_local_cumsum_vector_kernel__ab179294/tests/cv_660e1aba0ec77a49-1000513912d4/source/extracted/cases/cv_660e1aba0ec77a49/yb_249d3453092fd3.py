import torch

def compare_precision(name, tensor1, tensor2, verbose=True):
    if tensor1.shape != tensor2.shape:
        print(f"{name}: Shape mismatch {tensor1.shape} vs {tensor2.shape}")
        return False, None
    
    diff = tensor1.float() - tensor2.float()
    abs_diff = torch.abs(diff)
    rel_diff = torch.abs(diff / (torch.abs(tensor2.float()) + 1e-8))
    
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    mean_abs = abs_diff.mean().item()
    mean_rel = rel_diff.mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    
    if verbose:
        print(f"{name}:")
        print(f"  Max absolute error: {max_abs:.6e}")
        print(f"  Max relative error: {max_rel:.6e}")
        print(f"  Mean absolute error: {mean_abs:.6e}")
        print(f"  Mean relative error: {mean_rel:.6e}")
        print(f"  RMSE: {rmse:.6e}")
        
        if max_abs > 1e-5:
            max_abs_idx = abs_diff.argmax()
            flat_idx = max_abs_idx.item()
            idx = []
            size = 1
            for s in reversed(tensor1.shape):
                idx.insert(0, flat_idx % s)
                flat_idx //= s
            print(f"  Max error location: {idx}")
            print(f"  Expected (GPU): {tensor2[tuple(idx)].item():.6e}")
            print(f"  Got (Torch): {tensor1[tuple(idx)].item():.6e}")
        
        passed = max_abs < 1e-5
        print(f"  Result: {'PASS' if passed else 'FAIL'}")
        print()
    
    return max_abs < 1e-5, {'max_abs': max_abs, 'max_rel': max_rel, 'mean_abs': mean_abs, 'mean_rel': mean_rel, 'rmse': rmse}

def chunk_fwd_h_torch(k, v, g, h0, BT=64):
    """Torch implementation of chunk_fwd_h"""
    B, T, H, K = k.shape
    V = v.shape[-1]
    NT = (T + BT - 1) // BT
    
    h_list = []
    if h0 is not None:
        h = h0.clone().float().view(B, H, K, V)
    else:
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=k.device)
    
    for i_t in range(NT):
        t_start = i_t * BT
        t_end = min((i_t + 1) * BT, T)
        
        h_list.append(h.clone())
        
        k_block = k[:, t_start:t_end, :, :].float()
        v_block = v[:, t_start:t_end, :, :].float()
        
        if g is not None:
            g_block = g[:, t_start:t_end, :, :].float()
            v_block = v_block * torch.exp(g_block)
        
        for b in range(B):
            for h_idx in range(H):
                h[b, h_idx, :, :] = h[b, h_idx, :, :] + torch.matmul(k_block[b, :, h_idx, :].t(), v_block[b, :, h_idx, :])
    
    return torch.stack(h_list, dim=1), h

def chunk_bwd_dh_torch(q, g, do, dht, h0, BT=64):
    """Torch implementation of chunk_bwd_dh"""
    B, T, HQ, K = q.shape
    V = do.shape[-1]
    H = g.shape[2] if g is not None else HQ
    NG = HQ // H
    NT = (T + BT - 1) // BT
    
    dh_list = []
    if dht is not None:
        dh = dht.clone().float().view(B, HQ, K, V)
    else:
        dh = torch.zeros(B, HQ, K, V, dtype=torch.float32, device=q.device)
    
    for i_t in range(NT - 1, -1, -1):
        t_start = i_t * BT
        t_end = min((i_t + 1) * BT, T)
        
        dh_list.insert(0, dh.clone())
        
        q_block = q[:, t_start:t_end, :, :].float()
        do_block = do[:, t_start:t_end, :, :].float()
        
        if g is not None:
            i_h = torch.arange(HQ, device=q.device) // NG
            g_block = g[:, t_start:t_end, i_h, :].float()
            do_block = do_block * torch.exp(g_block)
        
        for b in range(B):
            for hq_idx in range(HQ):
                dh[b, hq_idx, :, :] = dh[b, hq_idx, :, :] + torch.matmul(q_block[b, :, hq_idx, :].t(), do_block[b, :, hq_idx, :])
    
    dh0 = dh.clone() if h0 is not None else None
    return torch.stack(dh_list, dim=1), dh0

def chunk_local_cumsum_torch(s, BT=64, reverse=False):
    """Torch implementation of chunk_local_cumsum"""
    B, T, H, S = s.shape
    NT = (T + BT - 1) // BT
    
    o = torch.zeros_like(s)
    
    for i_t in range(NT):
        t_start = i_t * BT
        t_end = min((i_t + 1) * BT, T)
        
        block = s[:, t_start:t_end, :, :].float()
        
        if reverse:
            cumsum_block = torch.cumsum(block.flip(1), dim=1).flip(1)
        else:
            cumsum_block = torch.cumsum(block, dim=1)
        
        o[:, t_start:t_end, :, :] = cumsum_block.to(s.dtype)
    
    return o

def chunk_gsa_bwd_k_kernel_dA_torch(v, g, do, BT=64, BC=16, BV=64, scale=1.0):
    """Torch implementation of chunk_gsa_bwd_k_kernel_dA"""
    B, T, H, V_val = v.shape
    HQ = do.shape[2]
    NG = HQ // H
    NC = (BT + BC - 1) // BC
    NV = (V_val + BV - 1) // BV
    NT = (T + BT - 1) // BT
    
    dA = torch.zeros(B, T, HQ, BT, dtype=torch.float32, device=v.device)
    
    for i_b in range(B):
        for i_hq in range(HQ):
            i_h = i_hq // NG
            
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                
                for i_i in range(NC):
                    i_start = i_i * BC
                    i_end_actual = min((i_i + 1) * BC, t_end - t_start)
                    
                    for i_j in range(NC):
                        j_start = i_j * BC
                        j_end_actual = min((i_j + 1) * BC, t_end - t_start)
                        
                        if i_i > i_j:
                            for ii in range(i_end_actual):
                                for jj in range(j_end_actual):
                                    t_i = t_start + i_start + ii
                                    t_j = t_start + j_start + jj
                                    
                                    # Check boundary: t_i and t_j must be within current chunk [t_start, t_end)
                                    if t_i >= t_end or t_j >= t_end:
                                        continue
                                    
                                    # Check boundary and lower triangular condition
                                    if t_i < T and t_j < T and j_start + jj < BT:
                                        # dA is lower triangular based on chunk-relative positions
                                        # i = t_i % BT = i_start + ii (relative position in chunk)
                                        # j = j_start + jj (relative position in chunk)
                                        i_rel = i_start + ii
                                        j_rel = j_start + jj
                                        
                                        if i_rel >= j_rel:
                                            do_val = do[i_b, t_i, i_hq, :].float()
                                            v_val = v[i_b, t_j, i_h, :].float()
                                            g_i = g[i_b, t_i, i_h, :].float()
                                            g_j = g[i_b, t_j, i_h, :].float()
                                            
                                            g_n = g_i[-1] if len(g_i) > 0 else 0.0
                                            do_scaled = do_val * torch.exp(g_i - g_n)
                                            v_scaled = v_val * torch.exp(g_n - g_j)
                                            
                                            dA_ij = torch.sum(do_scaled * v_scaled) * scale
                                            dA[i_b, t_i, i_hq, j_rel] = dA_ij
                        
                        elif i_i == i_j:
                            for j in range(min(BC, t_end - t_start - j_start)):
                                t_j = t_start + j_start + j
                                
                                if t_j < T:
                                    for ii in range(i_end_actual):
                                        t_i = t_start + i_start + ii
                                        
                                        # Check boundary: t_i must be within current chunk
                                        if t_i >= t_end:
                                            continue
                                        
                                        # Only compute when i >= j (lower triangular)
                                        if t_i < T and ii >= j and j_start + j < BT:
                                            do_val = do[i_b, t_i, i_hq, :].float()
                                            v_val = v[i_b, t_j, i_h, :].float()
                                            g_i = g[i_b, t_i, i_h, :].float()
                                            g_j = g[i_b, t_j, i_h, :].float()
                                            
                                            do_scaled = do_val * scale
                                            v_scaled = v_val * torch.exp(g_i - g_j)
                                            
                                            dA_ij = torch.sum(do_scaled * v_scaled)
                                            dA[i_b, t_i, i_hq, j_start + j] = dA_ij
    
    return dA

def chunk_gsa_bwd_k_kernel_dqkvg_torch(q, k, v, h, g, do, dh, dA, dg_input=None, BT=64, BK=64, BV=64, scale=1.0):
    """Torch implementation of chunk_gsa_bwd_k_kernel_dqkvg"""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    NG = HQ // H
    NT = (T + BT - 1) // BT
    NK = (K + BK - 1) // BK
    NV = (V + BV - 1) // BV
    
    if dg_input is None:
        dg_input = torch.zeros(B, T, HQ, V, dtype=torch.float32, device=q.device)
    
    A = torch.zeros(B, T, HQ, BT, dtype=torch.float32, device=q.device)
    dq = torch.zeros(B, T, HQ, K, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, T, HQ, K, dtype=torch.float32, device=q.device)
    dv = torch.zeros(B, T, HQ, V, dtype=torch.float32, device=q.device)
    dgv = torch.zeros(B, T, HQ, V, dtype=torch.float32, device=q.device)
    
    for i_b in range(B):
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min((i_t + 1) * BT, T)
            
            for i_hq in range(HQ):
                i_h = i_hq // NG
                
                for i_k in range(NK):
                    k_start = i_k * BK
                    k_end = min((i_k + 1) * BK, K)
                    
                    # Compute A
                    q_block = q[i_b, t_start:t_end, i_hq, k_start:k_end].float()
                    k_block = k[i_b, t_start:t_end, i_h, k_start:k_end].float()
                    
                    A_block = torch.matmul(q_block * scale, k_block.t())
                    
                    for i in range(t_end - t_start):
                        for j in range(t_end - t_start):
                            if i >= j:
                                A[i_b, t_start + i, i_hq, j] = A_block[i, j]
                    
                    # Compute gradients
                    b_dq = torch.zeros(t_end - t_start, k_end - k_start, dtype=torch.float32, device=q.device)
                    b_dk = torch.zeros(t_end - t_start, k_end - k_start, dtype=torch.float32, device=q.device)
                    
                    for i_v in range(NV):
                        v_start = i_v * BV
                        v_end = min((i_v + 1) * BV, V)
                        
                        v_block = v[i_b, t_start:t_end, i_h, v_start:v_end].float()
                        g_block = g[i_b, t_start:t_end, i_h, v_start:v_end].float()
                        do_block = do[i_b, t_start:t_end, i_hq, v_start:v_end].float()
                        
                        o_t = min(t_start + BT, T) - 1
                        g_n = g[i_b, o_t, i_h, v_start:v_end].float()
                        
                        h_block = h[i_b, i_t, i_h, k_start:k_end, v_start:v_end].float()  # [BK, BV]
                        dh_block = dh[i_b, i_t, i_hq, k_start:k_end, v_start:v_end].float()  # [BK, BV]
                        
                        gv = torch.exp(g_n.unsqueeze(0) - g_block)
                        do_scaled = do_block * torch.exp(g_block)
                        
                        dg = torch.sum(h_block * dh_block, dim=0) * torch.exp(g_n)  # sum over BK dimension
                        
                        b_dq += torch.matmul(do_scaled, h_block.t()) * scale
                        b_dk += torch.matmul(v_block * gv, dh_block.t())
                        
                        b_dv = torch.matmul(k_block, dh_block) * gv
                        b_dg = dg + torch.sum(b_dv * v_block, dim=0)
                        
                        # dgv accumulation: dgv = dg_input + sum(b_dg_i for all i_k)
                        # When i_k == 0, add dg_input; always add b_dg
                        if i_k == 0:
                            # Initialize dgv with dg_input + b_dg
                            dg_input_block = dg_input[i_b, t_start:t_end, i_hq, v_start:v_end].float()
                            dgv[i_b, t_start:t_end, i_hq, v_start:v_end] = dg_input_block + b_dg.unsqueeze(0)
                        else:
                            # Accumulate b_dg
                            dgv[i_b, t_start:t_end, i_hq, v_start:v_end] += b_dg.unsqueeze(0)
                        
                        dv[i_b, t_start:t_end, i_hq, v_start:v_end] += b_dv
                    
                    # Add dA contribution
                    dA_block = dA[i_b, t_start:t_end, i_hq, :]
                    
                    dq[i_b, t_start:t_end, i_hq, k_start:k_end] = b_dq + torch.matmul(dA_block, k_block)
                    dk[i_b, t_start:t_end, i_hq, k_start:k_end] = b_dk + torch.matmul(dA_block.t(), q_block)
    
    return A, dq, dk, dv, dgv

def chunk_gsa_bwd_k_kernel_intra_dvg_torch(v, g, o, A, do, dv_in, dg_in, BT=64, BC=16, BV=64):
    """Torch implementation of chunk_gsa_bwd_k_kernel_intra_dvg"""
    B, T, H, V_val = v.shape
    HQ = do.shape[2]
    NG = HQ // H
    NC = (BT + BC - 1) // BC
    NV = (V_val + BV - 1) // BV
    NT = (T + BT - 1) // BT
    
    dv = dv_in.clone()
    dg = dg_in.clone()
    
    for i_v in range(NV):
        v_start = i_v * BV
        v_end = min((i_v + 1) * BV, V_val)
        
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min((i_t + 1) * BT, T)
            
            for i_i in range(NC):
                i_start = i_i * BC
                i_end_actual = min((i_i + 1) * BC, t_end - t_start) - i_start
                i_end_actual = max(0, i_end_actual)
                
                if i_end_actual <= 0:
                    continue
                
                for i_bh in range(B * HQ):
                    i_b = i_bh // HQ
                    i_hq = i_bh % HQ
                    i_h = i_hq // NG
                    
                    gv_block = g[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_h, v_start:v_end].float()
                    g_n = g[i_b, min(t_start + i_start + BC, t_end) - 1, i_h, v_start:v_end].float()
                    
                    b_dv = torch.zeros(BC, BV, dtype=torch.float32, device=v.device)
                    
                    for i_j in range(i_i + 1, NC):
                        j_start = i_j * BC
                        j_end = min((i_j + 1) * BC, BT, t_end - t_start)
                        j_end_actual = min(j_end - j_start, t_end - t_start - j_start)
                        j_end_actual = max(0, j_end_actual)
                        
                        if j_end_actual <= 0 or j_start >= BT:
                            continue
                        
                        g_block = g[i_b, t_start + j_start:t_start + j_start + j_end_actual, i_h, v_start:v_end].float()
                        do_block = do[i_b, t_start + j_start:t_start + j_start + j_end_actual, i_hq, v_start:v_end].float()
                        
                        # A shape is [B, T, HQ, BT], accessed as (BT, T) in GPU
                        # From GPU: p_A shape (BT, T), offsets (i_i*BC, i_t*BT + i_j*BC), block (BC, BC)
                        # block[i, j] = A[b, t_j, hq, i_i*BC + i] where t_j = i_t*BT + i_j*BC + j
                        # But we need to match GPU's dot(b_A, b_do), so build as (i_end_actual, j_end_actual)
                        A_block = torch.zeros(i_end_actual, j_end_actual, dtype=torch.float32, device=v.device)
                        for ii in range(i_end_actual):
                            j_rel = i_start + ii  # position in BT dimension
                            for jj in range(j_end_actual):
                                t_j = t_start + j_start + jj
                                if t_j < T and j_rel < BT:
                                    A_block[ii, jj] = A[i_b, t_j, i_hq, j_rel]
                        
                        m_j = (t_start + j_start + torch.arange(j_end_actual, device=v.device)) < t_end
                        m_j_expanded = m_j[:do_block.shape[0]].unsqueeze(1).float()
                        do_scaled = do_block * torch.exp(g_block - g_n.unsqueeze(0)) * m_j_expanded
                        
                        b_dv += torch.matmul(A_block, do_scaled)
                    
                    b_dv *= torch.exp(g_n.unsqueeze(0) - gv_block)
                    
                    for j in range(min(BC, t_end - t_start - i_start)):
                        t_j = t_start + i_start + j
                        j_rel = t_j - t_start
                        
                        # Load A[b, t_j, hq, i_start + arange(0, i_end_actual)]
                        for ii in range(i_end_actual):
                            t_i = t_start + i_start + ii
                            
                            if t_i < T and t_j < T and ii <= j and j_rel < BT:
                                # A[b, t_j, hq, i_start + ii]
                                A_val = A[i_b, t_j, i_hq, i_start + ii]
                                g_val = g[i_b, t_j, i_h, v_start:v_end].float()
                                do_val = do[i_b, t_j, i_hq, v_start:v_end].float()
                                
                                b_dv[ii, :] += torch.exp(g_val - gv_block[ii, :]) * A_val * do_val
                    
                    o_block = o[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end].float()
                    v_block = v[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_h, v_start:v_end].float()
                    do_block = do[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end].float()
                    dv_block = dv[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end].float()
                    
                    b_dv = b_dv + dv_block
                    b_dg = o_block * do_block - v_block * b_dv
                    
                    dv[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end] = b_dv[:i_end_actual, :]
                    dg[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end] = b_dg[:i_end_actual, :]
    
    return dv, dg

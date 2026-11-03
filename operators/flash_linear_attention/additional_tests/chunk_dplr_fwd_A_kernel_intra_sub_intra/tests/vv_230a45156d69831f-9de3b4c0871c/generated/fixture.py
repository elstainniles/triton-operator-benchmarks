"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

def fused_recurrent_rwkv4_backward_pytorch_fp64(w, u, k, v, state, grad_wkv, grad_state):
    """PyTorch CPU float64参考实现"""
    bsz, tsz, chans = k.shape
    gw = torch.zeros(bsz, chans, dtype=torch.float64, device='cpu')
    gu = torch.zeros(bsz, chans, dtype=torch.float64, device='cpu')
    gk = torch.zeros(bsz, tsz, chans, dtype=torch.float64, device='cpu')
    gv = torch.zeros(bsz, tsz, chans, dtype=torch.float64, device='cpu')
    gstate = torch.zeros(bsz, 3, 1, chans, dtype=torch.float64, device='cpu')
    for b in range(bsz):
        galpha = grad_state[b, 0, 0, :].double()
        gbeta = grad_state[b, 1, 0, :].double()
        geps = grad_state[b, 2, 0, :].double()
        for t in range(tsz - 1, -1, -1):
            kt = k[b, t, :].double()
            vt = v[b, t, :].double()
            alpha_prev = state[b, 0, t, :].double()
            beta_prev = state[b, 1, t, :].double()
            eps_prev = state[b, 2, t, :].double()
            alpha_curr = state[b, 0, t + 1, :].double()
            beta_curr = state[b, 1, t + 1, :].double()
            eps_curr = state[b, 2, t + 1, :].double()
            ukt = u + kt
            tau = torch.maximum(ukt, eps_prev)
            e1 = torch.exp(eps_prev - tau)
            e2 = torch.exp(ukt - tau)
            euke = torch.exp(ukt + eps_prev - 2 * tau)
            denom = e1 * beta_prev + e2
            denom_sq = denom * denom
            gwkvt = grad_wkv[b, t, :].double()
            guk = gwkvt * e2 * (e1 * beta_prev * vt - e1 * alpha_prev) / denom_sq
            gu[b, :] += guk
            gk[b, t, :] = guk
            gv[b, t, :] = gwkvt * e2 / denom
            galpha_wkv = gwkvt * e1 / denom
            gbeta_wkv = -gwkvt * e1 * (e2 * vt + e1 * alpha_prev) / denom_sq
            geps_wkv_denom = e1 * beta_prev + e2
            geps_wkv = gwkvt * euke * (alpha_prev - vt * beta_prev) / (geps_wkv_denom * geps_wkv_denom)
            e1 = torch.exp(w + eps_prev - eps_curr)
            e2 = torch.exp(kt - eps_curr)
            galpha_we = galpha * e1 * alpha_prev
            gw[b, :] += galpha_we
            gk[b, t, :] += galpha * e2 * vt
            gv[b, t, :] += galpha * e2
            geps += galpha * -alpha_curr
            gbeta_we = gbeta * e1 * beta_prev
            gw[b, :] += gbeta_we
            gk[b, t, :] += gbeta * e2
            geps += gbeta * -beta_curr
            geps_mask = w + eps_prev > kt
            geps_we = torch.where(geps_mask, geps, torch.zeros_like(geps))
            gw[b, :] += geps_we
            gk[b, t, :] += torch.where(geps_mask, torch.zeros_like(geps), geps)
            galpha = galpha * e1 + galpha_wkv
            gbeta = gbeta * e1 + gbeta_wkv
            geps = galpha_we + gbeta_we + geps_we + geps_wkv
        gstate[b, 0, 0, :] = galpha
        gstate[b, 1, 0, :] = gbeta
        gstate[b, 2, 0, :] = geps
    gw = gw * w
    return (gw.sum(0), gu.sum(0), gk, gv, gstate)

SEED = 20261031

def dplr_matrices(d, scale):
    q, k, a, b, gi, ge = (d[n].double() for n in ('q', 'k', 'a', 'b', 'gi', 'ge'))
    B, T, H, K = q.shape
    BT = d['chunk_size']
    NT = T // BT

    def chunk(x):
        return x.reshape(B, NT, BT, H, K).permute(0, 1, 3, 2, 4)
    q, k, a, b, gi, ge = map(chunk, (q, k, a, b, gi, ge))
    qi = q * gi.exp2() * scale
    ki = k * (-gi).exp2()
    bb = b * (-gi).exp2()
    ae = a * ge.exp2()
    values = {'Aqk': (qi @ ki.transpose(-1, -2)).tril(), 'Aqb': (qi @ bb.transpose(-1, -2)).tril(), 'Aak': (ae @ ki.transpose(-1, -2)).tril(-1), 'Aab': (ae @ bb.transpose(-1, -2)).tril(-1)}
    out = {n: x.permute(0, 1, 3, 2, 4).reshape(B, T, H, BT) for n, x in values.items()}
    last = gi[:, :, :, -1:, :]
    aux = dict(qg=qi, kg=k * (last - gi).exp2(), ag=ae, bg=b * (last - gi).exp2())
    out.update({n: x.permute(0, 1, 3, 2, 4).reshape(B, T, H, K) for n, x in aux.items()})
    return out

def build(kind):
    gen = torch.Generator().manual_seed(SEED)

    def rand(*shape, dtype=torch.float32):
        return torch.randn(shape, generator=gen, dtype=dtype)
    if kind in ('dplr_bwd', 'dplr_fwd_intra'):
        B, T, H, K, BT = (2, 256, 4, 64, 32) if kind == 'dplr_bwd' else (2, 1024, 8, 64, 64)
        dt = torch.float16
        d = {n: rand(B, T, H, K, dtype=dt) for n in ('q', 'k', 'a', 'b', 'gi', 'ge')}
        d.update(B=B, T=T, H=H, K=K, scale=K ** (-0.5), chunk_size=BT, cu_seqlens=None, chunk_indices=None, safe_gate=False)
        if kind == 'dplr_fwd_intra':
            return ({'input_data': d}, dplr_matrices(d, d['scale']))
        for n in ('dAqk', 'dAqb', 'dAak', 'dAab'):
            d[n] = rand(B, T, H, BT, dtype=dt)
        for n in ('dqg', 'dkg', 'dag', 'dbg'):
            d[n] = rand(B, T, H, K, dtype=dt)
        d['dgk_last'] = rand(B * (T // BT), H, K)
        x = {n: v.double().requires_grad_(n in ('q', 'k', 'a', 'b')) if isinstance(v, torch.Tensor) else v for n, v in d.items()}
        ys = dplr_matrices(x, 1.0)
        loss = sum(((ys[n] * x['d' + n]).sum() for n in ('Aqk', 'Aqb', 'Aak', 'Aab'))) + sum(((ys[n] * x['d' + n] * (d['scale'] if n == 'qg' else 1.0)).sum() for n in ('qg', 'kg', 'ag', 'bg')))
        grads = torch.autograd.grad(loss, tuple((x[n] for n in ('q', 'k', 'a', 'b'))))
        out = {'d' + n: v.detach() for n, v in zip(('q', 'k', 'a', 'b'), grads)}
        raw = out['dq'] * x['q'] + out['da'] * x['a'] - out['dk'] * x['k'] - out['db'] * x['b']
        offset = out['da'] * x['a']
        raw = raw.detach()
        offset = offset.detach()
        out['dgk_raw'] = raw
        out['dgk_offset'] = offset
        out['dgk_output'] = raw.reshape(B, T // BT, BT, H, K).flip(2).cumsum(2).flip(2).reshape(B, T, H, K) + x['dgk_last'].reshape(B, T // BT, 1, H, K).expand(B, T // BT, BT, H, K).reshape(B, T, H, K) - offset
        return ({'input_data': d}, out)
    if kind == 'iplr_fwd':
        B, T, H, K, V = (1, 512, 8, 16, 16)
        d = {n: rand(B, T, H, V if n == 'v' else K) * 0.1 for n in ('q', 'k', 'v', 'a', 'b')}
        d.update(scale=K ** (-0.5), initial_state=None, output_final_state=False, cu_seqlens=None, B=B, T=T, H=H, K=K, V=V)
        state = torch.zeros(B, H, K, V, dtype=torch.float64)
        out = []
        ha = []
        for t in range(T):
            av = torch.einsum('bhk,bhkv->bhv', d['a'][:, t].double(), state)
            ha.append(av)
            state += d['b'][:, t].double().unsqueeze(-1) * av.unsqueeze(-2) + d['k'][:, t].double().unsqueeze(-1) * d['v'][:, t].double().unsqueeze(-2)
            out.append(torch.einsum('bhk,bhkv->bhv', d['q'][:, t].double(), state) * d['scale'])
        return ({'input_data': d}, {'o': torch.stack(out, 1), 'ha': torch.stack(ha, 1), 'final_state': None})
    if kind == 'naive_attn':
        N, T, H, HQ, K, V = (1, 512, 8, 16, 16, 16)
        dt = torch.bfloat16
        q = rand(1, N, HQ, K, dtype=dt) * 0.1
        k = rand(1, T, H, K, dtype=dt) * 0.1
        v = rand(1, T, H, V, dtype=dt) * 0.1
        g = rand(1, T, HQ, dtype=dt) * 0.1
        gc = g.float().cumsum(1)
        cu = torch.tensor([0, T], dtype=torch.int32)
        d = dict(q=q, k=k, v=v, g=g, g_cumsum=gc, cu_seqlens=cu, scale=K ** (-0.5), sink_bias=None, N=N, T=T, H=H, HQ=HQ, G=HQ // H, K=K, V=V)
        kk = k.double().repeat_interleave(HQ // H, 2)[0]
        vv = v.double().repeat_interleave(HQ // H, 2)[0]
        s = torch.einsum('hk,thk->th', q.double()[0, 0], kk) * d['scale'] + g.double().cumsum(1)[0, -1:].expand(T, HQ) - g.double().cumsum(1)[0]
        o = torch.einsum('th,thv->hv', s.softmax(0), vv)[None, None]
        return ({'input_data': d}, {'o': o})
    if kind == 'rwkv4_bwd':
        B, T, C = (4, 2048, 512)
        d = dict(w_processed=-rand(C).exp(), u=rand(C), k=rand(B, T, C), v=rand(B, T, C), state=torch.zeros(B, 3, T + 1, C), grad_wkv=rand(B, T, C), grad_state=rand(B, 3, 1, C), bsz=B, tsz=T, chans=C)
        args = [d[k].double() for k in ('w_processed', 'u', 'k', 'v', 'state', 'grad_wkv', 'grad_state')]
        ys = fused_recurrent_rwkv4_backward_pytorch_fp64(*args)
        return (d, {k: v for k, v in zip(('gw', 'gu', 'gk', 'gv', 'gstate'), ys)})
    raise ValueError(kind)

def make_fixtures():
    inputs, reference = build('dplr_fwd_intra')
    values = {'test_chunk_dplr_fwd_A_kernel_intra_sub_intra_cuda.pt': inputs}
    values['reference_cpu64.pt'] = reference
    return values

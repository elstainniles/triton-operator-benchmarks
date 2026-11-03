"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

SEED = 20261031

def delta_backward(d):
    x = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    q, k, v, beta, dy = (x['q'], x['k'], x['v'], x['beta'], x['do'])
    B, T, H, K = q.shape
    V = v.shape[-1]
    dh = x['dht'].clone()
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    db = torch.empty_like(beta)
    dq = torch.empty_like(q)
    for t in range(T - 1, -1, -1):
        dh = dh + q[:, t].unsqueeze(-1) * dy[:, t].unsqueeze(-2) * d['scale']
        dk[:, t] = torch.einsum('bhkv,bhv->bhk', dh, v[:, t] * beta[:, t].unsqueeze(-1))
        du = torch.einsum('bhkv,bhk->bhv', dh, k[:, t])
        db[:, t] = (du * v[:, t]).sum(-1)
        dv[:, t] = du * beta[:, t].unsqueeze(-1)
        dh = dh - k[:, t].unsqueeze(-1) * dv[:, t].unsqueeze(-2)
    state = x['initial_state'].clone()
    for t in range(T):
        dk[:, t] -= torch.einsum('bhkv,bhv->bhk', state, dv[:, t])
        state = state + k[:, t].unsqueeze(-1) * (v[:, t] * beta[:, t].unsqueeze(-1)).unsqueeze(-2)
        dq[:, t] = torch.einsum('bhkv,bhv->bhk', state, dy[:, t]) * d['scale']
    return dict(dq=dq, dk=dk, dv=dv, db=db, dh0=dh)

def build(kind):
    gen = torch.Generator().manual_seed(SEED)

    def rand(*shape, dtype=torch.float32):
        return torch.randn(shape, generator=gen, dtype=dtype)

    def uniform(*shape, dtype=torch.float32):
        return torch.rand(shape, generator=gen, dtype=dtype)
    if kind == 'delta_fwd':
        B, T, H, K, V = (4, 128, 8, 128, 128)
        q = torch.nn.functional.normalize(rand(B, T, H, K), dim=-1)
        k = torch.nn.functional.normalize(rand(B, T, H, K), dim=-1)
        v = rand(B, T, H, V)
        beta = uniform(B, T, H).sigmoid()
        h0 = rand(B, H, K, V)
        d = dict(q=q, k=k, v=v, beta=beta, scale=K ** (-0.5), initial_state=h0, output_final_state=True)
        h = h0.double()
        outs = []
        us = []
        for t in range(T):
            residual = v[:, t].double() - torch.einsum('bhk,bhkv->bhv', k[:, t].double(), h)
            us.append(residual)
            h = h + k[:, t].double().unsqueeze(-1) * (residual * beta[:, t].double().unsqueeze(-1)).unsqueeze(-2)
            outs.append(torch.einsum('bhk,bhkv->bhv', q[:, t].double(), h) * d['scale'])
        return (d, dict(o=torch.stack(outs, 1), u=torch.stack(us, 1), final_state=h))
    if kind == 'delta_bwd':
        B, T, H, D = (3, 2000, 4, 128)
        dt = torch.float16
        d = dict(q=torch.nn.functional.normalize(rand(B, T, H, D, dtype=dt), dim=-1), k=torch.nn.functional.normalize(rand(B, T, H, D, dtype=dt), dim=-1), v=rand(B, T, H, D, dtype=dt), beta=rand(B, T, H, dtype=dt).sigmoid(), do=uniform(B, T, H, D, dtype=dt), dht=uniform(B, H, D, D), initial_state=rand(B, H, D, D), scale=0.1, use_qk_l2norm_in_kernel=False)
        return (d, delta_backward(d))
    if kind == 'dh_reduction':
        B, T, H, HQ, K, V, BT = (4, 2048, 8, 8, 128, 128, 64)
        NT = T // BT
        g = torch.nn.functional.logsigmoid(rand(B, T, H)).reshape(B, NT, BT, H).cumsum(2).reshape(B, T, H)
        dh = rand(B, NT, HQ, K, V)
        d = dict(g_cumsum=g, dh=dh, doq0=None, dh0_input=None, h0=None, B=B, T=T, H=H, HQ=HQ, K=K, V=V, BT=BT, NT=NT, NG=1, scale=K ** (-0.5))
        out = dh.double().clone()
        acc = torch.zeros_like(out[:, 0])
        for t in range(NT - 1, -1, -1):
            acc += dh[:, t].double()
            if t < NT - 1:
                out[:, t] = acc
            acc *= g[:, (t + 1) * BT - 1].double().exp()[..., None, None]
        return ({'input_data': d}, {'dh': out, 'dh0': None})
    if kind == 'gla_intra':
        B, T, H, K, BT = (2, 2048, 8, 128, 64)
        q = rand(B, T, H, K, dtype=torch.float16)
        k = rand(B, T, H, K, dtype=torch.float16)
        g = rand(B, T, H, K, dtype=torch.float16)
        out = torch.empty(B, T, H, BT, dtype=torch.float64)
        for t in range(0, T, BT):
            qq = (q[:, t:t + BT].double() * g[:, t:t + BT].double().exp()).transpose(1, 2)
            kk = (k[:, t:t + BT].double() * (-g[:, t:t + BT].double()).exp()).transpose(1, 2)
            out[:, t:t + BT] = (qq @ kk.transpose(-1, -2)).tril().transpose(1, 2) * K ** (-0.5)
        return ({'q.pt': q, 'k.pt': k, 'g.pt': g}, {'A': out})
    if kind == 'copy_last':
        B, T, H, K, V, L, BT = (2, 1024, 8, 128, 64, 7, 64)
        dt = torch.float16
        d = dict(q=rand(B, T, 1, K, dtype=dt), k=rand(B, T, 1, K, dtype=dt), v=rand(B, T, H, V, dtype=dt), g=rand(B, T, H, dtype=dt), level_scales=rand(B, T, H, L, dtype=dt), q_prev=torch.zeros(B, BT, 1, K, dtype=dt), k_prev=torch.zeros(B, BT, 1, K, dtype=dt), v_prev=torch.zeros(B, BT, H, V, dtype=dt), g_prev=torch.zeros(B, BT, H, dtype=dt), level_scales_prev=torch.zeros(B, BT, H, L, dtype=dt), offsets=torch.zeros(B, dtype=torch.int32), B=B, T=T, G=1, H=H, K=K, V=V, L=L, BT=BT)
        return ({'input_data': d}, {'defined_source_rows': 0, 'reason': 'Original T=1024,BT=64 gives seq_offset=T. Every block load is outside T with no padding_option. Numeric expected outputs are undefined; no fabricated zero reference.', 'source_row_valid': torch.zeros(BT, dtype=torch.bool)})
    if kind == 'iplr_bwd':
        B, T, H, K, V = (2, 1024, 8, 64, 64)
        dt = torch.float16
        d = {n: rand(B, T, H, V if n in ('v', 'do', 'ha') else K, dtype=torch.float32 if n == 'ha' else dt) for n in ('q', 'k', 'v', 'a', 'b', 'ha', 'do')}
        d['a'] *= 0.05
        d['b'] *= 0.05
        d.update(dht=None, initial_state=None, scale=K ** (-0.5), cu_seqlens=None, B=B, T=T, H=H, K=K, V=V)
        x = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        dh = torch.zeros(B, H, K, V, dtype=torch.float64)
        dq = torch.empty_like(x['q'])
        dk = torch.empty_like(dq)
        da = torch.empty_like(dq)
        db = torch.empty_like(dq)
        dv = torch.empty_like(x['v'])
        dha = torch.empty_like(dv)
        for t in range(T - 1, -1, -1):
            dh += x['q'][:, t].unsqueeze(-1) * x['do'][:, t].unsqueeze(-2) * d['scale']
            dk[:, t] = torch.einsum('bhkv,bhv->bhk', dh, x['v'][:, t])
            dv[:, t] = torch.einsum('bhkv,bhk->bhv', dh, x['k'][:, t])
            dha[:, t] = torch.einsum('bhkv,bhk->bhv', dh, x['b'][:, t])
            db[:, t] = torch.einsum('bhkv,bhv->bhk', dh, x['ha'][:, t])
            dh += x['a'][:, t].unsqueeze(-1) * dha[:, t].unsqueeze(-2)
        state = torch.zeros_like(dh)
        for t in range(T):
            da[:, t] = torch.einsum('bhkv,bhv->bhk', state, dha[:, t])
            state += x['k'][:, t].unsqueeze(-1) * x['v'][:, t].unsqueeze(-2) + x['b'][:, t].unsqueeze(-1) * x['ha'][:, t].unsqueeze(-2)
            dq[:, t] = torch.einsum('bhkv,bhv->bhk', state, x['do'][:, t]) * d['scale']
        return ({'input_data': d}, {'dq': dq, 'dk': dk, 'dv': dv, 'da': da, 'db': db, 'dha': dha, 'dh0': None, 'input_distribution_note': 'a and b are fixed-seed random FP16 scaled by 0.05 for finite recurrent behavior; shape/dtype/tile unchanged'})
    if kind == 'mesa':
        B, H, K, V = (2, 4, 60, 60)
        z = rand(B, H, K, K)
        Q, _ = torch.linalg.qr(z)
        eigen = 0.5 + 4.5 * uniform(B, H, K)
        hkk = Q * eigen.unsqueeze(-2) @ Q.transpose(-1, -2)
        d = dict(q=uniform(B, H, K), k=uniform(B, H, K), v=uniform(B, H, V), g=uniform(B, H), lamb=uniform(H, K) + 0.1, beta=uniform(B, H), prev_h_kk=hkk, prev_h_kv=uniform(B, H, K, V), o=torch.full((B, H, V), float('nan')), curr_h_kk=torch.full_like(hkk, float('nan')), curr_h_kv=torch.full((B, H, K, V), float('nan')), B=B, H=H, K=K, V=V, BK=64, BV=64, MAX_CG_STEP=5)
        x = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        kk = x['prev_h_kk'] * x['g'].exp()[..., None, None] + x['k'].unsqueeze(-1) * x['k'].unsqueeze(-2) * x['beta'][..., None, None]
        kv = x['prev_h_kv'] * x['g'].exp()[..., None, None] + x['k'].unsqueeze(-1) * x['v'].unsqueeze(-2) * x['beta'][..., None, None]
        xx = x['q'] / (kk.diagonal(dim1=-2, dim2=-1) + x['lamb'] + 1e-05)
        rr = x['q'] - torch.einsum('bhk,bhkv->bhv', xx, kk) - x['lamb'] * xx
        pp = rr.clone()
        delta = rr.square().sum(-1)
        for _ in range(5):
            ap = torch.einsum('bhk,bhkv->bhv', pp, kk) + x['lamb'] * pp
            alpha = delta / ((pp * ap).sum(-1) + 1e-05)
            xx = xx + alpha[..., None] * pp
            rr = rr - alpha[..., None] * ap
            new = rr.square().sum(-1)
            pp = rr + (new / (delta + 1e-05))[..., None] * pp
            delta = new
        return ({'input_data': d, 'grid': (B * H,)}, {'o': torch.einsum('bhk,bhkv->bhv', xx, kv), 'curr_h_kk': kk, 'curr_h_kv': kv})
    if kind == 'prepare_wy':
        B, T, H, BT = (4, 2048, 16, 32)
        a = rand(B, T, H, BT)
        blocks = a.double().reshape(B, T // BT, BT, H, BT).permute(0, 1, 3, 2, 4).tril(-1)
        eye = torch.eye(BT, dtype=torch.float64).expand_as(blocks)
        inv = torch.linalg.solve_triangular(eye - blocks, eye, upper=False, unitriangular=True).permute(0, 1, 3, 2, 4).reshape_as(a)
        return (dict(A_ab=a, cu_seqlens=None, chunk_size=BT), {'A_ab_inv': inv})
    raise ValueError(kind)

def _make_fixtures():
    inputs, reference = build('mesa')
    values = {'test_mesa_net_decoding_one_step_kernel.pt': inputs}
    values['reference_cpu64.pt'] = reference
    return values

def make_fixtures():
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        return _make_fixtures()
    finally:
        torch.set_num_threads(previous_threads)

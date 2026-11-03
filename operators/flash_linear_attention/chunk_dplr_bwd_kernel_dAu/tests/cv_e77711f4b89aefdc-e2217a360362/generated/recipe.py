"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_fixtures():
    d = inputs('dau')
    r = reference('dau', d)
    return {'test_chunk_dplr_bwd_kernel_dAu_cuda.pt': dict(input_data=d, gpu_output=r, reference_origin='CPU float64 independent math; no GPU execution')}
SEED = 20261031

def inputs(kind):
    g = torch.Generator().manual_seed(SEED)

    def rand(*shape, dtype=torch.float16):
        return (torch.randn(shape, generator=g, dtype=torch.float32) * 0.1).to(dtype)
    if kind == 'dau':
        B, T, H, V, BT = (2, 512, 8, 128, 64)
        return dict(v=rand(B, T, H, V), v_new=rand(B, T, H, V), do=rand(B, T, H, V), A_qb=rand(B, T, H, BT), scale=V ** (-0.5), chunk_size=BT, cu_seqlens=None, chunk_indices=None, B=B, T=T, H=H, V=V)
    B, T, H, K, V, BT = (1, 128, 2, 32, 32, 16)
    d = {n: rand(B, T, H, V if n in ('do', 'dv') else K) for n in ('qg', 'bg', 'w', 'gk', 'do', 'dv')}
    d.update(h0=rand(B, H, K, V, dtype=torch.float32), dht=rand(B, H, K, V, dtype=torch.float32), chunk_size=BT, cu_seqlens=None, chunk_indices=None, B=B, T=T, H=H, K=K, V=V)
    return d

def reference(kind, d):
    x = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    B, T, H, BT = (d['B'], d['T'], d['H'], d['chunk_size'])
    N = T // BT

    def blocks(k):
        return x[k].reshape(B, N, BT, H, -1).permute(0, 1, 3, 2, 4)

    def flatten(z):
        return z.permute(0, 1, 3, 2, 4).reshape(B, T, H, -1)
    if kind == 'dau':
        v, vn, do, A = map(blocks, ('v', 'v_new', 'do', 'A_qb'))
        A = A.tril()
        return {'dv_new': flatten(A.transpose(-1, -2) @ do).half(), 'dA_qk': flatten((do @ v.transpose(-1, -2) * d['scale']).tril()).float(), 'dA_qb': flatten((do @ vn.transpose(-1, -2) * d['scale']).tril()).float()}
    dh = torch.empty(B, N, H, d['K'], d['V'], dtype=torch.float16)
    dv2 = torch.empty_like(d['dv'])
    state = x['dht'].clone()
    for c in range(N - 1, -1, -1):
        sl = slice(c * BT, (c + 1) * BT)
        dh[:, c] = state.half()
        q = x['qg'][:, sl].transpose(1, 2)
        bg = x['bg'][:, sl].transpose(1, 2)
        w = x['w'][:, sl].transpose(1, 2)
        do = x['do'][:, sl].transpose(1, 2)
        dv = x['dv'][:, sl].transpose(1, 2)
        d2 = dv + bg @ state.half().double()
        dv2[:, sl] = d2.transpose(1, 2).half()
        state = state * x['gk'][:, (c + 1) * BT - 1].exp2().unsqueeze(-1) + q.transpose(-1, -2) @ do + w.transpose(-1, -2) @ d2.half().double()
    return {'dh': dh, 'dh0': state.float(), 'dv2': dv2}

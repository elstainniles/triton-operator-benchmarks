"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

SEED = 20261031

def inputs(kind):
    gen = torch.Generator().manual_seed(SEED)

    def rand(*shape):
        return torch.randn(shape, generator=gen, dtype=torch.float32)
    if kind == 'rwkv4':
        B, T, C = (4, 2048, 512)
        return dict(w_processed=-rand(C).exp(), u=rand(C), k=rand(B, T, C), v=rand(B, T, C), state=torch.zeros(B, 3, 1, C), bsz=B, tsz=T, chans=C)
    if kind == 'dplr_dgk':
        B, T, H, K = (2, 256, 4, 64)
        return dict(dgk=rand(B, T, H, K), dgk_offset=rand(B, T, H, K), dgk_last=rand(B * (T // 32), H, K), cu_seqlens=None, chunk_size=32, chunk_indices=None, B=B, T=T, H=H, K=K)
    B, T, H, K, V = (1, 256, 4, 16, 16) if kind == 'dplr_fwd' else (2, 256, 4, 16, 16)
    names = ('q', 'k', 'v', 'a', 'b', 'gk') if kind == 'dplr_fwd' else ('r', 'w', 'k', 'v', 'kk', 'a')
    d = {n: rand(B, T, H, V if n == 'v' else K) * 0.1 for n in names}
    d.update(B=B, T=T, H=H, K=K, V=V, scale=0.125 if kind == 'dplr_fwd' else 1.0, initial_state=rand(B, H, K, V) * 0.1 if kind == 'dplr_fwd' else None, output_final_state=True, reverse=False, cu_seqlens=None)
    return d

def reference(kind, d):
    if kind == 'rwkv4':
        x = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        alpha, beta, eps = x['state'][:, :, 0].unbind(1)
        outs = []
        states = []
        for t in range(d['tsz']):
            k, v = (x['k'][:, t], x['v'][:, t])
            tau = torch.maximum(x['u'] + k, eps)
            e1, e2 = ((eps - tau).exp(), (x['u'] + k - tau).exp())
            outs.append((e1 * alpha + e2 * v) / (e1 * beta + e2))
            new_eps = torch.maximum(x['w_processed'] + eps, k)
            e1 = (x['w_processed'] + eps - new_eps).exp()
            e2 = (k - new_eps).exp()
            alpha, beta, eps = (e1 * alpha + e2 * v, e1 * beta + e2, new_eps)
            states.append(torch.stack((alpha, beta, eps), 1))
        return {'wkv': torch.stack(outs, 1).float(), 'state_out_raw': torch.stack(states, 2).float()}
    if kind == 'dplr_dgk':
        B, T, H, K = d['dgk'].shape
        BT = d['chunk_size']
        x = d['dgk'].double().reshape(B, T // BT, BT, H, K)
        y = x.flip(2).cumsum(2).flip(2) + d['dgk_last'].double().reshape(B, T // BT, 1, H, K) - d['dgk_offset'].double().reshape_as(x)
        return {'dgk_output': y.reshape(B, T, H, K).float()}
    x = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    h = x['initial_state'] if x['initial_state'] is not None else torch.zeros(d['B'], d['H'], d['K'], d['V'], dtype=torch.float64)
    outs = []
    for t in range(d['T']):
        if kind == 'dplr_fwd':
            q = x['q'][:, t]
            a = x['a'][:, t]
            b = x['b'][:, t]
            decay = x['gk'][:, t].exp()
        else:
            q = x['r'][:, t]
            a = -x['kk'][:, t]
            b = x['kk'][:, t] * x['a'][:, t]
            decay = x['w'][:, t].exp()
        transition = torch.diag_embed(decay) + b.unsqueeze(-1) @ a.unsqueeze(-2)
        h = transition @ h + x['k'][:, t].unsqueeze(-1) @ x['v'][:, t].unsqueeze(-2)
        outs.append((q.unsqueeze(-2) @ h).squeeze(-2) * d['scale'])
    return {'o': torch.stack(outs, 1).float(), 'final_state' if kind == 'dplr_fwd' else 'ht': h.float()}

def make_fixtures():
    data = inputs('dplr_dgk')
    expected = reference('dplr_dgk', data)
    return {'test_chunk_dplr_bwd_dgk_kernel_cuda.pt': dict(input_data=data, gpu_output=expected, reference_origin='synthetic_cpu_float64_math_not_GPU')}

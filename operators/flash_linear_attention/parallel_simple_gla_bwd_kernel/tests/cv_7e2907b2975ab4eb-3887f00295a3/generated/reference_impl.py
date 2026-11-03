"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def heads(x):
    return x.double().permute(0, 2, 1, 3)

def unheads(x):
    return x.permute(0, 2, 1, 3).contiguous()

def reference(cid, d):
    checks = []
    notes = {}
    assert d['cu_seqlens'] is None and d['chunk_indices'] is None
    q, k, v, do = (heads(d[n]) for n in ('q', 'k', 'v', 'do'))
    bsz, heads_n, length, width = q.shape
    bt = d['chunk_size']
    local = d['g'].double().reshape(bsz, length // bt, bt, heads_n)
    ends = local[:, :, -1]
    prefix = ends.cumsum(1) - ends
    global_g = (local + prefix[:, :, None]).reshape(bsz, length, heads_n).permute(0, 2, 1)
    decay = torch.exp2(global_g[..., :, None] - global_g[..., None, :]).tril()
    ds = do @ v.transpose(-1, -2) * decay * d['scale']
    dq, dk = (ds @ k, ds.transpose(-1, -2) @ q)
    dv = (q @ k.transpose(-1, -2) * decay * d['scale']).transpose(-1, -2) @ do
    dg = (dq * q - dk * k).sum(-1)
    qa, ka, va = (x.detach().requires_grad_() for x in (q, k, v))
    ga = global_g.detach().requires_grad_()
    weight = (qa @ ka.transpose(-1, -2) * torch.exp2(ga[..., :, None] - ga[..., None, :])).tril() * d['scale']
    auto = torch.autograd.grad((weight @ va * do).sum(), (qa, ka, va, ga))
    checks.extend((float((actual - gold).abs().max()) for actual, gold in zip((dq, dk, dv, dg * math.log(2)), auto)))
    out = {'dq': unheads(dq), 'dk': unheads(dk), 'dv': unheads(dv), 'dg': dg.permute(0, 2, 1).contiguous()}
    method = 'Independent CPU64 causal gated linear attention derivatives; reconstruct global cumulative gate from original chunk-local g; full autograd cross-check'
    notes.update(autograd_full_outputs_checked=True, original_absolute_tolerance=1e-05, actual_output_dtype='float32')
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert checks and max(checks) < 1e-08, ('cv_7e2907b2975ab4eb', checks)
    return (out, {'method': method, 'scalar_checked_values': len(checks), 'scalar_max_abs_difference': max(checks), **notes})

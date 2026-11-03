"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(cid, d):
    bt, s = (d['BT'], d['S'])
    b, t, h, k = d['k'].shape
    assert t % s == s % bt == 0
    ns = t // s
    nc = s // bt

    def blocks(value, heads=h):
        return value.double().reshape(b, ns, nc, bt, heads, k).permute(0, 1, 4, 2, 3, 5)

    def unblocks(value):
        return value.permute(0, 1, 3, 4, 2, 5).reshape(b, t, value.shape[2], k)
    w1, w2, key = (blocks(d[n]) for n in ['w1', 'w2', 'k'])
    identity = torch.eye(k, dtype=torch.float64)
    errors = []
    residual = identity.expand(b, ns, h, k, k).clone()
    suffix = torch.empty(b, ns, h, nc, k, k, dtype=torch.float64)
    result = torch.empty_like(key)
    state = torch.zeros_like(residual)
    for j in range(nc - 1, -1, -1):
        suffix[:, :, :, j] = identity - residual
        result[:, :, :, j] = key[:, :, :, j] @ residual.transpose(-1, -2)
        residual = residual @ (identity - w1[:, :, :, j].transpose(-1, -2) @ w2[:, :, :, j])
        state = state + (w1[:, :, :, j].transpose(-1, -2) - state @ w1[:, :, :, j].transpose(-1, -2)) @ w2[:, :, :, j]
        errors.append(float((state - (identity - residual)).abs().max()))
    out = {'k_new': unblocks(result), 'hc_suffix': suffix.permute(0, 1, 3, 2, 4, 5).reshape(b * ns * nc, h, k, k), 'hc_whole': (identity - residual).reshape(b * ns, h, k, k)}
    method = 'CPU64 product of residual matrices I-w1T@w2 in reverse chunk order, cross-checked independent additive state recurrence; original bf16 inputs and outputs retained'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'independent_cross_checks': len(errors), 'max_cross_check_error': max(errors)})

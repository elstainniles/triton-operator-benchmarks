"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
import torch

def reference(cid, d):
    errors = []
    bt = d['BT']
    k, v = (tile(d['k'], bt), tile(d['v'], bt))
    beta, g = (scalar_tile(d['beta'], bt), scalar_tile(d['g'], bt))
    state = d['h_init'].double().clone()
    cross = d['h_kv_init'].double().clone()
    saved = []
    saved_cross = []
    for i in range(k.shape[1]):
        if i % (d['BS'] // bt) == 0:
            saved.append(state.clone())
            saved_cross.append(cross.clone())
        gate = g[:, i]
        last = gate[..., -1]
        weighted = k[:, i] * (beta[:, i] * torch.exp(last[..., None] - gate))[..., None]
        old = state
        state = state * torch.exp(last)[..., None, None] + weighted.transpose(-1, -2) @ k[:, i]
        cross = cross * torch.exp(last)[..., None, None] + weighted.transpose(-1, -2) @ v[:, i]
        if i == 0:
            for ki, vi in [(0, 0), (1, 1), (k.shape[-1] - 1, k.shape[-1] - 1)]:
                expected = float(old[0, 0, ki, vi]) * math.exp(float(last[0, 0])) + math.fsum((float(k[0, i, 0, j, ki]) * float(k[0, i, 0, j, vi]) * float(beta[0, i, 0, j]) * math.exp(float(last[0, 0] - gate[0, 0, j])) for j in range(bt)))
                errors.append(abs(float(state[0, 0, ki, vi]) - expected))
    out = {'h': torch.stack(saved, dim=1), 'h_kv': torch.stack(saved_cross, dim=1), 'h_final': state, 'h_kv_final': cross}
    method = 'Independent CPU64 decayed key-key and key-value state recurrence; preserve all original split snapshots and both final states'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})

"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
import torch

def reference(cid, d):
    errors = []
    bt = d['chunk_size']
    q, do = (tile(d['q'], bt), tile(d['do'], bt))
    state = d['dht'].double().clone()
    saved = [None] * q.shape[1]
    for i in range(q.shape[1] - 1, -1, -1):
        saved[i] = state.clone()
        state = state + q[:, i].transpose(-1, -2) @ do[:, i] * d['scale']
    out = {'dh': torch.stack(saved, dim=1), 'dh0': state}
    expected = float(d['dht'][0, 0, 0, 0]) + math.fsum((float(d['q'][0, t, 0, 0]) * float(d['do'][0, t, 0, 0]) * d['scale'] for t in range(d['q'].shape[1])))
    errors.append(abs(float(state[0, 0, 0, 0]) - expected))
    method = 'CPU64 reverse qT@do*scale state accumulation, all chunk snapshots and initial-state gradient'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})

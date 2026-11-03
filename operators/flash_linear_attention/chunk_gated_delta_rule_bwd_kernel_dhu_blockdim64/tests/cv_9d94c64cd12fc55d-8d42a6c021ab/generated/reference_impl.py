"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    bt = d.get('BT', d.get('chunk_size'))
    errors = []
    q, k, w, do, dv = (tile(d[n], bt) for n in ['q', 'k', 'w', 'do', 'dv'])
    b, n, h, _, width = k.shape
    vd = dv.shape[-1]
    state = torch.zeros((b, h, width, vd), dtype=torch.float64) if d.get('dht') is None else d['dht'].double().clone()
    gate = scalar_tile(d['g'], bt) if d.get('g') is not None else None
    exp = torch.exp2
    saved = [None] * n
    values = [None] * n
    for chunk in range(n - 1, -1, -1):
        saved[chunk] = state.clone()
        new = k[:, chunk] @ state
        if gate is not None:
            new = new * exp(gate[:, chunk, :, -1, None] - gate[:, chunk])[..., None]
        new = new + dv[:, chunk]
        values[chunk] = new
        previous = state
        query = q[:, chunk]
        if gate is not None:
            state = state * exp(gate[:, chunk, :, -1])[..., None, None]
            query = query * exp(gate[:, chunk])[..., None]
        state = state + query.transpose(-1, -2) @ do[:, chunk] * d['scale'] - w[:, chunk].transpose(-1, -2) @ new
        if chunk == n - 1:
            for ki, vi in [(0, 0), (1, 1), (width - 1, vd - 1)]:
                expected = float(previous[0, 0, ki, vi])
                if gate is not None:
                    expected *= float(exp(gate[0, chunk, 0, -1]))
                expected += math.fsum((float(query[0, 0, t, ki]) * float(do[0, chunk, 0, t, vi]) * d['scale'] - float(w[0, chunk, 0, t, ki]) * float(new[0, 0, t, vi]) for t in range(bt)))
                errors.append(abs(float(state[0, 0, ki, vi]) - expected))
    out = {'dh': torch.stack(saved, dim=1), 'dv2': untile(torch.stack(values, dim=1))}
    if d.get('h0') is not None:
        out['dh0'] = state
    method = 'CPU64 reverse recurrence: dv2=dv+gated(k@dh_next); dh_prev=decay*dh_next+gated(q)^T@do*scale-w^T@dv2, original exp/exp2 choice'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})

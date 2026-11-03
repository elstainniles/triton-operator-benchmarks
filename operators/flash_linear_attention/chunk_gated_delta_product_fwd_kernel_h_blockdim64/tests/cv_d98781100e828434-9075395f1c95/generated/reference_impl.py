"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    bt = d['chunk_size']
    errors = []
    k, w, u = (tile(d[n], bt) for n in ['k', 'w', 'u'])
    batch, count, heads, _, width = k.shape
    value_width = u.shape[-1]
    state = torch.zeros((batch, heads, width, value_width), dtype=torch.float64) if d['initial_state'] is None else d['initial_state'].double().clone()
    gate = scalar_tile(d['g'], bt) if d['g'] is not None else None
    saved = []
    values = []
    frequency = d.get('num_householder', 1)
    for chunk in range(count):
        if chunk % frequency == 0:
            saved.append(state.clone())
        new = u[:, chunk] - w[:, chunk] @ state
        values.append(new)
        weighted = new
        previous = state
        if gate is not None:
            last = gate[:, chunk, :, -1]
            weighted = new * torch.exp2(last[..., None] - gate[:, chunk])[..., None]
            state = state * torch.exp2(last)[..., None, None]
        state = state + k[:, chunk].transpose(-1, -2) @ weighted
        if chunk == 0:
            for key_coord, value_coord in [(0, 0), (1, 1), (width - 1, value_width - 1)]:
                row = 0
                scalar_new = float(u[0, chunk, 0, row, value_coord]) - math.fsum((float(w[0, chunk, 0, row, c]) * float(previous[0, 0, c, value_coord]) for c in range(width)))
                errors.append(abs(float(new[0, 0, row, value_coord]) - scalar_new))
                scalar_state = float(previous[0, 0, key_coord, value_coord])
                if gate is not None:
                    scalar_state *= 2.0 ** float(gate[0, chunk, 0, -1])
                scalar_state += math.fsum((float(k[0, chunk, 0, t, key_coord]) * float(weighted[0, 0, t, value_coord]) for t in range(bt)))
                errors.append(abs(float(state[0, 0, key_coord, value_coord]) - scalar_state))
    out = {'h': torch.stack(saved, dim=1), 'v_new': untile(torch.stack(values, dim=1))}
    if d['output_final_state']:
        out['final_state'] = state
    method = 'CPU64 chunk recurrence: v_new=u-w@state, gated state decay plus k^T@weighted v_new; original Householder state-save interval'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})

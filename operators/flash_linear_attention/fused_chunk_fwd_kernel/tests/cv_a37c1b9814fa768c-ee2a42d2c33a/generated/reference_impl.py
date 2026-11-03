"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    q, k, v = (d[n].double().transpose(1, 2) for n in ['q', 'k', 'v'])
    g = d['g'].double().transpose(1, 2)
    b, h, t, width = q.shape
    state = d['initial_state'].double().clone()
    out = torch.empty_like(v)
    for start in range(0, t, 64):
        end = min(start + 64, t)
        query = q[:, :, start:end] * d['scale']
        key = k[:, :, start:end]
        value = v[:, :, start:end]
        gate = g[:, :, start:end]
        local = torch.tril(query @ key.transpose(-1, -2) * torch.exp(gate[..., None] - gate[..., None, :]))
        out[:, :, start:end] = local @ value + query @ state * torch.exp(gate)[..., None]
        if start == 0:
            for row, col in [(0, 0), (1, 1), (end - start - 1, v.shape[-1] - 1)]:
                cross = math.fsum((float(query[0, 0, row, ki]) * float(state[0, 0, ki, col]) for ki in range(width))) * math.exp(float(gate[0, 0, row]))
                causal = math.fsum((math.fsum((float(query[0, 0, row, ki]) * float(key[0, 0, j, ki]) for ki in range(width))) * math.exp(float(gate[0, 0, row] - gate[0, 0, j])) * float(value[0, 0, j, col]) for j in range(row + 1)))
                errors.append(abs(float(out[0, 0, row, col]) - (cross + causal)))
        last = gate[..., -1]
        state = state * torch.exp(last)[..., None, None] + key.transpose(-1, -2) @ (value * torch.exp(last[..., None] - gate)[..., None])
    outputs = {'o': out.transpose(1, 2), 'ht': state}
    method = 'Independent CPU64 gated causal matrix attention plus recurrent initial/final state; unchanged BT64 and natural exponential'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in outputs.values()))
    assert max(errors) < 1e-09, errors
    return (outputs, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})

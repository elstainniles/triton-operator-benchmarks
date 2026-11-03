"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(cid, d):
    errors = []
    k, v, a, b = (d[n].double().transpose(1, 2) for n in ['k', 'v', 'a', 'b'])
    state = d['initial_state'].double().clone()
    saved = []
    values = []
    bt = d['chunk_size']
    t = k.shape[-2]
    for i in range(t):
        if i % bt == 0:
            saved.append(state.clone())
        value = (a[:, :, i, None, :] @ state).squeeze(-2)
        values.append(value)
        state = state + k[:, :, i, :, None] * v[:, :, i, None, :] + b[:, :, i, :, None] * value[:, :, None, :]
    new = torch.stack(values, dim=2)
    assert t <= bt
    eye = torch.eye(t, dtype=torch.float64).expand(*a.shape[:2], t, t)
    system = eye - torch.tril(a @ b.transpose(-1, -2), diagonal=-1)
    rhs = a @ d['initial_state'].double() + torch.tril(a @ k.transpose(-1, -2), diagonal=-1) @ v
    solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    errors.append(float((solved - new).abs().max()))
    out = {'h': torch.stack(saved, dim=1), 'v_new': new.transpose(1, 2), 'final_state': state}
    method = 'Independent CPU64 token recurrence v_new_t=a_t@state; state+=k_t outer v_t+b_t outer v_new_t; cross-checked with dense triangular solve'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})

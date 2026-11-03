"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    v, w, u = (d[n].double().transpose(1, 2) for n in ('v', 'w', 'u'))
    gv = d['gv'].double().transpose(1, 2)
    state = d['initial_state'].double().clone()
    b, h, t, vwidth = v.shape
    bt = d['chunk_size']
    kwidth = u.shape[-1]
    h_before = []
    keys = []
    scalar_checks = []
    for start in range(0, t, bt):
        sl = slice(start, min(t, start + bt))
        vv, ww, uu = (v[:, :, sl], w[:, :, sl], u[:, :, sl])
        h_before.append(state.clone())
        projected = ww @ state.to(d['w'].dtype).double().transpose(-1, -2)
        key = uu - projected
        keys.append(key)
        gain = gv[:, :, min(t, start + bt) - 1].exp()
        rounded_key = key.to(d['v'].dtype).double()
        new_state = state * gain[..., None, :] + rounded_key.transpose(-1, -2) @ vv
        if start in (0, t - bt):
            for row, col in [(0, 0), (kwidth - 1, vwidth - 1)]:
                expected = float(state[0, 0, row, col]) * float(gain[0, 0, col]) + math.fsum((float(rounded_key[0, 0, j, row]) * float(vv[0, 0, j, col]) for j in range(vv.shape[-2])))
                scalar_checks.append(abs(float(new_state[0, 0, row, col]) - expected))
            projected_alt = torch.einsum('bhiv,bhkv->bhik', ww, state.to(d['w'].dtype).double())
            torch.testing.assert_close(projected, projected_alt, rtol=1e-12, atol=1e-12)
        state = new_state
    gold = {'h': torch.stack(h_before, 1), 'k_new': torch.cat(keys, 2).transpose(1, 2).contiguous(), 'final_state': state}
    assert all((z.dtype == torch.float64 and torch.isfinite(z).all() for z in gold.values()))
    assert max(scalar_checks) < 1e-09
    details = {'method': 'Independent CPU64 state recurrence: Knew=U-W@H^T; Hnext=H*exp(g_last)+Knew^T@V; retain explicit original intermediate casts, no final gold rounding', 'scalar_checks': len(scalar_checks), 'max_scalar_error': max(scalar_checks), 'first_and_last_chunk_einsum_crosscheck': True, 'original_absolute_tolerance': {'rtol': 1e-05, 'atol': 1e-05}}
    return gold

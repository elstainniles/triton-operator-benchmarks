"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(d):
    assert (d['B'], d['T'], d['H'], d['K'], d['V'], d['chunk_size']) == (1, 63, 1, 64, 64, 64)
    assert d['cu_seqlens'] is None and d['chunk_indices'] is None and (d['output_final_state'] is True)
    h0 = d['initial_state'][0, 0].double()
    state_operand = h0.to(d['w'].dtype).double()
    kg, v, w, u, bg = [d[k][0, :, 0].double() for k in ('kg', 'v', 'w', 'u', 'bg')]
    vn = w @ state_operand + u
    final = h0 * torch.exp2(d['gk'][0, -1, 0].double())[:, None] + kg.T @ v + bg.T @ vn
    alt_vn = torch.einsum('tk,kv->tv', w, state_operand) + u
    alt_final = h0 * torch.exp2(d['gk'][0, -1, 0].double())[:, None]
    for t in range(d['T']):
        alt_final = alt_final + kg[t, :, None] * v[t, None, :] + bg[t, :, None] * alt_vn[t, None, :]
    torch.testing.assert_close(vn, alt_vn, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(final, alt_final, rtol=1e-12, atol=1e-12)
    return ({'h': h0.to(d['kg'].dtype).reshape(1, 1, 1, 64, 64), 'v_new': vn.to(d['u'].dtype).reshape(1, 63, 1, 64), 'final_state': final.float().reshape(1, 1, 64, 64)}, float((final - alt_final).abs().max()))

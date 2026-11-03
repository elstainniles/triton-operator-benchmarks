"""Existing independent CPU equations; file-generation main omitted."""
import reference_impl_builder as builder
import torch

def reference(cid, d):
    bt = d['BT']
    q, k, v, g = (builder.tile(d[n], bt) for n in ('q', 'k', 'v', 'g'))
    batch, count, heads, _, width = q.shape
    assert q.shape[2] == k.shape[2] and d['h0'] is None and (not d['output_final_state'])
    state = torch.zeros((batch, heads, width, v.shape[-1]), dtype=torch.float64)
    states = []
    outputs = []
    attention = torch.tril(q @ k.transpose(-1, -2)) * d['scale']
    for i in range(count):
        states.append(state.clone())
        expg = g[:, i].exp()
        outputs.append((q[:, i] @ state * d['scale'] + attention[:, i] @ (v[:, i] / expg)) * expg)
        state = state * expg[..., -1, None, :] + k[:, i].transpose(-1, -2) @ (v[:, i] * expg[..., -1:, :] / expg)
    out = {'A': builder.untile(attention), 'h': torch.stack(states, dim=1), 'o': builder.untile(torch.stack(outputs, dim=1))}
    original = chunk_gsa_fwd_k_torch(*(d[n].double() for n in ('q', 'k', 'v', 'g')), scale=d['scale'], BT=bt)
    expected = dict(zip(('A', 'h', 'unused', 'o'), original))
    differences = {}
    for name in out:
        error = float((out[name] - expected[name]).abs().max())
        differences[name] = error
        torch.testing.assert_close(out[name], expected[name], rtol=1e-12, atol=1e-10)
    return out

from original_cpu import chunk_gsa_fwd_k_torch

"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(cid, d):
    assert all((d[x] is None for x in ('g', 'gk', 'gv', 'h0')))
    B, T, H, K = d['k'].shape
    V = d['v'].shape[-1]
    assert d['g_gamma'] is None and d['output_final_state'] is False
    S = d['chunk_size']
    a = d['k'].double()
    b = d['v'].double()
    states = []
    acc = torch.zeros(B, H, K, V, dtype=torch.float64)
    for start in range(0, T, S):
        states.append(acc.clone())
        aa = a[:, start:start + S].transpose(1, 2)
        bb = b[:, start:start + S].transpose(1, 2)
        acc += aa.transpose(-1, -2) @ bb
    final = torch.stack(states, 1)
    alternate = torch.zeros_like(final)
    for i in range(1, len(states)):
        alternate[:, i] = torch.einsum('bthk,bthv->bhkv', a[:, :i * S], b[:, :i * S])
    torch.testing.assert_close(final, alternate, rtol=1e-10, atol=1e-10)
    return ({'h': final.to(d['k'].dtype), 'ht': None}, float((final - alternate).abs().max()), 'prefix K^T V before each original chunk; original FP16 output cast')

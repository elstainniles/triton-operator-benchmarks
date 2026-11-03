"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(cid, d):
    assert all((d[x] is None for x in ('g', 'gk', 'gv', 'h0')))
    B, T, H, K = d['k'].shape
    V = d['v'].shape[-1]
    assert d['dht'] is None and d['HQ'] == H
    S = d['split_size']
    a = (d['q'].float() * d['scale']).to(d['q'].dtype).double()
    b = d['do'].double()
    N = T // S
    result = torch.zeros(B, N, H, K, V, dtype=torch.float64)
    acc = torch.zeros(B, H, K, V, dtype=torch.float64)
    for i in range(N - 1, -1, -1):
        result[:, i] = acc
        aa = a[:, i * S:(i + 1) * S].transpose(1, 2)
        bb = b[:, i * S:(i + 1) * S].transpose(1, 2)
        acc += aa.transpose(-1, -2) @ bb
    alt = torch.zeros_like(result)
    for i in range(N - 1):
        alt[:, i] = torch.einsum('bthk,bthv->bhkv', a[:, (i + 1) * S:], b[:, (i + 1) * S:])
    torch.testing.assert_close(result, alt, rtol=1e-10, atol=1e-10)
    return ({'dhr': result.float().reshape(B * N, H, K, V), 'dh0': None}, float((result - alt).abs().max()), 'suffix (FP16(FP32(q)*scale))^T do strictly after each split; original split+reduction chain and FP32 output')

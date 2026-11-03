"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(d):
    assert d['h0'] is None and d['output_final_state'] is True
    B, T, H, K = d['k'].shape
    V = d['v'].shape[-1]
    BT = d['BT']
    assert T % BT == 0
    N = T // BT
    g = d['g_per_token'].reshape(B, N, BT, H).cumsum(2)
    last = g[:, :, -1]
    v = d['v'].reshape(B, N, BT, H, V)
    weighted = (v.float() * torch.exp(last[:, :, None] - g)[..., None]).to(v.dtype).double().permute(0, 1, 3, 2, 4)
    k = d['k'].double().reshape(B, N, BT, H, K).permute(0, 1, 3, 2, 4)
    contributions = (k.transpose(-1, -2) @ weighted).float().double()
    decay = torch.exp(last).double()
    state = torch.zeros(B, H, K, V, dtype=torch.float64)
    states = []
    for i in range(N):
        states.append(state.clone())
        state = state * decay[:, i, :, None, None] + contributions[:, i]
    h = torch.stack(states, 1)
    alt = torch.zeros_like(h)
    for i in range(1, N):
        for j in range(i):
            gain = decay[:, j + 1:i].prod(1) if j + 1 < i else torch.ones(B, H, dtype=torch.float64)
            alt[:, i] += contributions[:, j] * gain[:, :, None, None]
    torch.testing.assert_close(h, alt, rtol=1e-10, atol=1e-10)
    assert torch.isfinite(state).all() and torch.isfinite(h).all()
    return ({'h': h.float(), 'ht': state.float()}, float((h - alt).abs().max()))

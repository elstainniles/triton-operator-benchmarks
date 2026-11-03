"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math
V = 128
T = 1024
BT = 64
H = 8
B = 2

def make_fixtures():
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    v = torch.randn((B, T, H, V), generator=gen).half()
    do = torch.randn((B, T, H, V), generator=gen).half()
    out = reference(v, do)
    count = verify(v, do, out)
    return {'input_v.pt': v, 'input_do.pt': do, 'output_dA.pt': out}
SEED = 20261030
SCALE = V ** (-0.5)

def reference(v, do):
    result = torch.zeros((B, T, H, BT), dtype=torch.float32)
    for t in range(0, T, BT):
        n = min(BT, T - t)
        x = do[:, t:t + n].permute(0, 2, 1, 3).double()
        y = v[:, t:t + n].permute(0, 2, 1, 3).double()
        values = torch.tril(x @ y.transpose(-1, -2) * SCALE)
        result[:, t:t + n, :, :n] = values.permute(0, 2, 1, 3).float()
    return result

def verify(v, do, out):
    checks = 0
    for b in range(B):
        for t in (0, 1, 63, 64, 127, 1023):
            begin = t // BT * BT
            for h in (0, 3, 7):
                for j in (0, 1, 31, 62, 63):
                    want = 0.0
                    if j <= t - begin:
                        want = math.fsum((float(do[b, t, h, k]) * float(v[b, begin + j, h, k]) for k in range(V))) * SCALE
                    got = float(out[b, t, h, j])
                    if not math.isclose(got, want, abs_tol=1e-07, rel_tol=1e-06):
                        raise AssertionError((b, t, h, j, got, want))
                    checks += 1
    for t in range(0, T, BT):
        mask = torch.triu(torch.ones((BT, BT), dtype=torch.bool), diagonal=1)
        assert not out[:, t:t + BT].masked_select(mask[None, :, None, :]).count_nonzero()
    return checks

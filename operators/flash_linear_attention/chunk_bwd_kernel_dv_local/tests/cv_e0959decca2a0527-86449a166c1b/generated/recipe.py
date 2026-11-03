"""Existing independent CPU equations; file-generation main omitted."""
import torch
import torch.nn.functional as F
import math
D = 128
T = 2048
BT = 64
H = 8
B = 4

def make_fixtures():
    rng = torch.Generator(device='cpu').manual_seed(SEED)
    q = torch.randn((B, T, H, D), generator=rng, dtype=torch.float16)
    k = torch.randn((B, T, H, D), generator=rng, dtype=torch.float16)
    do = torch.randn((B, T, H, D), generator=rng, dtype=torch.float16)
    g = F.logsigmoid(torch.rand((B, T, H), generator=rng))
    A = torch.randn((B, T, H, BT), generator=rng)
    dv = reference(A, do)
    checks = verify(A, do, dv)
    inputs = {'q': q, 'k': k, 'do': do, 'g': g, 'A': A, 'B': B, 'T': T, 'H': H, 'D': D, 'chunk_size': BT, 'scale': SCALE, 'dtype': torch.float16}
    name = 'test_chunk_bwd_kernel_dv_local_cuda.pt'
    return {name: {'input_data': inputs, 'gpu_output': {'dv': dv}}}
SEED = 42
SCALE = 0.1

def reference(A, do):
    output = torch.empty((B, T, H, D), dtype=torch.float16)
    for t in range(0, T, BT):
        block = A[:, t:t + BT].permute(0, 2, 1, 3).transpose(-2, -1).triu().half().double()
        d = do[:, t:t + BT].permute(0, 2, 1, 3).double()
        output[:, t:t + BT] = (block @ d).permute(0, 2, 1, 3).half()
    return output

def verify(A, do, out):
    checked = 0
    for b in (0, 3):
        for t in (0, 1, 63, 64, 127, 2047):
            begin = t // BT * BT
            i = t % BT
            for h in (0, 7):
                for d in (0, 63, 127):
                    val = math.fsum((float(A[b, begin + j, h, i].half()) * float(do[b, begin + j, h, d]) for j in range(i, BT)))
                    expected = float(torch.tensor(val, dtype=torch.float16))
                    got = float(out[b, t, h, d])
                    if got != expected:
                        raise AssertionError((b, t, h, d, expected, got))
                    checked += 1
    return checked

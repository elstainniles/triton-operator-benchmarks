"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math
K = 128
V = 128
T = 1024
BT = 64
H = 8
B = 4

def make_fixtures():
    g = torch.Generator(device='cpu').manual_seed(SEED)
    q = torch.randn((B, T, H, K), dtype=torch.float16, generator=g)
    k = torch.randn((B, T, H, K), dtype=torch.float16, generator=g)
    v = torch.randn((B, T, H, V), dtype=torch.float16, generator=g)
    do = torch.randn((B, T, H, V), dtype=torch.float16, generator=g)
    dh = reference(q, do)
    checks = verify(q, do, dh)
    inputs = dict(q=q, k=k, v=v, g=None, gk=None, gv=None, do=do, h0=None, dht=None, scale=SCALE, B=B, T=T, H=H, HQ=H, K=K, V=V, chunk_size=BT)
    name = 'test_chunk_bwd_kernel_dh_parallel_cuda.pt'
    return {name: {'input_data': inputs, 'gpu_output': {'dh': dh, 'dh0': None}}}
SEED = 20261030
SCALE = K ** (-0.5)
NT = T // BT

def reference(q, do):
    out = torch.zeros((B, NT, H, K, V), dtype=torch.float16)
    for t in range(1, NT):
        query = (q[:, t * BT:(t + 1) * BT].float() * SCALE).half().double().permute(0, 2, 3, 1)
        grad = do[:, t * BT:(t + 1) * BT].double().permute(0, 2, 1, 3)
        out[:, t - 1] = (query @ grad).half()
    return out

def verify(q, do, out):
    assert out[:, -1].count_nonzero() == 0
    checks = 0
    for b in (0, 3):
        for t in (0, 1, 14):
            for h in (0, 7):
                for k in (0, 127):
                    for v in (0, 127):
                        expected = math.fsum((float((q[b, (t + 1) * BT + j, h, k].float() * SCALE).half()) * float(do[b, (t + 1) * BT + j, h, v]) for j in range(BT)))
                        expected = float(torch.tensor(expected, dtype=torch.float16))
                        got = float(out[b, t, h, k, v])
                        if expected != got:
                            raise AssertionError((b, t, h, k, v, expected, got))
                        checks += 1
    return checks

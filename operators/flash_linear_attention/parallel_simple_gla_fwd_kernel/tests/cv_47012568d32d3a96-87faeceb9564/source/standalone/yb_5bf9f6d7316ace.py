import torch

from yb_a017fb308312f4 import parallel_simple_gla_fwd

device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cuda'

def naive_parallel_simple_gla(q, k, v, g, scale):
    B, T, H, D = q.shape
    o = torch.zeros(*v.shape, dtype=torch.float32, device=v.device)

    if g is not None:
        g = g.float().cumsum(dim=1)

    for i in range(B):
        for h in range(H):
            for t in range(T):
                for s in range(t + 1):
                    score = torch.dot(q[i, t, h].float(), k[i, s, h].float()) * scale
                    if g is not None:
                        score = score * torch.exp(g[i, t, h] - g[i, s, h])
                    o[i, t, h] += score * v[i, s, h].float()

    return o.to(v.dtype), None

def test_parallel_simple_gla_fwd():
    torch.manual_seed(42)

    B, T, H, D = 2, 64, 4, 64
    scale = 0.1
    dtype = torch.float16

    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    g = None

    ref, _ = naive_parallel_simple_gla(q, k, v, g, scale)
    tri, _, _ = parallel_simple_gla_fwd(q, k, v, g, scale, output_attentions=False)
    torch.testing.assert_close(ref, tri, rtol=1e-2, atol=1e-2)
    assert torch.allclose(ref, tri, 1e-2, 1e-2)
    print(f"Parallel simple GLA fwd test passed! (B={B}, T={T}, H={H}, D={D})")

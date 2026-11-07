"""RMSNorm CPU-double reference; eps=10 exactly matches the supplied code."""
from opbench.generated_case import GeneratedCase
from .baseline import CONFIG


def make_case(shape, dtype, seed, device='npu'):
    import torch
    if dtype != torch.bfloat16:
        raise ValueError('RMSNorm baseline uses BF16 activations and FP32 weights')
    torch.manual_seed(seed)
    rows, cols = shape
    x = torch.randn((rows, cols), dtype=dtype)
    weight = torch.randn(cols, dtype=torch.float32)
    dy = torch.randn((rows, cols), dtype=dtype)
    xd = x.double().requires_grad_(True)
    wd = weight.double().requires_grad_(True)
    rrms = torch.rsqrt(xd.square().mean(-1) + CONFIG['eps'])
    y = xd * rrms[:, None] * wd
    case = GeneratedCase(torch, shape, dtype, device)
    case.input('x', x)
    case.input('weight', weight)
    case.output('y', shape, dtype, y, 0.02, 0.02)
    case.output('rrms', (rows,), torch.float32, rrms, 1e-5, 1e-6)
    return case

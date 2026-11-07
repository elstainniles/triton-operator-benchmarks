"""CPU autograd independently differentiates the SiTU-GLU v2 formula."""
from opbench.generated_case import GeneratedCase


def make_case(shape, dtype, seed, device='npu'):
    import torch
    if dtype != torch.bfloat16:
        raise ValueError('SiTU-GLU v2 baseline uses BF16')
    torch.manual_seed(seed)
    a = torch.randn(tuple(shape), dtype=dtype)
    b = torch.randn(tuple(shape), dtype=dtype)
    dout = torch.randn(tuple(shape), dtype=dtype)
    ad, bd = a.double().requires_grad_(True), b.double().requires_grad_(True)
    y = (0.5 * torch.tanh(ad / 0.5) * torch.sigmoid(ad)) * (0.5 * torch.tanh(bd / 0.5))
    case = GeneratedCase(torch, shape, dtype, device)
    case.input('a', a)
    case.input('b', b)
    y.backward(dout.double())
    case.input('dout', dout)
    case.output('da', shape, dtype, ad.grad, 0.02, 0.002)
    case.output('db', shape, dtype, bd.grad, 0.02, 0.002)
    return case

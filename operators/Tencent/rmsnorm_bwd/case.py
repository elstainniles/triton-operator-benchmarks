"""CPU autograd checks dx and dw; the target writes dx and partial dw only."""
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
    y.backward(dy.double())
    case.input('dy', dy)
    case.input('rrms', rrms.detach().float())
    groups = CONFIG['GROUPS']
    block_s = (rows + groups - 1) // groups
    partial = torch.zeros((groups, cols), dtype=torch.float64)
    for group in range(groups):
        begin, end = group * block_s, min((group + 1) * block_s, rows)
        if begin < rows:
            partial[group] = (dy[begin:end].double() * xd.detach()[begin:end]
                              * rrms.detach()[begin:end, None]).sum(0)
    case.output('dx', shape, dtype, xd.grad, 0.02, 0.02)
    dw = case.output('partial_dw', (groups, cols), torch.float32, partial, 1e-4, 1e-3)
    case.check('dw_sum', lambda: dw.cpu().double().sum(0), wd.grad, 1e-4, 1e-3)
    return case

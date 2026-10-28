"""Full-output reference; shape denotes (M, N) or (M, H), not every input shape."""
from opbench.tensors import Case, random_cpu


def make_case(shape, dtype, seed):
    import torch
    torch.manual_seed(seed)
    case = Case(torch, dtype)
    a_cpu = random_cpu(torch, shape, dtype)
    b_cpu = random_cpu(torch, shape, dtype)
    grad_cpu = random_cpu(torch, shape, dtype)
    a64, b64, grad64 = a_cpu.double(), b_cpu.double(), grad_cpu.double()
    sig = torch.sigmoid(a64)
    case.inputs = dict(shape=shape, ab=torch.cat([a_cpu, b_cpu], dim=1).contiguous().to('npu'),
                       grad=grad_cpu.to('npu'))
    case.actual = [case.output(shape)]
    case.expected = [(a64 * sig * b64).to(dtype)]
    return case

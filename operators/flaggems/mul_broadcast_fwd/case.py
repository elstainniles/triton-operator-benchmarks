"""Full-output reference; shape denotes (M, N) or (M, H), not every input shape."""
from opbench.tensors import Case, random_cpu


def make_case(shape, dtype, seed):
    import torch
    torch.manual_seed(seed)
    case = Case(torch, dtype)
    a_cpu = random_cpu(torch, shape, dtype)
    b_cpu = random_cpu(torch, shape, dtype)
    grad_cpu = random_cpu(torch, shape, dtype)
    case.inputs = dict(shape=shape, a=a_cpu[:, :1].contiguous().to('npu'),
                       b=b_cpu[:1, :].contiguous().to('npu'))
    case.actual = [case.output(shape)]
    case.expected = [(a_cpu[:, :1].double() * b_cpu[:1, :].double()).to(dtype)]
    return case

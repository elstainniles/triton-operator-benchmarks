"""Full-output CPU FP64 reference with inputs rounded to the tested dtype."""
from opbench.tensors import Case, random_cpu


def make_case(shape, dtype, seed):
    import torch
    torch.manual_seed(seed)
    case = Case(torch, dtype)
    x_cpu, y_cpu = random_cpu(torch, shape, dtype), random_cpu(torch, shape, dtype)
    mask_cpu = torch.arange(shape[0]) % 3 == 0
    xd, yd = x_cpu.double(), y_cpu.double()
    reference = (xd * 0.75).to(dtype)
    case.inputs = dict(shape=shape, x=x_cpu.to('npu'), y=y_cpu.to('npu'), mask=mask_cpu.to('npu'))
    case.actual = [case.output(shape)]
    case.expected = [reference]
    return case

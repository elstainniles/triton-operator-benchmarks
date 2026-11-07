"""In-place CE gradient with an independent CPU-double softmax reference."""
from opbench.generated_case import GeneratedCase


def make_case(shape, dtype, seed, device='npu'):
    import torch
    if dtype != torch.float32:
        raise ValueError('CE baseline is FP32')
    torch.manual_seed(seed)
    rows, cols = shape
    x = torch.randn((rows, cols), dtype=torch.float32)
    labels = torch.randint(0, cols, (rows,), dtype=torch.int64)
    # Exercise ignore_index, the last class, first class and a tile boundary.
    for row, label in enumerate((-1, cols - 1, 0, min(2048, cols - 1))):
        if row < rows:
            labels[row] = label
    valid = labels != -1
    row_max = torch.empty(rows, dtype=torch.float32)
    sum_exp = torch.empty(rows, dtype=torch.float32)
    dloss = torch.randn(rows, dtype=torch.float32)
    dloss[~valid] = 0.0
    reference = torch.empty_like(x)
    for begin in range(0, rows, 16):
        end = min(begin + 16, rows)
        chunk = x[begin:end].double()
        maximum = chunk.max(-1).values
        shifted = (chunk - maximum[:, None]).exp()
        summed = shifted.sum(-1)
        row_max[begin:end], sum_exp[begin:end] = maximum.float(), summed.float()
        grad = shifted / summed[:, None]
        grad[torch.arange(end - begin), labels[begin:end].clamp(0, cols - 1)] -= 1.0
        grad *= dloss[begin:end].double()[:, None]
        reference[begin:end] = grad.float()
    case = GeneratedCase(torch, shape, dtype, device)
    original = case.input('logits', x)
    case.input('labels', labels)
    case.input('row_max', row_max)
    case.input('sum_exp', sum_exp)
    case.input('dloss', dloss)
    target = case.output('dlogits', shape, torch.float32, reference, 1e-5, 1e-5)
    case.restore(target, original)
    return case

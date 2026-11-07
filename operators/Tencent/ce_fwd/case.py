"""CPU-double partial reductions; neither row maxima nor final loss are timed."""
from opbench.generated_case import GeneratedCase
from .baseline import CONFIG


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
    tile = CONFIG['BLOCK_SIZE']
    blocks = (cols + tile - 1) // tile
    expected_sum = torch.zeros((blocks, rows), dtype=torch.float64)
    expected_loss = torch.zeros_like(expected_sum)
    loss = torch.empty(rows, dtype=torch.float64)
    lse = torch.empty_like(loss)
    # Keep FP64 temporaries bounded; no full-sized double copy of the logits.
    for begin in range(0, rows, 16):
        end = min(begin + 16, rows)
        chunk = x[begin:end].double()
        maximum = chunk.max(-1).values
        row_max[begin:end] = maximum.float()
        shifted = (chunk - maximum[:, None]).exp()
        summed = shifted.sum(-1)
        sum_exp[begin:end] = summed.float()
        for block in range(blocks):
            expected_sum[block, begin:end] = shifted[:, block * tile:(block + 1) * tile].sum(-1)
        lab = labels[begin:end]
        good = valid[begin:end]
        local_rows = torch.arange(end - begin)
        safe = lab.clamp(0, cols - 1)
        delta = maximum - chunk[local_rows, safe]
        expected_loss[safe[good] // tile, torch.arange(begin, end)[good]] = delta[good]
        lse[begin:end] = maximum + summed.log()
        loss[begin:end] = torch.where(good, summed.log() + delta, 0.0)
    case = GeneratedCase(torch, shape, dtype, device)
    case.input('logits', x)
    case.input('labels', labels)
    case.input('row_max', row_max)
    partial_loss = case.output('partial_loss', (blocks, rows), torch.float32, expected_loss, 1e-5, 1e-5)
    partial_sum = case.output('partial_sum_exp', (blocks, rows), torch.float32, expected_sum, 1e-4, 1e-2)
    # These CPU reductions are validation only, outside all target launch labels.
    case.check('sum_exp', lambda: partial_sum.cpu().double().sum(0), sum_exp.double(), 1e-4, 1e-2)
    case.check('loss', lambda: torch.where(valid, partial_sum.cpu().double().sum(0).log()
                                         + partial_loss.cpu().double().sum(0), 0.0), loss, 1e-5, 1e-5)
    case.check('lse', lambda: partial_sum.cpu().double().sum(0).log() + row_max.double(), lse, 1e-5, 1e-5)
    return case

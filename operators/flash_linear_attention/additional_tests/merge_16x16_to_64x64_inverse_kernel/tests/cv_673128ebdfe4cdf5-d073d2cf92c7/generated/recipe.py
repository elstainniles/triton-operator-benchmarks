"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_fixtures():
    torch.set_num_threads(1)
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    b, t, h, bt, d = (4, 2048, 8, 64, 64)
    k = F.normalize(torch.randn((b, h, t, d), generator=gen, dtype=torch.float32), dim=-1)
    chunks = k.reshape(b, h, t // bt, bt, d)
    lower = (chunks @ chunks.transpose(-1, -2)).tril(-1)
    a = lower.reshape(b, h, t, bt).transpose(1, 2).contiguous()
    matrix = lower.to(torch.float64) + torch.eye(bt, dtype=torch.float64)
    eye = torch.eye(bt, dtype=torch.float64).expand_as(matrix)
    inverse = torch.linalg.solve_triangular(matrix, eye, upper=False)
    residual = (matrix @ inverse - eye).abs().max().item()
    if residual > 1e-12:
        raise RuntimeError(f'independent solve residual too large: {residual}')
    output = inverse.to(torch.float32).reshape(b, h, t, bt).transpose(1, 2).contiguous()
    return {'input_A_input.pt': a, 'output.pt': output}
SEED = 20261030

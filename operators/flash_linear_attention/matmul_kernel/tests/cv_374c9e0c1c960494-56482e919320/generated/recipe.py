"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_fixtures():
    gen = torch.Generator(device='cpu').manual_seed(20261030)
    a = torch.randn(4, 1024, 512, generator=gen, dtype=torch.float16)
    b = torch.randn(512, 2048, generator=gen, dtype=torch.float16)
    c = torch.matmul(a.double(), b.double()).half()
    return {'input_a.pt': a, 'input_b.pt': b, 'output.pt': c}

"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

FILE = 'test_parallel_attn_bwd_kernel_preprocess_cuda.pt'
SEED = 20261030
SHAPE = (4, 67, 3, 33)

def make():
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    o = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
    do = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
    delta = (o.double() * do.double()).sum(dim=-1).float()
    return {'input_data': {'o': o, 'do': do}, 'gpu_output': {'delta': delta}}

def make_fixtures():
    return {FILE: make()}

"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

def make_fixtures():
    gen=torch.Generator(device='cpu').manual_seed(42)
    x=torch.randn((4,1024,8),dtype=torch.float32,generator=gen)
    y=x.cumsum(dim=1)
    return {'input.pt':x,'reference.pt':y}

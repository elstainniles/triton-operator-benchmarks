"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

def make_fixtures():
    gen=torch.Generator(device='cpu').manual_seed(42)
    b,t,h,d=(2, 256, 3, 129)
    x=torch.randn((b,t,h,d),dtype=torch.float32,generator=gen)
    y=x.reshape(b,t//128,128,h,d).cumsum(2).reshape_as(x)
    return {'input.pt':{'s':x},'reference.pt':{'gpu_output':{'output':y}}}

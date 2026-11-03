"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

def make_fixtures():
    shape = (8, 4096, 4096)
    b,t,d = shape
    rng = torch.Generator(device='cpu').manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32, generator=rng)
    dy = torch.randn(shape, dtype=torch.float32, generator=rng)
    y = torch.empty_like(x)
    y[:,0,:] = -x[:,0,:]
    torch.sub(x[:,:-1,:], x[:,1:,:], out=y[:,1:,:])
    dx = torch.empty_like(dy)
    dx[:,-1,:] = -dy[:,-1,:]
    torch.sub(dy[:,1:,:], dy[:,:-1,:], out=dx[:,:-1,:])
    return {'token_shift_fwd_kernel_short_in.pt': {'x': x},
            'token_shift_bwd_kernel_short_in.pt': {'x': x, 'dy': dy},
            'token_shift_fwd_kernel_short_out.pt': {'y': y, 'N': b, 'T_actual': t, 'use_short_kernel': True, 'cache_out': None},
            'token_shift_bwd_kernel_short_out.pt': {'dx': dx, 'grad_cache': None}}

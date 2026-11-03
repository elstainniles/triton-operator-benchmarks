"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

def layernorm_reference(inputs):
    """Analytic Jacobian uses the exact supplied mean/rstd tensor values."""
    x, dy, weight = [inputs[name].double() for name in ('x', 'dy', 'weight')]
    mean, rstd = (inputs['mean'].double(), inputs['rstd'].double())
    xhat = (x - mean[:, None]) * rstd[:, None]
    weighted_dy = dy * weight
    dx = (weighted_dy - weighted_dy.mean(-1, keepdim=True) - xhat * (xhat * weighted_dy).mean(-1, keepdim=True)) * rstd[:, None]
    dw_rows, db_rows = (dy * xhat, dy.clone())
    return {'dx': dx.float(), 'dw_rows': dw_rows, 'db_rows': db_rows, 'dw': dw_rows.sum(0).float(), 'db': db_rows.sum(0).float()}

def make_fixtures():
    rng=torch.Generator(device='cpu').manual_seed(20261030)
    def randn(*shape,dtype=torch.float32):
        return torch.randn(*shape,generator=rng,dtype=dtype,device='cpu')
    x0=randn(2048,1024)
    inputs={'x':x0,'dy':randn(2048,1024),'weight':randn(1024),'bias':randn(1024),
            'mean':x0.mean(-1),'rstd':(x0.var(-1,unbiased=False)+1e-5).rsqrt()}
    payload=dict(operation='layernorm',input_origin='synthetic_cpu',seed=20261030,gpu_reference_created=False,inputs=inputs)
    return {'inputs.pt':payload,'reference.pt':layernorm_reference(inputs)}

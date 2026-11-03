"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

def make_fixtures():
    g=torch.Generator(device='cpu').manual_seed(20261030)
    B,T,D,BD,BT = 2,19,70,128,None
    k=torch.randn(B,T,D,generator=g,dtype=torch.float32)*.2
    a=.2+torch.rand(B,T,D,generator=g,dtype=torch.float32)
    ka=.1+.6*torch.rand(D,generator=g,dtype=torch.float32)
    expected=k.double()*(1+(a.double()-1)*ka.double())
    data=dict(k=k,a=a,ka=ka,B=B,T=T,D=D,BD=BD)
    if BT is not None: data.update(BT=BT,NT=(T+BT-1)//BT)
    payload=dict(input_data=data,gpu_output={'out':expected.float()},reference_origin='CPU float64 independent; gpu_output is historical test key only')
    return {'test_k_update_fwd_kernel_short_cuda.pt':payload}

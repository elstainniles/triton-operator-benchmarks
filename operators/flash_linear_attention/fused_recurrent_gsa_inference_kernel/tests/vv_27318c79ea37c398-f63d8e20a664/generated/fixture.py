"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch
import math
B, T, H, HQ, K, V, M = (4, 1, 4, 4, 64, 64, 16)

SEED = 20261030
NAME = 'test_fused_recurrent_gsa_inference_kernel_cuda.pt'

def make_inputs():
    gen = torch.Generator(device='cpu').manual_seed(SEED)

    def rand(shape):
        return torch.randn(shape, generator=gen, dtype=torch.float32)
    return {'q': rand((B, T, HQ, K)), 'k': rand((B, T, H, K)), 'v': rand((B, T, H, V)), 's': rand((B, T, H, M)), 'g': rand((B, T, H, M)), 'scale': K ** (-0.5), 'output_final_state': True, 'B': B, 'T': T, 'H': H, 'HQ': HQ, 'K': K, 'V': V, 'M': M}

def reference(inputs, dtype=torch.float64):
    """T=1, equal query/key head counts, zero initial states, final states enabled."""
    q, k, v, s = (inputs[name][:, 0].to(dtype) for name in ('q', 'k', 'v', 's'))
    hk = torch.einsum('bhk,bhm->bhkm', k, s)
    hv = torch.einsum('bhm,bhv->bhmv', s, v)
    logits = torch.einsum('bhk,bhkm->bhm', q * inputs['scale'], hk)
    output = torch.einsum('bhm,bhmv->bhv', torch.softmax(logits, dim=-1), hv)[:, None]
    return {'o': output, 'hkt': hk, 'hvt': hv}

def make_fixtures():
    inputs = make_inputs()
    expected = {name: value.float() for name, value in reference(inputs).items()}
    return {NAME: {'input_data': inputs, 'gpu_output': expected}}

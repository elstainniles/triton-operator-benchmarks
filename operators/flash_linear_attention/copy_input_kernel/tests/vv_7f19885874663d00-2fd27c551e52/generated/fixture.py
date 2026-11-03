"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

SEED = 20261030
FIELDS = ('q', 'k', 'v', 'g', 'level_scales')

def make(spec='full'):
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    if spec not in ('full', 'tail'):
        raise ValueError('unknown input specification')
    B, T, H, K, V, L, BT = (4, 64 if spec == 'full' else 67, 1, 16, 16, 2, 16)
    shape = {'q': (B, T, K), 'k': (B, T, K), 'v': (B, T, H, V), 'g': (B, T, H), 'level_scales': (B, T, H, L)}
    previous_shape = {'q': (B, BT, K), 'k': (B, BT, K), 'v': (B, BT, H, V), 'g': (B, BT, H), 'level_scales': (B, BT, H, L)}
    data = {'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'L': L, 'BT': BT, 'offsets': torch.tensor([0, 16, 32, 48] if spec == 'tail' else [0] * B, dtype=torch.int32)}
    reference = {}
    for name in FIELDS:
        data[name] = torch.randn(shape[name], dtype=torch.float32, generator=generator)
        data[name + '_prev'] = torch.randn(previous_shape[name], dtype=torch.float32, generator=generator)
        data[name + '_new'] = torch.zeros(shape[name], dtype=torch.float32)
        expected = data[name].clone()
        expected[:, :BT] = data[name][:, :BT] + data[name + '_prev']
        reference[name + '_new'] = expected
    return {'input_data': data, 'gpu_output': reference}

def make_fixtures():
    return {"test_copy_input_kernel_cuda.pt": make('full')}

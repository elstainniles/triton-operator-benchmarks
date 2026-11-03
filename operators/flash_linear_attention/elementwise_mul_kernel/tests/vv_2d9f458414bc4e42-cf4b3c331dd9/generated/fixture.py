"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: elementwise_mul
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'elementwise_mul'
    fixtures = {}
    folder = PurePosixPath('') / op
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (2048, 128) if op == 'elementwise_mul' else (2, 1024, 4096) if op in ('relu_square_bwd', 'rwkv_relu_square') else (2, 1024, 128)
    dtype = torch.bfloat16 if op in ('relu_square_bwd', 'rwkv_relu_square') else torch.float32
    x = torch.randn(shape, generator=gen, dtype=dtype)
    g = torch.tensor(2.0, dtype=torch.float32)
    inputs, output = (dict(x=x, g=g, N=2048, D=128), dict(x=x * g))
    name = {'elementwise_mul': 'test_elementwise_mul_kernel_cuda.pt', 'swish_fwd': 'test_swish_fwd_kernel_cuda.pt', 'swish_bwd': 'test_swish_bwd_kernel_cuda.pt', 'swiglu_fwd': 'test_swiglu_fwd_kernel_cuda.pt', 'swiglu_fwdbwd': 'test_swiglu_fwdbwd_kernel_cuda.pt', 'relu_square_bwd': 'test_relu_square_bwd_kernel_cuda.pt', 'rwkv_relu_square': 'test_rwkv_channel_mixing_pow_and_relu_cuda.pt'}[op]
    path = folder / name
    fixtures[path.name] = dict(input_data=inputs, gpu_output=output)
    return fixtures

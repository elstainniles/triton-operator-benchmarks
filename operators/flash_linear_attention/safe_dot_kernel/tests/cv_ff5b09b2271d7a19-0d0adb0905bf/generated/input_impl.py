"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(42)
    BM, BN, BK = (64, 64, 64)
    dtype = torch.float32
    device = 'cpu'
    torch.manual_seed(42)
    a = torch.randn(BM, BK, device=device, dtype=dtype)
    b = torch.randn(BK, BN, device=device, dtype=dtype)
    c = torch.zeros(BM, BN, device=device, dtype=dtype)
    data = {'input_data': {'a': a, 'b': b, 'c': c, 'BM': BM, 'BN': BN, 'BK': BK}, 'gpu_output': {'c': torch.zeros(BM, BN, device=device, dtype=dtype)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'a': a, 'b': b, 'c': c, 'BM': BM, 'BN': BN, 'BK': BK}}
        except NameError:
            pass
    return locals()

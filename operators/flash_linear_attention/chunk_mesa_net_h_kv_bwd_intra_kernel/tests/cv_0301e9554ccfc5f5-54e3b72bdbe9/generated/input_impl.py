import torch
from input_original import generate_test_data

def make_inputs():
    data, grid = generate_test_data(dtype=torch.float16, device='cpu')
    for key in ('dq', 'dk', 'dv', 'dg'):
        data[key].fill_(float('nan'))
    return {'input_data': data, 'grid': grid}

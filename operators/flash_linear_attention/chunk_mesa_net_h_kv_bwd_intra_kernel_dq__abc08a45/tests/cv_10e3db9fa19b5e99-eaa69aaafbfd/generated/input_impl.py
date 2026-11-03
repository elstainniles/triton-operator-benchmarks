import torch
from input_original import generate_test_data

def make_inputs():
    data, grid = generate_test_data(dtype=torch.float16, device='cpu')
    for key in ('dq','dk','dv','dg'):
        data[key] = torch.zeros_like(data[key], dtype=torch.float32)
    data['dg_final'] = torch.zeros_like(data['dg'])
    return {'input_data':data, 'grid':grid}

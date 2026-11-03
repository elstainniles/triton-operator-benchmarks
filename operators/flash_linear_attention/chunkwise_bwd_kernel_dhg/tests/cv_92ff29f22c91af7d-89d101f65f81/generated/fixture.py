"""Fixed-seed original-spec inputs and independent CPU reference, in memory."""
import torch
from input_impl import make_inputs
from reference_impl import reference

def make_fixtures():
    values = make_inputs()
    if isinstance(values.get('inputs_dict'), dict):
        inputs = values['inputs_dict']
        payload = inputs
    elif isinstance(values.get('input_dict'), dict):
        inputs = values['input_dict']
        payload = {'input_data': inputs}
    elif isinstance(values.get('input_data'), dict):
        inputs = values['input_data']
        payload = {'input_data': inputs}
    elif isinstance(values.get('data'), dict) and 'input_data' in values['data']:
        inputs = values['data']['input_data']
        payload = {'input_data': inputs}
    elif isinstance(values.get('inputs'), dict):
        inputs = values['inputs']
        payload = inputs
    else:
        raise RuntimeError('Original input prefix did not produce a complete input schema')
    grid = None
    if 'input_data' in payload and grid is not None and not callable(grid):
        payload['grid'] = tuple(grid)
    generator = torch.Generator().manual_seed(20261030)
    extra = torch.randn(*inputs['l'].shape[:-1], 1, generator=generator, dtype=inputs['l'].dtype) * 0.1
    inputs['l'] = torch.cat((inputs['l'], extra), dim=-1)
    inputs['L'] = 8
    gold = reference('cv_92ff29f22c91af7d', inputs)
    if isinstance(gold, tuple):
        gold = gold[0]
    fixtures = {}
    for name in ['inputs_L8_valid_level.pt', 'test_chunkwise_bwd_kernel_dhg_cuda.pt']:
        fixtures[name] = payload
    for name in ['reference_fp64.pt']:
        fixtures[name] = gold
    return fixtures

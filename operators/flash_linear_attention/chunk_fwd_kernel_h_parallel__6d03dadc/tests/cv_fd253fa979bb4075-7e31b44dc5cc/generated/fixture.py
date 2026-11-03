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
    gold = reference('cv_fd253fa979bb4075', inputs)
    if isinstance(gold, tuple):
        gold = gold[0]
    fixtures = {}
    for name in ['inputs_synthetic.pt', 'test_chunk_fwd_kernel_h_parallel_fp64_v2.pt', 'test_chunk_fwd_kernel_h_parallel_v2.pt']:
        fixtures[name] = payload
    for name in ['reference_fp64.pt']:
        fixtures[name] = gold
    return fixtures

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
    grid = values.get('grid', [4, 16])
    if 'input_data' in payload and grid is not None and not callable(grid):
        payload['grid'] = tuple(grid)
    gold = reference({'id': 'cv_0dcf8e4cb1081f20'}, None, inputs)
    if isinstance(gold, tuple):
        gold = gold[0]
    fixtures = {}
    for name in ['test_chunk_mesa_net_h_kv_bwd_intra_separate_kernel_fp16_v2.pt']:
        fixtures[name] = payload
    for name in ['independent_reference_cpu_fp32.pt', 'reference_fp64.pt']:
        fixtures[name] = gold
    from legacy_reference import reference as legacy_reference
    fixtures['independent_reference_cpu_fp32.pt'] = legacy_reference(inputs)
    return fixtures

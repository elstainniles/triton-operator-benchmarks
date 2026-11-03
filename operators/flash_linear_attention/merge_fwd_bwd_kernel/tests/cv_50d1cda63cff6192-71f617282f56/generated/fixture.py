"""Original seeded inputs and independent CPU equations, generated in memory."""
from input_impl import make_inputs
from reference_impl import *

def make_fixtures():
    values=make_inputs()
    inputs=values.get('inputs_dict') or values.get('input_dict') or values.get('input_data') or values.get('inputs') or values.get('data',{}).get('input_data')
    if not isinstance(inputs,dict):
        raise ValueError('Original input schema unavailable')
    gold=reference_merge(inputs)[0]
    payload={'input_data':inputs,'gpu_output':gold}
    fixtures={}
    for key in ['inputs_and_reference.pt']: fixtures[key]=payload
    aliases={'test_merge_fwd_bwd_kernel_cuda.pt': 'inputs_and_reference.pt'}
    for old,new in aliases.items():
        if new in fixtures: fixtures[old]=fixtures[new]
    return fixtures

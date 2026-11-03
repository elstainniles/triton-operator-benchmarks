"""Original seeded inputs and independent CPU equations, generated in memory."""
from input_impl import make_inputs
from reference_impl import *

def make_fixtures():
    values=make_inputs()
    inputs=values.get('inputs_dict') or values.get('input_dict') or values.get('input_data') or values.get('inputs') or values.get('data',{}).get('input_data')
    if not isinstance(inputs,dict):
        raise ValueError('Original input schema unavailable')
    gold=based_math(inputs)[0]
    gold={k:v.squeeze(0) for k,v in gold.items()}
    payload={'input_data':inputs}
    grid=values.get('grid') or None
    if grid is not None and not callable(grid): payload['grid']=tuple(grid)
    fixtures={}
    for key in ['inputs_synthetic.pt']: fixtures[key]=payload
    for key in ['reference_fp64.pt']: fixtures[key]=gold
    aliases={'test_fused_chunk_based_bwd_kernel_cuda.pt': 'inputs_synthetic.pt', 'test_fused_chunk_based_bwd_kernel_cuda_fp64.pt': 'inputs_synthetic.pt'}
    for old,new in aliases.items():
        if new in fixtures: fixtures[old]=fixtures[new]
    return fixtures

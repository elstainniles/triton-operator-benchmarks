"""Original seeded inputs and independent CPU equations, generated in memory."""
from input_impl import make_inputs
from reference_impl import *

def make_fixtures():
    values=make_inputs()
    inputs=values.get('inputs_dict') or values.get('input_dict') or values.get('input_data') or values.get('inputs') or values.get('data',{}).get('input_data')
    if not isinstance(inputs,dict):
        raise ValueError('Original input schema unavailable')
    gold=mesa(inputs,None)[0]
    payload={'input_data':inputs}
    grid=values.get('grid') or None
    if grid is not None and not callable(grid): payload['grid']=tuple(grid)
    fixtures={}
    for key in ['inputs_synthetic.pt']: fixtures[key]=payload
    for key in ['reference_fp64.pt']: fixtures[key]=gold
    aliases={'test_chunk_mesa_net_h_kk_bwd_intra_kernel_fp16_v2.pt': 'inputs_synthetic.pt'}
    for old,new in aliases.items():
        if new in fixtures: fixtures[old]=fixtures[new]
    return fixtures

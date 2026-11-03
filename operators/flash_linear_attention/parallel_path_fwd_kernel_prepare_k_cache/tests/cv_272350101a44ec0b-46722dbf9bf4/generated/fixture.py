"""Original seeded inputs and independent CPU equations, generated in memory."""
from input_impl import make_inputs
from reference_impl import *

def make_fixtures():
    values=make_inputs()
    inputs=values.get('inputs_dict') or values.get('input_dict') or values.get('input_data') or values.get('inputs') or values.get('data',{}).get('input_data')
    if not isinstance(inputs,dict):
        raise ValueError('Original input schema unavailable')
    gold=cache_reference(inputs,None)[0]
    payload=inputs
    fixtures={}
    for key in ['inputs.pt']: fixtures[key]=payload
    for key in ['reference_fp64.pt']: fixtures[key]=gold
    aliases={'inputs.pt': 'inputs.pt', 'outputs.pt': 'inputs.pt'}
    for old,new in aliases.items():
        if new in fixtures: fixtures[old]=fixtures[new]
    return fixtures

"""Original seeded inputs and independent CPU equations, generated in memory."""
from input_impl import make_inputs
from reference_impl import *

def make_fixtures():
    values=make_inputs()
    inputs=values.get('inputs_dict') or values.get('input_dict') or values.get('input_data') or values.get('inputs') or values.get('data',{}).get('input_data')
    if not isinstance(inputs,dict):
        raise ValueError('Original input schema unavailable')
    gold=forward_inter(inputs)
    validate_scalar_points(inputs,gold,'cv_60edb64ab2e833aa')
    payload=inputs
    fixtures={}
    for key in ['inputs_synthetic.pt']: fixtures[key]=payload
    for key in ['independent_outputs.pt']: fixtures[key]=gold
    aliases={}
    for old,new in aliases.items():
        if new in fixtures: fixtures[old]=fixtures[new]
    return fixtures

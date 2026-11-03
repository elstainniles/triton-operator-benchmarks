"""Original seeded inputs and independent CPU equations, generated in memory."""
from input_impl import make_inputs
from reference_impl import *

def make_fixtures():
    values=make_inputs()
    inputs=values.get('inputs_dict') or values.get('input_dict') or values.get('input_data') or values.get('inputs') or values.get('data',{}).get('input_data')
    if not isinstance(inputs,dict):
        raise ValueError('Original input schema unavailable')
    gold=ttt(inputs,None)[0]
    payload={'input_data':inputs}
    grid=values.get('grid') or None
    if grid is not None and not callable(grid): payload['grid']=tuple(grid)
    fixtures={}
    for key in ['q.pt', 'k.pt', 'v.pt', 'v_new.pt', 'x.pt', 'y.pt', 'dv_new.pt', 'do_tensor.pt', 'rstd.pt', 'w.pt', 'b.pt', 'eta.pt', 'h0.pt', 'hb0.pt', 'h.pt', 'dht.pt', 'dhbt.pt']: fixtures[key]=inputs[key.removesuffix('.pt')]
    for key in ['reference_fp64.pt']: fixtures[key]=gold
    aliases={'q.pt': 'q.pt', 'k.pt': 'k.pt', 'v.pt': 'v.pt', 'v_new.pt': 'v_new.pt', 'x.pt': 'x.pt', 'y.pt': 'y.pt', 'rstd.pt': 'rstd.pt', 'w.pt': 'w.pt', 'b.pt': 'b.pt', 'eta.pt': 'eta.pt', 'h0.pt': 'h0.pt', 'hb0.pt': 'hb0.pt', 'h.pt': 'h.pt', 'dht.pt': 'dht.pt', 'dhbt.pt': 'dhbt.pt', 'dv_new.pt': 'dv_new.pt', 'do_tensor.pt': 'do_tensor.pt', 'output_0.pt': 'output_0.pt', 'output_1.pt': 'output_1.pt', 'output_2.pt': 'output_2.pt', 'output_3.pt': 'output_3.pt', 'output_4.pt': 'output_4.pt', 'output_5.pt': 'output_5.pt', 'output_6.pt': 'output_6.pt', 'output_7.pt': 'output_7.pt'}
    for old,new in aliases.items():
        if new in fixtures: fixtures[old]=fixtures[new]
    return fixtures

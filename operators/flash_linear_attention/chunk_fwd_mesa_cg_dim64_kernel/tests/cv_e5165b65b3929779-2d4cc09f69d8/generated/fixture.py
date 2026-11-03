import torch
from input_impl import make_inputs

def _build():
    values=make_inputs()
    if 'data' in values and 'input_data' in values['data']:
        data=values['data']
    elif 'input_data' in values:
        data={'input_data':values['input_data']}
    else:
        data={'input_data':values.get('input_dict',values.get('inputs_dict'))}
    data['grid']=(8,8)
    return {'inputs_synthetic.pt':data}

def make_fixtures():
    previous=torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        return _build()
    finally:
        torch.set_num_threads(previous)

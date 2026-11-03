import torch
from input_impl import make_inputs
from reference_impl import reference

def _build():
    values=make_inputs()
    if 'data' in values and 'input_data' in values['data']:
        data=values['data']
    elif 'input_data' in values:
        data={'input_data':values['input_data']}
    else:
        data={'input_data':values.get('input_dict',values.get('inputs_dict'))}
    gold=reference(data["input_data"])[0]
    data["gpu_output"]=gold
    data["reference_origin"]='CPU64 equations with original operand/output casts; gpu_output is legacy schema only'
    return {'test_chunk_dplr_fwd_kernel_h_cpu_reference.pt':data}

def make_fixtures():
    previous=torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        return _build()
    finally:
        torch.set_num_threads(previous)

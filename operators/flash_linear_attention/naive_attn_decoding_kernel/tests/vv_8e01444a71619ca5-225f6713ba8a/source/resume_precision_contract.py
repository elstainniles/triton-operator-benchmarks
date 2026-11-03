import torch
SPECS = {
    'vv_2ec9259896ac4103': ('fused_recurrent_bwd_kernel', (0,1,2,3,4,5,8), ((9,'dq'),(10,'dk'),(11,'dv'),(12,'da'),(13,'db'),(14,'dha')), (6,7,15,17)),
    'vv_bea43e08f1b68a1e': ('fused_recurrent_fwd_kernel', (0,1,2,3,4), ((5,'o'),(6,'ha')), (7,8,9)),
    'vv_8e01444a71619ca5': ('naive_attn_decoding_kernel', (0,1,2,4,7), ((3,'o'),), (5,)),
}
def register(audited, arguments, multi, nested, nulls):
    for cid,(kernel,inputs,outputs,aux) in SPECS.items():
        audited[cid]=kernel
        arguments[cid]={'inputs':inputs,'output':outputs[0][0]}
        multi[cid]=outputs
        nested.add(cid)
        nulls[cid]=aux

def extract(torch, case, path, reference):
    cid=case['id']
    if cid not in SPECS:
        return None
    if path.name != 'reference_cpu64.pt':
        raise ValueError('expected original independent CPU reference')
    result={}
    for index,key in SPECS[cid][2]:
        value=reference[key]
        dtype=(torch.bfloat16 if cid=='vv_8e01444a71619ca5' else
               torch.float16 if cid=='vv_2ec9259896ac4103' and key!='dha' else torch.float32)
        value=value.to(dtype)
        if cid=='vv_2ec9259896ac4103' and key in ('dq','dk','da','db'):
            value=value.unsqueeze(0)  # Original V=64, BV=64 gives exactly one partial.
        result[str(index)]=value
    return result

def compare_raw(case_id, reference, outputs, compare_data_precision):
    # Preserve the original utility's per-output acceptance, including BF16 5e-3.
    values=extract(torch, {'id':case_id}, __import__('pathlib').Path('reference_cpu64.pt'), reference)
    expected={key:values[str(index)] for index,key in SPECS[case_id][2]}
    compare_data_precision(expected, outputs, device_type='cpu')

"""GRPO full-storage and individual semantic-output validation.

The original loss comparator is retained. KL, packed-buffer and masked LSE
checks are explicitly additional checks, never labelled original acceptance.
"""
from contextlib import contextmanager
CASES={'vv_4e7b5a0967a2b961','vv_6c97bc8a9d6edaae'}
KERNEL='grpo_fwd_kernel'
CURRENT_REGION=None
ACCEPTANCE={5:'original loss acceptance plus added packed-storage and KL validation',6:'added independent LSE validation including masked-row preservation'}
@contextmanager
def region(name):
    global CURRENT_REGION
    if CURRENT_REGION is not None:raise RuntimeError('nested GRPO validation region')
    CURRENT_REGION=name
    try:yield
    finally:CURRENT_REGION=None

def register(audited,arguments,outputs):
    for cid in CASES:
        audited[cid]=KERNEL;arguments[cid]={'inputs':(0,1,2,3,4),'output':5};outputs[cid]=((5,'packed_loss'),(6,'lse'))

def active_mask(torch,case,device='cpu'):
    values=case.get('grpo_active_mask')
    if not isinstance(values,list) or len(values)!=2048 or any(type(v) is not int or v not in (0,1) for v in values):
        raise ValueError('GRPO immutable completion-mask contract missing')
    return torch.tensor(values,dtype=torch.bool,device=device).reshape(2,1024)

def validate(case,args,torch):
    if case.get('id') not in CASES:return
    if len(args)!=24:raise ValueError('GRPO compiled argument count differs')
    x=tuple(args[9:]);save=case.get('grpo_save_kl')
    shapes=[(2,1025,32000),(2,1024),(2,1088),(2,),(2,1024),(4 if save else 2,1024),(2,1024)]
    dtypes=[torch.bfloat16,torch.float32,torch.int64,torch.float32,torch.int32,torch.float32,torch.float32]
    if type(save) is not bool:raise ValueError('GRPO save_kl specification missing')
    for value,shape,dtype in zip(x[:7],shapes,dtypes):
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or value.dtype!=dtype or not value.is_contiguous() or value.storage_offset()!=0:
            raise ValueError('GRPO original tensor specification differs')
    if tuple(args[:3])!=(2048,1,1) or x[7:14]!=(0.04,save,2,2048,32000,1024,64) or x[14] not in (1024,2048,4096,8192):
        raise ValueError('GRPO original grid, beta, save_kl or tile differs')
    if not torch.equal(x[4].cpu().bool(),active_mask(torch,case)):
        raise ValueError('GRPO original completion mask differs')

def initialize_unused(torch,args,case):
    if case.get('id') in CASES:
        lse=args[15];lse.masked_fill_(~active_mask(torch,case,lse.device),0.0)

def require_outputs(torch,output,case):
    if case.get('id') not in CASES:return
    mask=active_mask(torch,case,output.device)
    if tuple(output.shape)==(4,1024):mask=mask.repeat(2,1)
    if tuple(output.shape)!=tuple(mask.shape) or not bool(torch.isfinite(output).all()) or not bool((output[~mask]==0).all()):
        raise AssertionError('GRPO active outputs unwritten/nonfinite or masked rows changed')

def extract(torch,case,path,value):
    if case.get('id') not in CASES:return None
    if path.name!='reference.pt' or not isinstance(value,dict):raise ValueError('GRPO independent reference file differs')
    key='packed_true' if case['grpo_save_kl'] else 'packed_false'
    result={'5':value.get(key),'6':value.get('lse')}
    shapes={'5':(4 if case['grpo_save_kl'] else 2,1024),'6':(2,1024)}
    for index,tensor in result.items():
        if not isinstance(tensor,torch.Tensor) or tuple(tensor.shape)!=shapes[index] or tensor.dtype!=torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError('GRPO full-output reference specification differs')
    return result

def acceptance(case,index):
    return ACCEPTANCE[index] if case.get('id') in CASES else 'original torch.testing.assert_close'

def expected_regions(case):
    return {'original_loss','packed_storage','added_lse'} | ({'added_kl'} if case['grpo_save_kl'] else set())

def record_assertion(torch,case,result,observed,actual,expected,args,kwargs):
    if case.get('id') not in CASES:return
    name=CURRENT_REGION
    if name not in expected_regions(case):raise ValueError('GRPO assertion outside explicit validation region')
    tolerance=.005 if name in ('original_loss','packed_storage') else 1e-5
    equal_nan=name=='original_loss'
    if args or kwargs!={'rtol':tolerance,'atol':tolerance,'equal_nan':equal_nan}:
        raise ValueError('GRPO original/additional acceptance threshold changed')
    index=6 if name=='added_lse' else 5;target=observed['outputs'].get(index)
    if target is None:raise ValueError('GRPO assertion precedes target launch')
    offset=2048 if name=='added_kl' else 0
    shape=tuple(target.shape) if name in ('packed_storage','added_lse') else (2,1024)
    def alias(value):
        return (isinstance(value,torch.Tensor) and tuple(value.shape)==shape and value.dtype==target.dtype
            and value.untyped_storage()._cdata==target.untyped_storage()._cdata
            and value.storage_offset()==target.storage_offset()+offset and value.is_contiguous())
    matches=[i for i,v in enumerate((actual,expected)) if alias(v)]
    if len(matches)!=1:raise ValueError('GRPO semantic assertion does not bind exact output storage region')
    cal=(actual,expected)[matches[0]];gold=(actual,expected)[1-matches[0]]
    refs=[]
    for path,(digest,values) in observed['expected_cpu'].items():
        value=values[str(index)]
        if name=='original_loss':value=value[:2]
        elif name=='added_kl':value=value[2:]
        if isinstance(gold,torch.Tensor) and gold.dtype==value.dtype and torch.equal(gold.cpu(),value):refs.append((path,digest))
    if len(refs)!=1:raise ValueError('GRPO semantic assertion reference not uniquely bound')
    checks=result.setdefault('grpo_semantic_checks',{})
    if name in checks:raise ValueError('GRPO semantic assertion repeated')
    checks[name]=dict(status='passed',source='original acceptance' if name=='original_loss' else 'additional independent validation',
        output_argument=index,offset_elements=offset,elements=cal.numel(),shape=list(shape),rtol=tolerance,atol=tolerance,equal_nan=equal_nan,
        reference_file_path=refs[0][0],reference_file_sha256=refs[0][1],binding='exact contiguous target storage region')

def finish(case,result):
    if case.get('id') not in CASES:return
    checks=result.get('grpo_semantic_checks',{})
    if set(checks)!=expected_regions(case):raise AssertionError('GRPO loss/KL/LSE or packed-buffer validation missing')
    result['grpo_reference_validation']=dict(save_kl=case['grpo_save_kl'],original_loss_acceptance_preserved=True,
        additional_checks=sorted(expected_regions(case)-{'original_loss'}),masked_lse_initialization='zero only where original kernel does not write',
        all_active_outputs_checked=True,masked_rows_zero_preserved=True)

def proof(case,record,expected_refs):
    if case.get('id') not in CASES:return True
    checks=record.get('grpo_semantic_checks',{});summary=record.get('grpo_reference_validation',{})
    if set(checks)!=expected_regions(case) or summary.get('save_kl') is not case['grpo_save_kl'] or summary.get('original_loss_acceptance_preserved') is not True or summary.get('all_active_outputs_checked') is not True or summary.get('masked_rows_zero_preserved') is not True:return False
    if (summary.get('additional_checks')!=sorted(expected_regions(case)-{'original_loss'})
            or summary.get('masked_lse_initialization')!='zero only where original kernel does not write'):
        return False
    for name,item in checks.items():
        tolerance=.005 if name in ('original_loss','packed_storage') else 1e-5
        index=6 if name=='added_lse' else 5
        offset=2048 if name=='added_kl' else 0
        shape=[4,1024] if name=='packed_storage' and case['grpo_save_kl'] else [2,1024]
        source='original acceptance' if name=='original_loss' else 'additional independent validation'
        if (item.get('status')!='passed' or item.get('reference_file_sha256') not in expected_refs
                or not item.get('reference_file_path') or item.get('rtol')!=tolerance or item.get('atol')!=tolerance
                or item.get('equal_nan')!=(name=='original_loss') or item.get('binding')!='exact contiguous target storage region'
                or item.get('source')!=source or item.get('output_argument')!=index or item.get('offset_elements')!=offset
                or item.get('shape')!=shape or item.get('elements')!=shape[0]*shape[1]):
            return False
    return True

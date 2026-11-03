"""User-approved CPU FP64 comparisons for four former CUDA-ratio cases."""
import copy
import hashlib
import json
from pathlib import Path

STANDARD='cpu_fp64_absolute_v1'
ACCEPTANCE='user-approved CPU FP64 absolute comparison'
TOLERANCES={'torch.float32':1e-5,'torch.float16':1e-3,'torch.bfloat16':1e-2}
FWD='vv_07df4bbe51f098fb';BWD='vv_9ecf456248792a04'
LOG2={'vv_ab52a2ea7627760b','vv_2bd7e3ed0e209f67'}
CONTRACTS={FWD:('gate_output_correction_fwd_kernel',tuple(range(6)),((6,'output'),)),
 BWD:('gate_output_correction_bwd_kernel',tuple(range(7)),((7,'grad_o'),(8,'grad_r'),(9,'grad_k'),(10,'grad_r_k_intermediate'),(11,'grad_v'),(12,'grad_g')))}
for cid in LOG2:CONTRACTS[cid]=('log2_kernel',(0,),((1,'output_ptr'),))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metadata(case,result):
 if case.get('validation_standard')==STANDARD:
  for key in ('validation_standard','validation_tolerances','accuracy_criteria_changed','original_validation_standard'):
   result[key]=copy.deepcopy(case[key])

def register(audited,arguments,outputs):
 for cid,(kernel,inputs,written) in CONTRACTS.items():
  audited[cid]=kernel;arguments[cid]={'inputs':inputs,'output':written[0][0]};outputs[cid]=written

def extract(case,reference):
 if case.get('validation_standard')!=STANDARD:return None
 return {str(i):reference[key] for i,key in CONTRACTS[case['id']][2]}

def compare_named(torch,values,reference):
 """Public baseline entry: compare actual original-wrapper outputs to FP64."""
 for name,actual in values.items():
  expected=reference[name]
  if expected.dtype!=torch.float64:raise ValueError('independent golden must remain FP64: '+name)
  tol=TOLERANCES[str(actual.dtype)]
  torch.testing.assert_close(actual.cpu(),expected,rtol=tol,atol=tol,check_dtype=False,equal_nan=False)

def check_actual_outputs(torch,case,outputs,refs,result,observed,original_close):
 if case.get('validation_standard')!=STANDARD:return
 from direct_reference_worker import record_reference_check
 metadata(case,result)
 for index,actual in outputs.items():
  matches=[(path,digest,values[str(index)]) for path,(digest,values) in refs.items() if str(index) in values]
  if len(matches)!=1:raise ValueError('CPU64 independent output is missing or ambiguous')
  path,digest,expected=matches[0]
  if expected.dtype!=torch.float64 or expected.shape!=actual.shape:raise ValueError('CPU64 raw target-output geometry differs')
  tol=TOLERANCES[str(actual.dtype)];cpu=actual.detach().cpu();delta=(cpu.double()-expected).abs()
  record={'acceptance':ACCEPTANCE,'validation_standard':STANDARD,'dtype':str(actual.dtype),
          'golden_dtype':str(expected.dtype),'rtol':tol,'atol':tol,'check_dtype':False,'equal_nan':False,
          'reference_file_path':path,'reference_file_sha256':digest,'output_assertion_argument':0,
          'output_binding':'actual compiled kernel output checked directly before original wrapper',
          'max_abs_error':float(delta.max()),'elements':actual.numel()}
  result.setdefault('validation_tolerances',{})[str(index)]={'dtype':str(actual.dtype),'rtol':tol,'atol':tol}
  result.setdefault('cpu64_output_checks',{})[str(index)]=dict(record,status='running')
  try:original_close(cpu,expected,rtol=tol,atol=tol,check_dtype=False,equal_nan=False)
  except AssertionError as exc:
   result['cpu64_output_checks'][str(index)].update(status='failed',error=str(exc));raise
  result['cpu64_output_checks'][str(index)]['status']='passed'
  record_reference_check(result,index,record);observed['assert_passed'].add(index)

def stage(cases,bundle,repo):
 bundle=Path(bundle);repo=Path(repo)
 packet=json.loads((bundle/'cpu64_absolute_cases.json').read_text());ready={x['id']:x for x in packet['cases']}
 for proof in packet['source_proofs']:
  for m in proof['module_proofs']:
   if sha(bundle/m['path'])!=m['sha256'] or sha(repo/m['source'])!=m['source_sha256']:
    raise ValueError('CPU64 adapter original source changed: '+m['path'])
 result=[]
 for previous in cases:
  if previous['id'] not in ready:result.append(previous);continue
  c=copy.deepcopy(ready[previous['id']]);assert c['original_nodeid']==previous['original_nodeid']
  for item in c['required_data']:
   if sha(item['path'])!=item['sha256']:raise ValueError('CPU64 case data changed')
  c['fla_source_binding']['source_repo']=str(repo)
  ref=copy.deepcopy(c);ref['function']='test_cpu64';ref['correctness_kind']='assertions'
  c['associated_reference_case']=ref;result.append(c)
 return result

"""Runtime records of independently generated CPU reference checks."""
import torch
CHECKS=[]
TOLS={'torch.float32':1e-5,'torch.float16':1e-3,'torch.bfloat16':1e-2}
def compare_cpu64(values,reference):
 for name,actual in values.items():
  expected=reference[name]
  if actual is None or expected is None:
   if actual is not None or expected is not None:raise AssertionError('None output mismatch: '+name)
   continue
  if expected.dtype!=torch.float64:raise ValueError('CPU golden is not FP64: '+name)
  tol=TOLS[str(actual.dtype)];cpu=actual.detach().cpu()
  row=dict(output=name,dtype=str(actual.dtype),reference_dtype=str(expected.dtype),rtol=tol,atol=tol,status='running');CHECKS.append(row)
  try:torch.testing.assert_close(cpu,expected,rtol=tol,atol=tol,check_dtype=False,equal_nan=False)
  except AssertionError as exc:row.update(status='failed',error=str(exc));raise
  row.update(status='passed')
def compare_original(values,reference,comparator):
 # The original reference PT used the original output dtype; retain that dtype and original validator.
 expected={name:(None if actual is None else reference[name].to(actual.dtype)) for name,actual in values.items()}
 comparator(expected,values,device_type='cpu')
def publish(result):
 if result.get('validation_standard')=='cpu_fp64_absolute_v1' and CHECKS:
  result['validation_tolerances']={r['output']:{k:r[k] for k in ('dtype','rtol','atol')} for r in CHECKS}
  result['cpu64_output_checks']=list(CHECKS)

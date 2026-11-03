#!/usr/bin/env python3
"""Extract existing family CPU recipes as in-memory fixtures; no device execution."""
import argparse,ast,copy,hashlib,json,sys
from pathlib import Path
from port_fla_cv_fixtures import PureExporter,input_prefix,check_pure,specialize_function

def parse(p):return ast.parse(Path(p).read_text())
def named(n):return ast.unparse(n)
def source_function(p,name='main'):
 return next(n for n in parse(p).body if isinstance(n,ast.FunctionDef) and n.name==name)
def cut(body,start,stop):
 active=False; result=[]
 for n in body:
  s=named(n)
  if not active and start in s:active=True
  if active:
   if stop in s:break
   result.append(copy.deepcopy(n))
 return result

def function(body,name='make_fixtures'):
 f=ast.parse(f'def {name}():\n pass').body[0];f.body=body;return ast.fix_missing_locations(f)
def ret(expr):return ast.parse('return '+expr).body[0]
def export(p,body,out,extra=None):
 overrides={'make_fixtures':function(body)}
 for n in parse(p).body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],(ast.Tuple,ast.List)) and isinstance(n.value,(ast.Tuple,ast.List)):
   for t,v in zip(n.targets[0].elts,n.value.elts):
    if isinstance(t,ast.Name):overrides[t.id]=ast.Assign(targets=[copy.deepcopy(t)],value=copy.deepcopy(v))
 overrides.update(extra or {})
 exporter=PureExporter(out,[Path(p).parent]);exporter.export(p,{'make_fixtures'},'recipe',overrides)
 (out/'fixture.py').write_text('"""Seeded CPU inputs and independent reference; no external tensors."""\nimport torch\nfrom recipe import make_fixtures as _build\n\ndef make_fixtures():\n    previous = torch.get_num_threads()\n    try:\n        torch.set_num_threads(4)\n        return _build()\n    finally:\n        torch.set_num_threads(previous)\n')
 return exporter.sources

def generate(entry,args,out):
 cid=entry['case_id'];bundle=Path(entry['bundle_path']);case=json.loads(Path(entry['case_path']).read_text());req=entry.get('required_data',[])
 if not req:
  (out/'fixture.py').write_text('"""This retained test generates seeded inputs and reference inline."""\ndef make_fixtures():\n    return {}\n')
  return [],None
 paths=' '.join(x['path'] for x in req);S=args.family_sources;B=args.fallback_bundle;seed=20261030
 def top(name,expr):
  p=S/name;body=cut(parse(p).body,"B, T, H",'files =')
  if not body:body=cut(parse(p).body,"gen = torch.Generator",'files =')
  return p,body+[ret(expr)]
 if '/data/ttt_h/' in paths:
  p,body=top('generate_ttt_h.py',"{name+'.pt': value.contiguous() for name,value in dict(k=k,v=v,w=w,b=b,eta=eta,h0=h0,hb0=hb0,output_0=hs,output_1=hbs,output_2=vnew,output_3=state,output_4=bias).items()}")
 elif '/data/ttt_o/' in paths:
  p,body=top('generate_ttt_o.py',"dict([('q.pt',q),('k.pt',k),('v_new.pt',v),('eta.pt',eta),('h.pt',h),('hb.pt',hb),('output.pt',result)])")
 elif '/data/ttt_dv/' in paths:
  p,body=top('generate_ttt_dv.py',"dict([('q.pt',q),('k.pt',k),('eta.pt',eta),('do_tensor.pt',do),('output.pt',result)])")
 elif '/data/matmul/' in paths:
  p,body=top('generate_matmul.py',"{'input_a.pt':a,'input_b.pt':b,'output.pt':c}")
 elif '/data/inverse32/' in paths:
  p=S/'generate_inverse.py';loop=next(n for n in parse(p).body if isinstance(n,ast.For));body=ast.parse('B,T,H,BT,cid=2,500,4,32,"cv_27eb4c65ae93688c"').body+cut(loop.body,'gen =','files =')+[ret("{'input_A_input.pt':inputs,'output.pt':output}")];seed=20261062
 elif '/data/merge_inverse/' in paths:
  p=B/'blue_adapters/data/merge_inverse/generate.py';body=cut(source_function(p).body,'torch.set_num_threads','inp, out =')+[ret("{'input_A_input.pt':a,'output.pt':output}")]
 elif '/data/kkt/' in paths or '/data/oja/' in paths:
  p=S/'generate.py';target='oja' if '/data/oja/' in paths else 'kkt';body=[]
  for n in cut(source_function(p).body,'gen =','NONEXISTENT'):
   if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and named(n.value.func)=='save_group':
    if ast.literal_eval(n.value.args[1])==target:
     body.append(ret("{name+'.pt':value.contiguous() if value.ndim else value for name,value in "+named(n.value.args[2])+'.items()}'));break
   else:body.append(n)
 elif '/data/kkt_original/' in paths or '/data/oja_original/' in paths:
  p=S/'generate_original_specs.py';body=cut(parse(p).body,'seed =','NONEXISTENT');body=[n for n in body if not isinstance(n,ast.FunctionDef) or n.name!='save']; new=[]
  target='oja' if '/data/oja_original/' in paths else 'kkt'
  for n in body:
   if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and named(n.value.func)=='save':
    if ast.literal_eval(n.value.args[0])==target:
     new.append(ret("{name+'.pt':value.contiguous() for name,value in "+named(n.value.args[1])+'.items()}'));break
   else:new.append(n)
  body=new
 elif any('/data/'+s+'/' in paths for s in ['attention','nsa','compression','gla_da','bwd_dh_parallel','bwd_dv_local']):
  table={'cv_ac478464b3ca205c':('attention','torch.manual_seed','dest =',"{'test_parallel_attn_fwd_kernel_cuda.pt':d}"),
  'cv_b1f04ab9f7e3a514':('compression','torch.manual_seed','directory =',"{name+'.pt':value for name,value in dict(q=q,k=k,v=v,o=o,lse=lse).items()}"),
  'cv_bb52bc5ec4f30103':('nsa','torch.manual_seed','folder =',"{name+'.pt':value for name,value in dict(q=q,k=k,v=v,block_indices=indices,o=o,lse=lse).items()}"),
  'cv_b38e5b96503bede4':('gla_da','gen =','for name, tensor',"{'input_v.pt':v,'input_do.pt':do,'output_dA.pt':out}"),
  'cv_e0959decca2a0527':('bwd_dv_local','rng =','torch.save',"{name:{'input_data':inputs,'gpu_output':{'dv':dv}}}"),
  'cv_5c136e24ece61c5d':('bwd_dh_parallel','g = torch.Generator','torch.save',"{name:{'input_data':inputs,'gpu_output':{'dh':dh,'dh0':None}}}")}
  name,start,stop,expr=table[cid];p=B/'blue_adapters/matrix_generators'/f'{name}_generate.py';body=cut(source_function(p).body,start,stop)+[ret(expr)];seed=42 if name=='bwd_dv_local' else seed
 elif cid=='cv_151ab6d14390e39a':
  p=args.scope_generator;body=cut(source_function(p).body,'gen =','cv =')+[ret("{'pre_process_bwd_seed20261030.pt':{'input_data':v,'gpu_output':{'dhm':y}}}")]
 elif cid in ('cv_5b3052c9fe198967','cv_e77711f4b89aefdc'):
  p=bundle/'resume_cv_absolute/generate.py';kind='dhu' if cid=='cv_5b3052c9fe198967' else 'dau';filename=Path(req[0]['path']).name
  body=ast.parse(f"d=inputs({kind!r})\nr=reference({kind!r},d)\nreturn {{{filename!r}:dict(input_data=d,gpu_output=r,reference_origin='CPU float64 independent math; no GPU execution')}}").body;seed=20261031
 elif cid=='cv_f7fb4478b1a09555':
  p=bundle/'baseline_inline_gla/generate.py';body=cut(parse(p).body,'torch.set_num_threads','torch.save')+[ret("{'input.pt':inputs,'reference.pt':{k:v.float() for k,v in expected.items()}}")];seed=20261031
 elif cid in ('cv_4193ceff2bfaca02','cv_5fe1f04a88159d60','cv_b01a8c32b8e4ec54','cv_f2d87df65a5ec616','cv_db7364f91a3cf933','cv_fce8ce9300d4fcc2','cv_e5165b65b3929779'):
  dirname='cv_grid_complete_inputs' if cid=='cv_e5165b65b3929779' else 'cv_inputs';meta=json.loads((args.resume_data/dirname/cid/'metadata.json').read_text());proof=meta['source_proof'];ip=args.source/proof['generator'];_,capture=input_prefix(ip,proof['source_function'],meta['seed']);ex=PureExporter(out,[ip.parent]);ex.export(ip,{'make_inputs'},'input_impl',{'make_inputs':capture},cpu=True)
  seed=meta['seed'];bodytext="values=make_inputs()\nif 'data' in values and 'input_data' in values['data']:\n    data=values['data']\nelif 'input_data' in values:\n    data={'input_data':values['input_data']}\nelse:\n    data={'input_data':values.get('input_dict',values.get('inputs_dict'))}\n"
  if cid=='cv_e5165b65b3929779':
   bodytext+="data['grid']=(8,8)\nreturn {'inputs_synthetic.pt':data}\n";sources=ex.sources
  else:
   mapping={'cv_4193ceff2bfaca02':('chunk_state_absolute',f'reference({cid!r}, data["input_data"])[0]',None),'cv_5fe1f04a88159d60':('chunk_state_absolute',f'reference({cid!r}, data["input_data"])[0]',None),'cv_b01a8c32b8e4ec54':('dplr_o_absolute','reference(data["input_data"])[0]','o'),'cv_f2d87df65a5ec616':('dplr_h_absolute','reference(data["input_data"])[0]',None),'cv_db7364f91a3cf933':('chunk_reduction_absolute','reference(data["input_data"])[0]',None),'cv_fce8ce9300d4fcc2':('kkt_solve_absolute','reference(data["input_data"])[0]','A')}
   folder,call,key=mapping[cid];refp=args.resume_data/folder/'generate.py';tree=parse(refp);ref=specialize_function(tree,source_function(refp,'reference'),{'cid':cid});ex.export(refp,{'reference'},'reference_impl',{'reference':ref});bodytext+='gold='+call+'\n';bodytext+='data["gpu_output"]='+('{'+repr(key)+':gold}' if key else 'gold')+'\n';origin={'dplr_o_absolute':'Independent CPU64 block equations; gpu_output is original test schema key, no GPU execution','dplr_h_absolute':'CPU64 equations with original operand/output casts; gpu_output is legacy schema only','chunk_reduction_absolute':'Independent CPU recurrence with original weighted-value cast and FP32 partial stores; no GPU run'}.get(folder,'Independent CPU64; gpu_output is legacy schema only');bodytext+='data["reference_origin"]='+repr(origin)+'\n';bodytext+='return {'+repr(Path(req[0]['path']).name)+':data}\n';sources=ex.sources
  text='import torch\nfrom input_impl import make_inputs\n'+('from reference_impl import reference\n' if cid!='cv_e5165b65b3929779' else '')+'\ndef _build():\n'+''.join('    '+l+'\n' for l in bodytext.splitlines())+'\ndef make_fixtures():\n    previous=torch.get_num_threads()\n    try:\n        torch.set_num_threads(4)\n        return _build()\n    finally:\n        torch.set_num_threads(previous)\n';(out/'fixture.py').write_text(text);return sources,seed
 elif cid in ('cv_0d2aa087ce3142e3','cv_a1323defc2a3e115'):
  p=args.source/'tests/ops/ascend_ops/cv_operator/recompute_w_u_fwd_kernel/test_recompute_w_u_fwd_kernel_gpu.py'
  code="""B,T,H,D,BT=4,2048,8,64,64
rng=torch.Generator(device='cpu').manual_seed(20261030)
k=torch.randn(B,T,H,D,generator=rng)*.25
v=torch.randn(B,T,H,D,generator=rng)*.25
beta=torch.rand(B,T,H,generator=rng)
g_cumsum=(torch.rand(B,T,H,generator=rng)-.5)*.25
A=torch.randn(B,T//BT,H,BT,BT,generator=rng)*.01
A=(A.tril(-1)+torch.eye(BT)).permute(0,1,3,2,4).reshape(B,T,H,BT).contiguous()
def blocks(x):return x.reshape(B,T//BT,BT,H,-1).permute(0,1,3,2,4)
w=(blocks(A).double()@blocks(k*beta.unsqueeze(-1)*g_cumsum.exp2().unsqueeze(-1)).double()).float().permute(0,1,3,2,4).reshape(B,T,H,D).contiguous()
u=(blocks(A).double()@blocks(v*beta.unsqueeze(-1)).double()).float().permute(0,1,3,2,4).reshape(B,T,H,D).contiguous()
return {name+'.pt':value for name,value in dict(k=k,v=v,beta=beta,g_cumsum=g_cumsum,A=A,output_0=w,output_1=u).items()}
""";body=ast.parse(code).body
 elif cid=='cv_6d47e9de750f0917':
  p=args.source/'tests/ops/ascend_ops/cv_operator/chunk_rwkv6_fwd_cumsum_kernel/test_chunk_rwkv6_fwd_cumsum_kernel_cuda.py';seed=42
  body=ast.parse("""torch.manual_seed(42)
B,T,H,S=4,2048,4,256
g=torch.randn(B,T,H,S,dtype=torch.float16)
x=g.double().reshape(B,T//64,64,H,S)
i=x.cumsum(2);e=i-x
gi=(i*1.4427).float().reshape_as(g);ge=(e*1.4427).float().reshape_as(g)
return {'test_chunk_rwkv6_fwd_cumsum_kernel_cuda.pt':{'input_data':dict(g=g,chunk_size=64,scale=1.4427,B=B,T=T,H=H,S=S),'gpu_output':dict(gi=gi,ge=ge)}}
""").body
 else:raise ValueError('Unmapped family '+cid+' '+paths)
 return export(p,body,out),seed

def main():
 p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--families',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--family-sources',type=Path,required=True);p.add_argument('--fallback-bundle',type=Path,required=True);p.add_argument('--scope-generator',type=Path,required=True);p.add_argument('--resume-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 catalog={r['case_path']:r for r in json.loads(a.catalog.read_text())['entries']};targets=[r for r in json.loads(a.families.read_text())['cases'] if 'No generic upstream input prefix recipe' in r.get('reason','')];records=[]
 primary={r['case_id']:r['case_path'] for r in targets}
 seen={r['case_path'] for r in targets}
 targets += [dict(case_id=e['case_id'],case_path=e['case_path']) for e in catalog.values() if e.get('available') and e.get('group')=='CV' and e['case_id'] in primary and e['case_path'] not in seen]
 for target in targets:
  entry=catalog[target['case_path']];suffix='' if target['case_path']==primary[entry['case_id']] else '-'+hashlib.sha256(target['case_path'].encode()).hexdigest()[:12];d=a.output/(entry['case_id']+suffix);d.mkdir(exist_ok=True);row=dict(case_id=entry['case_id'],case_path=entry['case_path'],provider_dir=str(d))
  try:
   provenance,seed=generate(entry,a,d)
   for f in d.glob('*.py'):check_pure(parse(f));compile(f.read_text(),str(f),'exec')
   row.update(status='ready',reason='',generator_provenance=provenance,seed=seed)
  except Exception as exc:row.update(status='blocked',reason=type(exc).__name__+': '+str(exc))
  records.append(row)
 (a.output/'index.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({s:sum(r['status']==s for r in records) for s in ['ready','blocked']}));print('\n'.join(r['case_id']+' '+r['reason'] for r in records if r['status']=='blocked'))
if __name__=='__main__':main()

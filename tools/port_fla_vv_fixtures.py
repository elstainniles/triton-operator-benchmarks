"""Export existing pure CPU VV fixture recipes without external tensor files.

This importer reads local evidence to select the original recipe. Generated
fixture.py modules only require PyTorch and return all tensor payloads in memory.
"""
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path


def selected_source(path, functions=(), constants=(), extra=''):
    tree = ast.parse(Path(path).read_text())
    body = [n for n in tree.body if
            (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in functions)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    return extra + '\n' + ast.unparse(ast.Module(body=body, type_ignores=[])) + '\n'


def specs(path):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SPECS' for t in n.targets))
    return eval(compile(ast.Expression(node.value), str(path), 'eval'), {'__builtins__': {}, 'dict': dict})


def build_provider(entry, fallback, synthetic_providers):
    b = Path(entry['bundle_path']); case = json.loads(Path(entry['case_path']).read_text()); cid = case['id']; kernel=entry['kernel']
    paths=[x['path'] for x in case.get('required_data',[])]; pathtext=' '.join(paths)
    def source(rel):
        p=b/rel
        return p if p.is_file() else fallback/rel
    prefix='"""Fixed-seed CPU inputs and independent references; no external tensor files."""\nimport torch\n'
    for family, funcs in [('resume_more',('delta_backward','build')),('resume_diagnostic',('build',)),('resume_scope_data',('dplr_matrices','build'))]:
        p=source(family+'/generate.py')
        if not p.exists():continue
        for kind,spec in specs(p).items():
            if cid not in spec['ids']:continue
            code=prefix
            sources=[p]
            if family=='resume_scope_data':
                ref=source(family+'/cpu_refs.py'); code+=selected_source(ref,('fused_recurrent_rwkv4_backward_pytorch_fp64',));sources.append(ref)
            code+=selected_source(p,funcs,('SEED',))
            if kind=='copy_last' and any('T1055' in x for x in paths):
                code+='''\ndef make_fixtures():
    original, _ = build('copy_last')
    gen=torch.Generator().manual_seed(20261031)
    data={}
    for name,value in original['input_data'].items():
        if isinstance(value,torch.Tensor):
            shape=list(value.shape)
            if name in ('q','k','v','g','level_scales'):
                shape[1]=1055
                data[name]=torch.randn(shape,generator=gen,dtype=value.dtype)
            else:data[name]=value.clone()
        else:data[name]=value
    data['T']=1055
    gold={name+'_prev':data[name][:,1024:].clone() for name in ('q','k','v','g','level_scales')}
    return {'test_copy_last_chunk_kernel_cuda.pt':dict(input_data=data,cpu_reference=gold,defined_rows=31,seed=20261031)}
'''
                return code,'copy_last_T1055',sources
            if kind=='gla_intra':
                code+='''\ndef make_fixtures():
    inputs, _ = build('gla_intra')
    q,k,g=(inputs[name+'.pt'].double() for name in ('q','k','g'))
    B,T,H,K=q.shape;BT,BC=64,16
    expected=torch.full((B,T,H,BT),0.375,dtype=torch.float64)
    mask=torch.zeros_like(expected,dtype=torch.bool)
    for start in range(0,T,BC):
        offset=start%BT
        qq=(q[:,start:start+BC]*g[:,start:start+BC].exp2()).transpose(1,2)
        kk=(k[:,start:start+BC]*(-g[:,start:start+BC]).exp2()).transpose(1,2)
        block=(qq@kk.transpose(-1,-2)).tril().transpose(1,2)*(K**-.5)
        expected[:,start:start+BC,:,offset:offset+BC]=block
        mask[:,start:start+BC,:,offset:offset+BC]=True
    values=expected.clone();values[~mask]=float('nan')
    inputs['gla-reference-defined.pt']={'expected':expected,'written_mask':mask}
    inputs['reference_cpu64.pt']={'expected_written':values,'valid_mask':mask,'undefined_regions':'All cross-16x16-block A entries are not initialized/written by the original test wrapper. NaN here is a marker, not a numeric reference.'}
    return inputs
'''
                return code,'gla_exp2_defined_writes',sources
            name=spec['file']
            code+='\ndef make_fixtures():\n    inputs, reference = build('+repr(kind)+')\n'
            if name:code+='    values = {'+repr(name)+': inputs}\n'
            else:code+='    values = dict(inputs)\n'
            if kind == 'gate_bwd':
                code += "    d=inputs['input_data']\n    r,k,v,dy,g=(d[n].double() for n in ('r','k','v','grad_output','g'))\n    reference['grad_r_k_intermediate']=((dy*g).reshape_as(v)*v).sum(-1,keepdim=True)*r*k\n"
            code+="    values['reference_cpu64.pt'] = reference\n    return values\n"
            return code,family+'/'+kind,sources
    p=source('resume_data/generate.py')
    if p.exists() and cid in specs(p):
        kind,name,_=specs(p)[cid]
        code=prefix+selected_source(p,('inputs','reference'),('SEED',))
        code+='\ndef make_fixtures():\n    data = inputs('+repr(kind)+')\n    expected = reference('+repr(kind)+', data)\n    return {'+repr(name)+": dict(input_data=data, gpu_output=expected, reference_origin='synthetic_cpu_float64_math_not_GPU')}\n"
        return code,'resume_data/'+kind,[p]
    if 'token_shift_' in kernel:
        kind='short' if 'short' in kernel else 'long'
        shape=(8,4096,4096) if kind=='short' else (4,8192,4096)
        p=source('extra_adapters/token_shift/generate.py')
        # Same seed/draw order/formulas as source run(); tensor serialization removed.
        code=prefix+f'''\ndef make_fixtures():
    shape = {shape!r}
    b,t,d = shape
    rng = torch.Generator(device='cpu').manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32, generator=rng)
    dy = torch.randn(shape, dtype=torch.float32, generator=rng)
    y = torch.empty_like(x)
    y[:,0,:] = -x[:,0,:]
    torch.sub(x[:,:-1,:], x[:,1:,:], out=y[:,1:,:])
    dx = torch.empty_like(dy)
    dx[:,-1,:] = -dy[:,-1,:]
    torch.sub(dy[:,1:,:], dy[:,:-1,:], out=dx[:,:-1,:])
    return {{'token_shift_fwd_kernel_{kind}_in.pt': {{'x': x}},
            'token_shift_bwd_kernel_{kind}_in.pt': {{'x': x, 'dy': dy}},
            'token_shift_fwd_kernel_{kind}_out.pt': {{'y': y, 'N': b, 'T_actual': t, 'use_short_kernel': {kind=='short'!r}, 'cache_out': None}},
            'token_shift_bwd_kernel_{kind}_out.pt': {{'dx': dx, 'grad_cache': None}}}}
'''
        return code,'token_shift/'+kind,[p]
    if cid=='vv_224ba42dda9e9c03':
        p=source('extra_adapters/hgrn/generate.py')
        code=prefix+selected_source(p,('make_inputs','reference_closed_form'),('SEED','BT','SHAPE'))
        code+="\ndef make_fixtures():\n    x, g, h0 = make_inputs()\n    out, cum = reference_closed_form(x, g, h0)\n    return {'x.pt': x, 'g.pt': g, 'h0.pt': h0, 'o.pt': out, 'gc.pt': cum}\n"
        return code,'hgrn',[p]
    if cid=='vv_27318c79ea37c398':
        p=source('extra_adapters/gsa_inference/generate.py')
        t=ast.parse(p.read_text());assign=next(n for n in t.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple))
        code=prefix+'import math\n'+ast.unparse(assign)+'\n'+selected_source(p,('make_inputs','reference'),('SEED','NAME'))
        code+="\ndef make_fixtures():\n    inputs = make_inputs()\n    expected = {name: value.float() for name, value in reference(inputs).items()}\n    return {NAME: {'input_data': inputs, 'gpu_output': expected}}\n"
        return code,'gsa_inference',[p]
    if cid=='vv_bbb761829a323b97':
        p=source('extra_adapters/kda_backward/generate.py')
        code=prefix+'import torch.nn.functional as F\n'+selected_source(p,('make_inputs','reference_autograd'),('SEED','SHAPE','FILES'))
        code+="\ndef make_fixtures():\n    g, a, bias, dy = make_inputs()\n    dg, da, dbias, _ = reference_autograd(g, a, bias, dy)\n    return dict(zip(FILES, (g, a, bias, dy, dg, da, dbias), strict=True))\n"
        return code,'kda_backward',[p]
    if cid=='vv_527d5af7786b1beb':
        p=source('blue_adapters/cumsum_global/generate.py')
        code=prefix+"\ndef make_fixtures():\n    gen=torch.Generator(device='cpu').manual_seed(42)\n    x=torch.randn((4,1024,8),dtype=torch.float32,generator=gen)\n    y=x.cumsum(dim=1)\n    return {'input.pt':x,'reference.pt':y}\n"
        return code,'cumsum_global',[p]
    if cid in ('vv_4aeab0ff686343bf','vv_646cc49b043e1def'):
        kind='short' if cid=='vv_4aeab0ff686343bf' else 'long';T,BD,BT=(19,128,None) if kind=='short' else (35,32,16)
        p=source('blue_adapters/k_update/generate.py')
        code=prefix+f'''\ndef make_fixtures():
    g=torch.Generator(device='cpu').manual_seed(20261030)
    B,T,D,BD,BT = 2,{T},70,{BD},{BT!r}
    k=torch.randn(B,T,D,generator=g,dtype=torch.float32)*.2
    a=.2+torch.rand(B,T,D,generator=g,dtype=torch.float32)
    ka=.1+.6*torch.rand(D,generator=g,dtype=torch.float32)
    expected=k.double()*(1+(a.double()-1)*ka.double())
    data=dict(k=k,a=a,ka=ka,B=B,T=T,D=D,BD=BD)
    if BT is not None: data.update(BT=BT,NT=(T+BT-1)//BT)
    payload=dict(input_data=data,gpu_output={{'out':expected.float()}},reference_origin='CPU float64 independent; gpu_output is historical test key only')
    return {{'test_k_update_fwd_kernel_{kind}_cuda.pt':payload}}
'''
        return code,'k_update/'+kind,[p]
    if cid=='vv_79c05e2a6845bf60':
        p=source('blue_adapters/extra_preprocess/prepare.py')
        code=prefix+selected_source(p,('make',),('SEED','SHAPE','FILE'))+"\ndef make_fixtures():\n    return {FILE: make()}\n"
        return code,'attn_preprocess',[p]
    if cid in ('vv_c55f012f45d8f495','vv_a679392925bf0d06'):
        p=source('blue_adapters/extra_preprocess/addcmul.py')
        code=prefix+selected_source(p,('data',),('SEED','SHAPE','NAMES'))+"\ndef make_fixtures():\n    return data()\n"
        return code,'addcmul',[p]
    if cid=='vv_7f19885874663d00':
        p=source('blue_adapters/copy_input/copy_input_data.py');kind='tail' if 'copy_input_tail' in pathtext else 'full'
        code=prefix+selected_source(p,('make',),('SEED','FIELDS'))+'\ndef make_fixtures():\n    return {"test_copy_input_kernel_cuda.pt": make('+repr(kind)+')}\n'
        return code,'copy_input/'+kind,[p]
    if cid=='vv_838d1377c54356dc':
        p=source('blue_adapters/cumsum_vector/cumsum_vector_data.py');kind='local_channel_tail' if 'cumsum_local_tail' in pathtext else 'local_original'
        shape=(2,256,3,129) if kind=='local_channel_tail' else (4,2048,8,1024)
        code=prefix+f'''\ndef make_fixtures():
    gen=torch.Generator(device='cpu').manual_seed(42)
    b,t,h,d={shape!r}
    x=torch.randn((b,t,h,d),dtype=torch.float32,generator=gen)
    y=x.reshape(b,t//128,128,h,d).cumsum(2).reshape_as(x)
    return {{'input.pt':{{'s':x}},'reference.pt':{{'gpu_output':{{'output':y}}}}}}
'''
        return code,'cumsum_vector/'+kind,[p]
    if cid in ('vv_02b9eb85959f5dea','vv_c17619d444266ce0'):
        p=source('remaining_vv_data/generate_reference.py');kind='layernorm' if cid=='vv_02b9eb85959f5dea' else 'rwkv6'
        code=prefix+selected_source(p,(kind+'_reference',))
        code+='''\ndef make_fixtures():
    rng=torch.Generator(device='cpu').manual_seed(20261030)
    def randn(*shape,dtype=torch.float32):
        return torch.randn(*shape,generator=rng,dtype=dtype,device='cpu')
'''
        if kind=='layernorm':
            code+='''    x0=randn(2048,1024)
    inputs={'x':x0,'dy':randn(2048,1024),'weight':randn(1024),'bias':randn(1024),
            'mean':x0.mean(-1),'rstd':(x0.var(-1,unbiased=False)+1e-5).rsqrt()}
'''
        else:
            code+='''    shape=(4,2048,4,256)
    inputs={name:randn(*shape,dtype=torch.float16) for name in ('q','k','v','do')}
    inputs.update(w=torch.nn.functional.logsigmoid(randn(*shape)),u=randn(4,256,dtype=torch.float16),
                  initial_state=randn(4,4,256,256,dtype=torch.float16),scale=256**-0.5,reverse=False)
'''
        opname='layernorm' if kind=='layernorm' else 'rwkv6_dq'
        code+='    payload=dict(operation='+repr(opname)+",input_origin='synthetic_cpu',seed=20261030,gpu_reference_created=False,inputs=inputs)\n    return {'inputs.pt':payload,'reference.pt':"+kind+'_reference(inputs)}\n'
        return code,'remaining_vv/'+kind,[p]
    if cid in ('vv_4e7b5a0967a2b961','vv_6c97bc8a9d6edaae'):
        p=source('synthetic/generate_data.py'); tree=ast.parse(p.read_text())
        # Use the independently ported synthetic provider for exactly its fixed
        # seed/input draw order, then append the existing LSE reference extension.
        candidates=sorted(Path(synthetic_providers).glob('grpo_fwd-*/fixture.py'))
        provider=candidates[0] if len(candidates)==1 else Path('missing-grpo-provider')
        if not provider.is_file():return None,'grpo_base_provider_not_yet_exported',[p]
        code=provider.read_text().replace('def make_fixtures(', 'def base_fixtures(',1)
        code+='''\ndef make_fixtures():
    import torch
    values=base_fixtures()
    v=values['grpo_fwd_kernel_in.pt'];old=values['grpo_fwd_kernel_out.pt']
    lse=torch.zeros((2,1024),dtype=torch.float32)
    for b in range(2):
        for start in range(0,1024,32):
            x=v['logits'][b,start:start+32].double();m=x.max(-1).values
            z=m+(x-m[:,None]).exp().sum(-1).log()
            active=v['completion_mask'][b,start:start+32].bool()
            lse[b,start:start+32]=torch.where(active,z,0).float()
    values['input.pt']=v
    values['reference.pt']=dict(loss=old['loss'],independent_kl=old['independent_kl'],packed_false=old['loss'],packed_true=torch.cat((old['loss'],old['independent_kl']),0),lse=lse)
    return values
'''
        return code,'grpo_fwd_extended',[p,source('blue_adapters/grpo/generate.py')]
    return None,'unmapped_generator',[]


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--exclude',type=Path,required=True);p.add_argument('--fallback-bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--synthetic-providers',type=Path,required=True);args=p.parse_args()
    skip={x['case_path'] for x in json.loads(args.exclude.read_text())}
    entries={e['case_path']:e for e in json.loads(args.catalog.read_text())['entries'] if e.get('available') and e['group']=='VV' and (e['case_path'] not in skip or e['case_id'] in ('vv_4e7b5a0967a2b961','vv_6c97bc8a9d6edaae'))}
    args.output.mkdir(exist_ok=True,parents=True);records=[];gaps=[]
    for casepath,e in entries.items():
        try:
            code,recipe,sources=build_provider(e,args.fallback_bundle,args.synthetic_providers)
            if code is None:
                gaps.append(dict(case_path=casepath,case_id=e['case_id'],reason=recipe));continue
            if recipe in ('resume_more/mesa','resume_scope_data/naive_attn'):
                code = code.replace('def make_fixtures():', 'def _make_fixtures():', 1)
                code += "\ndef make_fixtures():\n    previous_threads = torch.get_num_threads()\n    try:\n        torch.set_num_threads(4)\n        return _make_fixtures()\n    finally:\n        torch.set_num_threads(previous_threads)\n"
            ast.parse(code)
            if any(x in code for x in ('torch.load','torch.save','/home/')):raise ValueError('External tensor IO or machine path remains')
            key=hashlib.sha256(code.encode()).hexdigest()[:16];dest=args.output/e['case_id']/key;dest.mkdir(exist_ok=True,parents=True);(dest/'fixture.py').write_text(code)
            records.append(dict(case_path=casepath,case_id=e['case_id'],recipe=recipe,fixture=str(dest/'fixture.py'),sha256=hashlib.sha256(code.encode()).hexdigest(),source_generators=[dict(path=str(s),sha256=hashlib.sha256(s.read_bytes()).hexdigest()) for s in sources],cpu_validation='pending',device_execution=False))
        except Exception as error:gaps.append(dict(case_path=casepath,case_id=e['case_id'],reason=repr(error)))
    (args.output/'index.json').write_text(json.dumps([dict(case_path=r['case_path'],provider_dir=str(Path(r['fixture']).parent),status='ready',reason='',generator_provenance=r['source_generators'],seed=42 if r['recipe'].startswith(('token_shift','cumsum_')) else 20261031 if r['recipe'].startswith('resume_') else 20261030) for r in records]+[dict(case_path=g['case_path'],provider_dir=None,status='blocked',reason=g['reason'],generator_provenance=[],seed=None) for g in gaps],indent=2)+'\n')
    (args.output/'mapping.json').write_text(json.dumps(records,indent=2)+'\n');(args.output/'remaining-gaps.json').write_text(json.dumps(gaps,indent=2)+'\n');print(json.dumps(dict(mapped=len(records),unmapped=len(gaps),mapped_case_ids=len({r['case_id'] for r in records}))))
if __name__=='__main__':main()

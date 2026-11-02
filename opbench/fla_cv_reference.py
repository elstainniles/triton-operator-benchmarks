"""Device-free original-kernel semantics for the frozen CV cases."""
import hashlib
import json
import math
from pathlib import Path

from .fla_cv import load_case, load_kernel, selected, sha


def grid(case, kernel):
    import ast
    sc = case['scalar_choices']
    symbol = case['symbol']
    if symbol == 'safe_dot_kernel':
        return (1,)
    if symbol.startswith('softmax_'):
        return (math.prod(next(iter(case['tensors'].values()))['shape']) // sc['D'],)
    tree = ast.parse(Path(kernel.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == symbol)
    axes = {}
    for statement in fn.body:
        if not isinstance(statement, ast.Assign):
            continue
        lhs = statement.targets[0]
        names = lhs.elts if isinstance(lhs, ast.Tuple) else [lhs]
        rhs = statement.value.elts if isinstance(statement.value, ast.Tuple) else [statement.value]
        for name, value in zip(names, rhs):
            if (isinstance(name, ast.Name) and isinstance(value, ast.Call)
                    and ast.unparse(value.func) == 'tl.program_id'):
                axes[name.id] = ast.literal_eval(value.args[0])
    text = ast.unparse(fn)
    import re
    head = 'HV' if re.search(r'i_(?:bh|nh)\s*//\s*HV', text) else (
        'HQ' if re.search(r'i_(?:bh|nh)\s*//\s*HQ', text) else 'H')
    choices = {'i_t': math.ceil(sc['T']/sc['BT']), 'i_c': math.ceil(sc['T']/sc['BTL']),
               'i_bh': sc['B']*sc[head], 'i_nh': sc['B']*sc[head], 'i_b': sc['B'],
               'i_n': sc['B'], 'i_h': sc['H'], 'i_hq': sc['HQ'],
               'i_k': math.ceil(sc['K']/sc['BK']), 'i_v': math.ceil(sc['V']/sc['BV']),
               'i_s': math.ceil(sc['S']/sc['BS']),
               'i_kv': math.ceil(sc['K']/sc['BK'])*math.ceil(sc['V']/sc['BV']),
               'i_bc': sc['NC'], 'i_bg': sc['B']*sc['H']}
    if not axes:
        raise ValueError('No verified launch grid for ' + symbol)
    result = [1]*(max(axes.values())+1)
    for name, axis in axes.items():
        if name not in choices:
            raise ValueError('Unresolved grid dimension ' + name)
        result[axis] = choices[name]
    if symbol == 'chunk_transform_qk_fwd_kernel':
        result = [math.ceil(sc['T']/sc['BT']), sc['B']*sc['H']]
    result = tuple(result)
    if case.get('grid') and list(result) != case['grid']:
        raise ValueError('Grid differs from the frozen device-checked launch')
    return result


def make_inputs(case, seed):
    import torch
    rng = torch.Generator(device='cpu').manual_seed(seed)
    tensors = {}
    for name, info in case['tensors'].items():
        dtype = getattr(torch, info['dtype'])
        shape = tuple(info['shape'])
        if dtype in (torch.int32, torch.int64):
            value = torch.zeros(shape, dtype=dtype)
        elif dtype == torch.bool:
            value = torch.rand(shape, generator=rng) > .5
        else:
            value = torch.randn(shape, generator=rng)*.1
            if name in ('beta', 'eta'):
                value = torch.rand(shape, generator=rng)*.5+.1
            if name in ('g','gk','gv','g0','gi','ge','g_cumsum'):
                value = -torch.rand(shape, generator=rng)*.25
            if name in ('r','l','lse','z'):
                value = value.abs()+1
            value = value.to(dtype)
        tensors[name] = value
    return tensors


def arguments(case, tensors):
    return {key: tensors[value['tensor']] if isinstance(value, dict) else value
            for key, value in case['args'].items()}


def install_bf16_compatibility():
    import numpy as np
    import torch
    import triton.language as tl
    from triton.runtime.interpreter import interpreter_builder as ib, TensorHandle
    old_binary, old_cast = ib.binary_op, ib.cast_impl
    def decode(t):
        return (torch.from_numpy(t.data.copy()).view(torch.bfloat16).float().numpy()
                if t.dtype.scalar == tl.bfloat16 else t.data)
    def encode(a):
        return torch.from_numpy(np.asarray(a, dtype=np.float32).copy()).to(torch.bfloat16).view(torch.uint16).numpy()
    def binary(lhs, rhs, op):
        if lhs.dtype.scalar != tl.bfloat16 and rhs.dtype.scalar != tl.bfloat16:
            return old_binary(lhs, rhs, op)
        value = op(decode(lhs), decode(rhs))
        if value.dtype == np.bool_:
            return TensorHandle(value, tl.int1)
        return TensorHandle(encode(value), tl.bfloat16)
    def cast(src, dst_type):
        if src.dtype.scalar == tl.bfloat16 and dst_type.scalar != tl.bfloat16:
            return old_cast(TensorHandle(decode(src), tl.float32), dst_type)
        if dst_type.scalar == tl.bfloat16:
            return TensorHandle(encode(decode(src)), tl.bfloat16)
        return old_cast(src, dst_type)
    ib.get_bf16 = lambda value: TensorHandle(encode([value]), tl.bfloat16)
    ib.binary_op, ib.cast_impl = binary, cast


def compute_reference(case, tensors, kernel, launch_grid):
    import numpy as np
    import torch
    import triton.language as tl
    from triton.runtime.jit import JITFunction
    from triton.runtime.interpreter import InterpretedFunction, interpreter_builder as ib, TensorHandle, _get_np_dtype, _convert_float
    if any(t.dtype == torch.bfloat16 for t in tensors.values()):
        install_bf16_compatibility()
    if case['reference_exp'] == 'fp64_then_declared_dtype':
        ib.create_exp = lambda value: TensorHandle(
            np.exp(value.data.astype(np.float64)).astype(value.data.dtype), value.dtype.scalar)
    for name, fn in list(vars(kernel).items()):
        if isinstance(fn, JITFunction):
            setattr(kernel, name, InterpretedFunction(fn.fn))
    data = {key: value.clone() for key, value in tensors.items()}
    regions = [(t.data_ptr(), t.data_ptr()+t.numel()*t.element_size(), t.element_size(), key)
               for key, t in data.items()]
    writes = {}
    old_load, old_store, old_dot = ib.create_masked_load, ib.create_masked_store, ib.create_dot
    def check(ptr, mask, write):
        addresses = ptr.data[np.asarray(mask if isinstance(mask, np.ndarray) else mask.data, dtype=bool)]
        if addresses.size == 0:
            return
        lo, hi = int(addresses.min()), int(addresses.max())
        found = [(base,end,item,name) for base,end,item,name in regions if base <= lo and hi < end]
        if len(found) != 1:
            raise ValueError('OUT_OF_BOUNDS ' + ('store' if write else 'load'))
        base,end,item,name = found[0]
        if hi+item > end or np.any((addresses-base)%item):
            raise ValueError('Misaligned pointer')
        if write:
            flags = writes.setdefault(name, np.zeros(data[name].numel(), dtype=bool))
            flags[((addresses-base)//item).astype(np.int64)] = True
    def ld(ptr, mask, *args):
        if isinstance(mask, np.ndarray):mask = TensorHandle(mask, tl.int1)
        check(ptr, mask, False)
        return old_load(ptr, mask, *args)
    def st(ptr, value, mask, *args):
        if isinstance(mask, np.ndarray):mask = TensorHandle(mask, tl.int1)
        check(ptr, mask, True)
        return old_store(ptr, value, mask, *args)
    def dot(a,b,d,*args):
        def floating(t):
            if t.dtype == tl.bfloat16:
                return _convert_float(t.data,t.dtype,tl.float32,None).view(np.float32).astype(np.float64)
            return t.data.astype(np.float64)
        result = (floating(a)@floating(b)+d.data.astype(np.float64)).astype(_get_np_dtype(d.dtype))
        return TensorHandle(result,d.dtype.scalar)
    try:
        ib.create_masked_load, ib.create_masked_store, ib.create_dot = ld, st, dot
        getattr(kernel, case['symbol'])[launch_grid](**arguments(case,data))
    finally:
        ib.create_masked_load, ib.create_masked_store, ib.create_dot = old_load, old_store, old_dot
    if not writes:
        raise ValueError('CPU kernel wrote no outputs')
    if not all(torch.isfinite(data[name]).all() for name in writes):
        raise ValueError('Nonfinite CPU reference')
    return data, writes


def compare(actual, expected, writes):
    import torch
    checks = []
    for name, reference in expected.items():
        got = actual[name].detach().cpu()
        rtol,atol = {'float16':(1e-3,1e-3),'bfloat16':(5e-3,5e-3),
                     'float32':(1e-5,1e-5),'float64':(1e-7,1e-7)}.get(str(reference.dtype).split('.')[-1],(0,0))
        if name not in writes:
            rtol = atol = 0
        diff = (got.double()-reference.double()).abs()
        bad = ~torch.isfinite(got.double()) | ~torch.isfinite(reference.double()) | (diff > atol+rtol*reference.double().abs())
        checks.append(dict(name=name,passed=not bool(bad.any()),elements=reference.numel(),
                           mismatches=int(bad.sum()),max_abs=float(diff.max()),rtol=rtol,atol=atol))
    return checks


def reference_worker(args):
    import torch
    row = selected(args.cases)[0]
    if len(selected(args.cases)) != 1:
        raise ValueError('Reference process must select exactly one row')
    path, case = load_case(row)
    if case is None:
        raise ValueError('No exact source for this row')
    kernel = load_kernel(path,case)
    launch_grid = grid(case,kernel)
    tensors = make_inputs(case,args.seed)
    expected,writes = compute_reference(case,tensors,kernel,launch_grid)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    payload = args.output/'reference.pt'
    torch.save(dict(expected=expected,writes={k:torch.from_numpy(v) for k,v in writes.items()}),payload)
    meta = dict(row=case['row'],seed=args.seed,grid=launch_grid,
                source_sha256=case['source_sha256'],module_sha256=case['module_sha256'],
                input_sha256={k:hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in tensors.items()},
                data_sha256=sha(payload),outputs={k:int(v.sum()) for k,v in writes.items()},
                method='original kernel CPU semantics with bounded memory; FP64 dot',
                exp_policy=case['reference_exp'])
    (args.output/'reference.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('input_sha256',)}))

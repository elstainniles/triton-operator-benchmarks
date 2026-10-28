"""Regenerate the public, deterministic synthetic workload specifications (no Torch)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opbench.catalog import operators


def cases_for(operator):
    seed = 20261028
    def case(name, shape, dtype, tile, suite='default', **extra):
        return dict(id=name, shape=shape, dtype=dtype, tile=tile, suite=suite,
                    seed=seed, input_origin='generated_benchmark', **extra)
    if operator in ('TG01', 'TG02'):
        tile = {'BLOCK_D_OUTER':128,'BLOCK_D_INNER':128} if operator=='TG01' else {'BLOCK_D':128}
        return [case('smoke', [16,128,24,4], 'float32', tile, 'smoke'),
                case('wide-fp16', [256,4096,512,8], 'float16', tile),
                case('wide-bf16', [256,4096,512,8], 'bfloat16', tile,
                     **({'inactive_rows':3} if operator=='TG01' else {'clamp_max':True}))]
    if operator == 'TG03':
        tile = {'BLOCK_M':16,'BLOCK_N':32,'BLOCK_K':32}
        return [case('smoke', [32,64,64], 'float8_e4m3fn', tile, 'smoke'),
                case('tail-bias', [33,65,96], 'float8_e4m3fn', tile, use_bias=True),
                case('large', [256,512,512], 'float8_e4m3fn', tile)]
    if operator in ('TG04','TG05','TG06'):
        options = dict(launch_options={'num_warps':4,'num_stages':2,'force_simt_only':True})
        return [case('smoke', [1,2,32], 'float32', {'CHUNK_SIZE':16}, 'smoke', **options),
                case('upstream-shape', [16,32,2048], 'float32', {'CHUNK_SIZE':16},
                     input_origin_note='Upstream dimensions/distribution, new deterministic CPU seed', **options)]
    dims = [(1,64,64,2,1),(1,128,128,4,1),(2,64,128,4,2),(4,128,128,4,2),
            (8,128,256,8,2),(8,256,128,4,4),(8,512,512,8,2),(16,256,256,8,4),
            (16,512,256,16,2),(32,256,256,8,2),(32,512,512,16,4),(32,1024,256,8,2),
            (64,256,256,8,4),(64,512,512,16,2),(64,1024,512,32,4),(128,512,256,8,2),
            (128,1024,512,16,4),(256,512,256,8,4),(1,1024,1024,32,1),
            (64,128,1024,16,4),(32,2048,256,8,2),(16,1024,2048,32,4),
            (128,256,1024,16,2),(32,256,256,8,1),(64,256,256,32,8)]
    modes = ['int8_w8a16','block_int8_w8a8','fp8_w8a8_tensor','no_quant']
    tile = dict(BLOCK_SIZE_M=32,BLOCK_SIZE_N=64,BLOCK_SIZE_K=64,GROUP_SIZE_M=4)
    result=[]
    for j, mode in enumerate(modes):
        result.append(case('smoke-'+mode, [4,64,64,4,2], 'float16', tile, 'smoke',
                           mode=mode, output_dtype='float16', compute_dtype='float16',
                           c_sorted=j%2==0, scale_dtype='float32', mul_routed_weight=True))
    for i, dims_i in enumerate(dims):
        for j, mode in enumerate(modes):
            dtype = 'float16' if j==0 else ['float16','bfloat16'][(i+j)%2]
            compute = 'float16' if j==0 else ['float16','float32'][(i*2+j)%2]
            result.append(case(f'case-{i*4+j:03}', list(dims_i), dtype, tile, mode=mode,
                               output_dtype=dtype, compute_dtype=compute,
                               scale_dtype='float32' if compute=='float32' else 'float16',
                               c_sorted=(i+j)%5==0, mul_routed_weight=(i+j)%3!=0,
                               upstream_dimension_case_id=i*4+j))
    return result


def main():
    for entry in operators():
        if entry.get('runner') == 'graph_tests':
            path = ROOT / entry['path'] / 'cases.json'
            path.write_text(json.dumps(dict(schema_version=1, input_origin='generated_benchmark',
                external_input_required=False, cases=cases_for(entry['id'])), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

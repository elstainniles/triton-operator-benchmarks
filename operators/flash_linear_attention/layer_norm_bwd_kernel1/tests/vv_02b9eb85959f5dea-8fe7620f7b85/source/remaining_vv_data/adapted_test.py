
"""Load fixed original-shape inputs and call unchanged original FLA wrappers.

Both original validate_cmp primary assertions are preserved. Missing historical
GPU-error fallback data are not synthesized; primary rejection stays a failure.
"""
from opbench.fla_fixtures import load_fixture
import importlib
import json
from pathlib import Path
import torch
from .comparators import validate_rwkv6, validate_layernorm

def run(name, load_dir, device='npu'):
    folder = Path(load_dir)
    values = load_fixture(folder / 'inputs.pt', map_location='cpu', weights_only=False)['inputs']
    reference = load_fixture(folder / 'reference.pt', map_location='cpu', weights_only=False)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in values.items()}
    definition = json.loads((Path(__file__).parent / 'source-proof.json').read_text())['definitions'][name]
    wrapper = getattr(importlib.import_module(definition['module']), definition['wrapper'])
    if name == 'rwkv6':
        dq, dq1 = wrapper(**inputs)
        for value, key in ((dq, 'dq'), (dq1, 'dq1')):
            validate_rwkv6('float32', value, reference[key], device_type=device)
    elif name == 'layernorm':
        dx, dw, db, dres = wrapper(**inputs)
        if dres is not None:
            raise AssertionError('original no-residual fixture unexpectedly returns dres')
        for value, key in ((dx, 'dx'), (dw, 'dw'), (db, 'db')):
            validate_layernorm('float32', value, reference[key], device_type=device)
    else:
        raise ValueError('unknown remaining VV fixture')

def test_rwkv6(load_dir):
    return run('rwkv6', load_dir)

def test_layernorm(load_dir):
    return run('layernorm', load_dir)

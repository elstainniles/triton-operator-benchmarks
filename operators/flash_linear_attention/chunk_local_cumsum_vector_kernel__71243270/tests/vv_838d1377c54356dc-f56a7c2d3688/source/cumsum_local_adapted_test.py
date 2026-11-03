
"""Input-file wrapper for FLA's original local vector cumsum test.

The original non-A2/A3 test generates its own input and compares with CPU
torch.cumsum. This wrapper loads a pre-generated input/reference so two TA
environments consume identical bytes. Kernel and FLA launcher are extracted.
"""
from opbench.fla_fixtures import load_fixture
from pathlib import Path
import torch
from yb_712432704460d6 import chunk_local_cumsum_vector

def test_chunk_local_cumsum_vector_prepared(load_dir):
    directory = Path(load_dir)
    x = load_fixture(directory / 'input.pt', map_location='cpu', weights_only=False)['s']
    expected = load_fixture(directory / 'reference.pt', map_location='cpu', weights_only=False)['gpu_output']['output']
    soc = torch.npu.get_device_name(torch.npu.current_device())
    if '910B' in soc or '910_93' in soc:
        raise RuntimeError('original A2/A3 acceptance requires actual GPU error denominator; synthetic CPU reference cannot replace it')
    out = chunk_local_cumsum_vector(x.to('npu'), 128).cpu()
    torch.testing.assert_close(expected, out, rtol=1e-05, atol=1e-05, equal_nan=True)

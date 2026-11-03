
"""Original GLA wrapper/kernel with fixed inputs and independent added checks."""
from opbench.fla_fixtures import load_fixture
from pathlib import Path
import torch
from yb_62b4975145faac import chunk_gla_bwd_dqkg

def test_original_gla_inter_reference(load_dir):
    root = Path(load_dir)
    inputs = load_fixture(root / 'input.pt', map_location='cpu', weights_only=False)
    expected = load_fixture(root / 'reference.pt', map_location='cpu', weights_only=False)
    values = {k: v.to('npu') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    actual = chunk_gla_bwd_dqkg(**values)
    for key, value in zip(('dq2', 'dk2', 'dg'), actual):
        torch.testing.assert_close(value.cpu(), expected[key], rtol=1e-05, atol=1e-05, equal_nan=False)


"""Preserve original wrapper and kernel; compare both actual kernel outputs.
The original wrapper prepends its initial state to state_out with torch.cat.
We capture that exact raw output before concatenation for direct kernel binding,
then use the original helper/tolerance and verify the prefix byte-for-byte.
"""
from opbench.fla_fixtures import load_fixture
from pathlib import Path
from unittest.mock import patch
import torch
from yb_83cc11308a4a36 import fused_recurrent_rwkv4_forward
from yb_22cded5b91bba3 import compare_data_precision, convert_tensor_with_device_type

def test_rwkv4(load_dir):
    data = load_fixture(Path(load_dir) / 'test_fused_recurrent_rwkv4_forward_cuda.pt', map_location='cpu', weights_only=False)
    inp = convert_tensor_with_device_type(data['input_data'], 'npu')
    captured = []
    cat = torch.cat

    def capture(tensors, *args, **kwargs):
        if len(tensors) == 2 and tensors[0] is inp['state'] and (kwargs.get('dim', args[0] if args else 0) == 2):
            captured.append(tensors[1])
        return cat(tensors, *args, **kwargs)
    with patch.object(torch, 'cat', capture):
        wkv, combined = fused_recurrent_rwkv4_forward(inp['w_processed'], inp['u'], inp['k'], inp['v'], inp['state'])
    assert len(captured) == 1, 'expected exactly one initial-state concatenation'
    raw = captured[0]
    compare_data_precision(data['gpu_output'], {'wkv': wkv, 'state_out_raw': raw}, device_type='cpu')
    assert torch.equal(combined[:, :, :1].cpu(), data['input_data']['state'])
    assert torch.equal(combined[:, :, 1:].cpu(), raw.cpu())

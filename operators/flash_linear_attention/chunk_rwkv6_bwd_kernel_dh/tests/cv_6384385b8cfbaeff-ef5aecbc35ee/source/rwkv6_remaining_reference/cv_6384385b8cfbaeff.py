
from opbench.fla_fixtures import load_fixture
from yb_3adaaa4f22c6e4 import *

def test_original_cpu_absolute():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_rwkv6_bwd_kernel_dh/test_chunk_rwkv6_bwd_kernel_dh_v2.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data = data['input_data']
    grid = data['grid']
    npu_input = input_data.copy()
    for k, v in npu_input.items():
        if isinstance(v, torch.Tensor):
            npu_input[k] = v.to('npu')
    chunk_rwkv6_bwd_kernel_dh[grid](q=npu_input['q'], gi=npu_input['gi'], ge=npu_input['ge'], do=npu_input['do'], dh=npu_input['dh'], dht=npu_input['dht'], dh0=npu_input['dh0'], cu_seqlens=npu_input['cu_seqlens'] if npu_input['cu_seqlens'].numel() > 0 else None, chunk_offsets=npu_input['chunk_offsets'] if npu_input['chunk_offsets'].numel() > 0 else None, scale=npu_input['scale'], T=npu_input['T'], HQ=npu_input['HQ'], H=npu_input['H'], K=npu_input['K'], V=npu_input['V'], BT=npu_input['BT'], BK=npu_input['BK'], BV=npu_input['BV'], NG=npu_input['NG'])
    reference = load_fixture('generated-data/original_cpu_reference.pt', map_location='cpu', weights_only=False)
    torch.testing.assert_close(npu_input['dh'].detach().cpu().float(), reference['dh'].cpu(), rtol=0.01, atol=0.01)
    torch.testing.assert_close(npu_input['dh0'].detach().cpu().float(), reference['dh0'].cpu(), rtol=0.01, atol=0.1)


from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_93f71b12d34293 import chunk_dplr_bwd_dgk_kernel
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type
os.environ['FLA_USE_FAST_OPS'] = '0'
import triton

def chunk_dplr_bwd_dqk_intra(dgk: torch.Tensor, dgk_offset: torch.Tensor, dgk_last: torch.Tensor, cu_seqlens: torch.LongTensor | None=None, chunk_size: int=64, chunk_indices: torch.LongTensor | None=None):
    B, T, H, K = dgk.shape
    BT = chunk_size
    dgk_output = torch.empty_like(dgk)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)

    def grid(meta):
        return (NT, triton.cdiv(K, meta['BK']), B * H)
    chunk_dplr_bwd_dgk_kernel[grid](dgk=dgk, dgk_offset=dgk_offset, dgk_last=dgk_last, dgk_output=dgk_output, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, K=K, BT=BT)
    return dgk_output

@pytest.mark.functiontest
def test_chunk_dplr_bwd_dgk_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_chunk_dplr_bwd_dgk_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    dgk_output = chunk_dplr_bwd_dqk_intra(dgk=input_data['dgk'], dgk_offset=input_data['dgk_offset'], dgk_last=input_data['dgk_last'], cu_seqlens=input_data['cu_seqlens'], chunk_size=input_data['chunk_size'], chunk_indices=input_data['chunk_indices'])
    npu_out = {'dgk_output': dgk_output}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test chunk_dplr_bwd_dgk_kernel passed!')

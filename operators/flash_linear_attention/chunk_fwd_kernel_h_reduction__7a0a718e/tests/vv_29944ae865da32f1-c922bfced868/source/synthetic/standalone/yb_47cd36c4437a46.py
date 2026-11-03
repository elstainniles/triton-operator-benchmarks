
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import triton
import torch_npu
from yb_7a0a718e4486ff import chunk_fwd_kernel_h_reduction
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

def chunk_fwd_h_reduction(g: torch.Tensor, gk: torch.Tensor, gv: torch.Tensor, hs: torch.Tensor, hr: torch.Tensor, output_final_state: bool, cu_seqlens: torch.Tensor | None=None, split_offsets: torch.Tensor | None=None, chunk_size: int=64, split_size: int=256) -> torch.Tensor:
    NS, H, K, V = hs.shape
    BT = chunk_size
    S = split_size
    if cu_seqlens is None:
        N = NS // triton.cdiv(1024, S)
        T = 1024
    else:
        N = len(cu_seqlens) - 1
        T = (cu_seqlens[-1] - cu_seqlens[0]).item()
    ht = hs.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None

    def grid(meta):
        return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_fwd_kernel_h_reduction[grid](g=g, gk=gk, gv=gv, hs=hs, hr=hr, ht=ht, cu_seqlens=cu_seqlens, split_offsets=split_offsets, T=T, S=S, H=H, K=K, V=V, BT=BT, USE_G=g is not None, USE_GK=gk is not None, USE_GV=gv is not None)
    return ht

@pytest.mark.functiontest
def test_chunk_fwd_kernel_h_reduction(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_chunk_h_split_chunk_fwd_kernel_h_reduction_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    ht = chunk_fwd_h_reduction(g=input_data['g'], gk=input_data['gk'], gv=input_data['gv'], hs=input_data['hs'], hr=input_data['hr'], output_final_state=input_data['output_final_state'], cu_seqlens=input_data['cu_seqlens'], split_offsets=input_data['split_offsets'], chunk_size=input_data['chunk_size'], split_size=input_data['split_size'])
    npu_out = {'ht': ht.cpu() if ht is not None else None, 'hr': input_data['hr'].cpu()}
    del npu_out['ht']
    del data['gpu_output']['ht']
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Testcase Passed')

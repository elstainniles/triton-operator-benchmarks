
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_5a9aa3bd6a5163 import chunk_oja_fwd_h
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'

@pytest.mark.functiontest
def test_chunk_oja_fwd_kernel_h_blockdim64_function():
    B, T, H, D = (4, 2048, 8, 128)
    K, V = (D, D)
    BT = 64
    dtype = torch.float32
    kernel_name = 'chunk_oja_fwd_kernel_h_blockdim64'
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_fwd_kernel_h_blockdim64'
    print('Loading inputs from pt file...')
    inputs = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    v = inputs['v'].to(DEVICE)
    w = inputs['w'].to(DEVICE)
    u = inputs['u'].to(DEVICE)
    gv = inputs['gv'].to(DEVICE)
    initial_state = inputs['initial_state'].to(DEVICE)
    output_final_state = inputs['output_final_state']
    chunk_size = inputs['chunk_size']
    h, k_new, final_state = chunk_oja_fwd_h(v, w, u, gv=gv, initial_state=initial_state, output_final_state=output_final_state, chunk_size=chunk_size)
    torch.npu.synchronize()
    expected_outputs = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    expected_h = expected_outputs['h'].to(DEVICE)
    expected_k_new = expected_outputs['k_new'].to(DEVICE)
    expected_final_state = expected_outputs['final_state'].to(DEVICE)
    torch.testing.assert_close(h, expected_h, atol=1e-05, rtol=1e-05)
    print('h precision check passed')
    torch.testing.assert_close(k_new, expected_k_new, atol=1e-05, rtol=1e-05)
    print('k_new precision check passed')
    torch.testing.assert_close(final_state, expected_final_state, atol=1e-05, rtol=1e-05)
    print('final_state precision check passed')
    print(f'Precision check completed for {kernel_name}')


from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'chunk_oja_bwd_kernel_dqk'

def get_precision_thresholds(dtype):
    return (0.008, 0.008)

@pytest.mark.functiontest
def test_chunk_oja_bwd_kernel_dqk_simple_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dqk'
    from yb_8b1f9f64c8c620 import chunk_oja_bwd_dqk
    q = load_fixture(os.path.join(load_dir, 'q.pt')).to(DEVICE)
    k_new = load_fixture(os.path.join(load_dir, 'k_new.pt')).to(DEVICE)
    h = load_fixture(os.path.join(load_dir, 'h.pt')).to(DEVICE)
    gv_cumsum = load_fixture(os.path.join(load_dir, 'gv_cumsum.pt')).to(DEVICE)
    dA = load_fixture(os.path.join(load_dir, 'dA.pt')).to(DEVICE)
    do = load_fixture(os.path.join(load_dir, 'do.pt')).to(DEVICE)
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    A_qk, dq, dk = chunk_oja_bwd_dqk(q=q, k=k_new, h=h, gv=gv_cumsum, dA=dA, do=do, scale=scale)
    expected_A_qk = load_fixture(os.path.join(load_dir, 'output_0.pt')).to(DEVICE)
    expected_dq = load_fixture(os.path.join(load_dir, 'output_1.pt')).to(DEVICE)
    expected_dk = load_fixture(os.path.join(load_dir, 'output_2.pt')).to(DEVICE)
    atol_A_qk, rtol_A_qk = get_precision_thresholds(A_qk.dtype)
    torch.testing.assert_close(A_qk, expected_A_qk, atol=atol_A_qk, rtol=rtol_A_qk)
    print('Output 0 (A_qk) precision check passed')
    atol_dq, rtol_dq = get_precision_thresholds(dq.dtype)
    torch.testing.assert_close(dq, expected_dq, atol=atol_dq, rtol=rtol_dq)
    print('Output 1 (dq) precision check passed')
    atol_dk, rtol_dk = get_precision_thresholds(dk.dtype)
    torch.testing.assert_close(dk, expected_dk, atol=atol_dk, rtol=rtol_dk)
    print('Output 2 (dk) precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')

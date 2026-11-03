
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
def test_chunk_oja_bwd_kernel_dqk_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dqk/'
    from yb_8b1f9f64c8c620 import chunk_oja_bwd_dqk
    q = load_fixture(os.path.join(load_dir, 'q.pt')).to(DEVICE)
    k_new = load_fixture(os.path.join(load_dir, 'k_new.pt')).to(DEVICE)
    h = load_fixture(os.path.join(load_dir, 'h.pt')).to(DEVICE)
    gv_cumsum = load_fixture(os.path.join(load_dir, 'gv_cumsum.pt')).to(DEVICE)
    dA = load_fixture(os.path.join(load_dir, 'dA.pt')).to(DEVICE)
    do = load_fixture(os.path.join(load_dir, 'do.pt')).to(DEVICE)
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    chunk_size = load_fixture(os.path.join(load_dir, 'chunk_size.pt')).item()
    output = chunk_oja_bwd_dqk(q, k_new, h, gv_cumsum, dA, do, scale, None, None, chunk_size)
    if isinstance(output, tuple):
        for i, out in enumerate(output):
            expected_out = load_fixture(os.path.join(load_dir, f'output_{i}.pt')).to(DEVICE)
            atol, rtol = get_precision_thresholds(out.dtype)
            torch.testing.assert_close(out, expected_out, atol=atol, rtol=rtol)
            print(f'Output {i} precision check passed')
    else:
        expected_output = load_fixture(os.path.join(load_dir, 'output.pt')).to(DEVICE)
        atol, rtol = get_precision_thresholds(output.dtype)
        torch.testing.assert_close(output, expected_output, atol=atol, rtol=rtol)
        print('Output precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')
